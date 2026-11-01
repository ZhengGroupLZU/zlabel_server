"""Post-processing for the ONNX models (NMS, mask upscaling, contour handling)."""

from __future__ import annotations

import cv2
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    x1 = boxes[..., 0] - boxes[..., 2] / 2
    y1 = boxes[..., 1] - boxes[..., 3] / 2
    x2 = boxes[..., 0] + boxes[..., 2] / 2
    y2 = boxes[..., 1] + boxes[..., 3] / 2
    return np.stack([x1, y1, x2, y2], axis=-1)


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Class-agnostic NMS. Returns keep indices (desc score)."""
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes_xyxy[i, 0], boxes_xyxy[rest, 0])
        yy1 = np.maximum(boxes_xyxy[i, 1], boxes_xyxy[rest, 1])
        xx2 = np.minimum(boxes_xyxy[i, 2], boxes_xyxy[rest, 2])
        yy2 = np.minimum(boxes_xyxy[i, 3], boxes_xyxy[rest, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes_xyxy[i, 2] - boxes_xyxy[i, 0]) * (boxes_xyxy[i, 3] - boxes_xyxy[i, 1])
        area_r = (boxes_xyxy[rest, 2] - boxes_xyxy[rest, 0]) * (boxes_xyxy[rest, 3] - boxes_xyxy[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[iou <= iou_thresh]
    return np.asarray(keep, dtype=np.int64)


def pcs_scores(pred_logits, presence_logit):
    scores = sigmoid(pred_logits)[..., 0]
    presence = sigmoid(presence_logit)
    return scores * presence


def pcs_filter_nms(pred_boxes_xywh, pred_masks, scores, conf=0.25, iou=0.7, agnostic_nms=False):
    """SAM3 PCS postprocess on normalized box outputs. Returns (masks, boxes_xyxy_norm, score, cls)."""
    n_masks = pred_masks.shape[-1]
    pred_masks = pred_masks.reshape(-1, pred_masks.shape[-2], n_masks)
    pred_boxes_xywh = pred_boxes_xywh.reshape(-1, 4)
    scores = scores.reshape(-1)
    nq = pred_boxes_xywh.shape[0]
    keep = scores > conf
    if not np.any(keep):
        return (
            np.zeros((0, *pred_masks.shape[1:]), np.float32),
            np.zeros((0, 4), np.float32),
            np.zeros((0,), np.float32),
            np.zeros((0,), np.int64),
        )
    masks = pred_masks[keep]
    boxes = pred_boxes_xywh[keep]
    sc = scores[keep]
    cls = np.nonzero(keep)[0] // nq
    boxes_xyxy = xywh2xyxy(boxes)
    off = 0.0 if agnostic_nms else 7680.0
    keep2 = nms(boxes_xyxy + cls[:, None] * off, sc, iou)
    return masks[keep2], boxes_xyxy[keep2], sc[keep2], cls[keep2]


def upscale_mask(
    mask_logits: np.ndarray, target_hw: tuple[int, int], mask_threshold: float = 0.0
) -> np.ndarray:
    """Bilinear-upscale a (S,S) mask to (h,w) and threshold to binary (stretched inputs)."""
    m = mask_logits.astype(np.float32)
    if m.shape != target_hw:
        m = cv2.resize(m, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
    return (m > mask_threshold).astype(np.uint8)


def upscale_mask_pad(
    mask_logits: np.ndarray, target_hw: tuple[int, int], mask_threshold: float = 0.0
) -> np.ndarray:
    """Upscale a mask from letterboxed space back to the original image (crop then resize).

    Mirrors ``ultralytics.utils.ops.scale_masks(padding=False)``: crop the right/bottom
    padding of the letterboxed image, then bilinear-resize to the original shape.
    """
    m = mask_logits.astype(np.float32)
    H, W = target_hw
    gain = min(m.shape[0] / H, m.shape[1] / W)
    pad_w = m.shape[1] - round(W * gain)
    pad_h = m.shape[0] - round(H * gain)
    crop = m[: m.shape[0] - pad_h, : m.shape[1] - pad_w]
    if crop.shape[0] > 0 and crop.shape[1] > 0:
        m = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        m = np.zeros((H, W), np.float32)
    return (m > mask_threshold).astype(np.uint8)


def contour_filter(contours, min_area_ratio: float, img_shape: tuple[int, int]) -> list:
    """Drop contours smaller than min_area_ratio * image area."""
    img_area = img_shape[0] * img_shape[1]
    min_area = img_area * min_area_ratio
    return [c for c in contours if cv2.contourArea(c) >= min_area]


#: default smoothing strength, in *pixels* along the contour (not in contour points:
#: an index-window smooth blurs a 60-point seed and a 600-point root differently)
CONTOUR_SMOOTH_SIGMA = 1.5
#: arc-length resampling step, in pixels
CONTOUR_STEP = 1.0
#: outward offset applied after smoothing. ``findContours`` traces *pixel centres*, so a
#: polygon through them is short of the rasterised area by about perimeter/2; half a pixel
#: outward puts it back (measured: a disc goes from -2.3% to -0.1% area error).
CONTOUR_OFFSET = 0.5


def resample_closed(pts: np.ndarray, step: float = CONTOUR_STEP) -> np.ndarray:
    """Resample a closed polyline at a uniform arc-length step (float32 (N, 2)).

    Smoothing has to happen *along the curve*, not along the vertex list: the vertices of
    a ``CHAIN_APPROX_SIMPLE`` contour are spaced by pixel distance, so an index window
    would blur long straight parts more than short wiggly ones.
    """
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    n = len(pts)
    if n < 3:
        return pts
    nxt = np.roll(pts, -1, axis=0)
    seg = np.hypot(*(nxt - pts).T)
    perimeter = float(seg.sum())
    if perimeter <= 0:
        return pts
    counts = max(8, int(round(perimeter / max(step, 1e-3))))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    closed = np.vstack([pts, pts[:1]])
    targets = np.linspace(0.0, perimeter, counts, endpoint=False)
    return np.column_stack([np.interp(targets, cum, closed[:, 0]), np.interp(targets, cum, closed[:, 1])]).astype(
        np.float32
    )


def smooth_closed(pts: np.ndarray, sigma_px: float, step: float = CONTOUR_STEP) -> np.ndarray:
    """Circular Gaussian smoothing of the contour coordinates, ``sigma`` in pixels."""
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    if sigma_px <= 0 or len(pts) < 4:
        return pts
    sigma = max(0.3, float(sigma_px) / max(step, 1e-3))
    radius = max(1, int(np.ceil(3.0 * sigma)))
    idx = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (idx / sigma) ** 2)
    kernel /= kernel.sum()
    out = np.zeros_like(pts)
    for k, off in enumerate(idx):
        out += kernel[k] * np.roll(pts, -int(off), axis=0)
    return out.astype(np.float32)


def offset_polygon(pts: np.ndarray, dist: float = CONTOUR_OFFSET) -> np.ndarray:
    """Push every vertex along the outward normal (see ``CONTOUR_OFFSET``)."""
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    if not dist or len(pts) < 3:
        return pts
    tan = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    norm = np.column_stack([-tan[:, 1], tan[:, 0]])
    length = np.hypot(norm[:, 0], norm[:, 1])
    length[length == 0] = 1.0
    unit = norm / length[:, None]
    # Which sign is "outward" depends on the winding order the caller produced, so decide
    # by measuring instead of assuming: keep the side that grows the enclosed area.
    outward = (pts + dist * unit).astype(np.float32)
    inward = (pts - dist * unit).astype(np.float32)
    return outward if abs(cv2.contourArea(outward)) >= abs(cv2.contourArea(inward)) else inward


def smooth_contour(contour: np.ndarray, sigma_px: float = CONTOUR_SMOOTH_SIGMA) -> np.ndarray:
    """Uniformly resample a closed contour and Gaussian-smooth it: (N, 2) float32.

    Replaces the old moving average over a fixed *number of points*, whose blur depended on
    how densely the contour happened to be sampled, and its integer output, which snapped
    the polygon back onto the pixel grid it was being smoothed away from.
    """
    pts = np.asarray(contour, np.float32).reshape(-1, 2)
    return smooth_closed(resample_closed(pts, CONTOUR_STEP), sigma_px, CONTOUR_STEP)


def reduce_contour_points(
    pts: np.ndarray,
    max_points: int = 100,
    min_points: int = 5,
    epsilon_px: float = 0.0,
    max_iterations: int = 12,
) -> np.ndarray:
    """Douglas-Peucker with a tolerance in *pixels*; only used to cap the vertex count.

    ``epsilon_px`` > 0 simplifies every contour by that tolerance. With 0 the shape is kept
    as smoothed and the tolerance is raised only as far as needed to get under
    ``max_points`` - the old version always aimed for ~30% of the points with a tolerance
    up to 10 px, which replaced smooth arcs with long straight chords.
    """
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return pts
    poly = pts.reshape(-1, 1, 2)
    if epsilon_px > 0:
        poly = cv2.approxPolyDP(poly, float(epsilon_px), True)
    if len(poly) > max(max_points, min_points):
        low, high = 1e-3, 512.0
        for _ in range(max(1, max_iterations)):
            mid = (low + high) / 2.0
            if len(cv2.approxPolyDP(poly, mid, True)) > max_points:
                low = mid
            else:
                high = mid
        poly = cv2.approxPolyDP(poly, high, True)
    out = poly.reshape(-1, 2).astype(np.float32)
    return out if len(out) >= 3 else pts


def nms_filter(boxes: list[tuple[float, float, float, float]], iou_threshold: float = 0.5) -> list[int]:
    """NMS over xywh boxes; returns kept indices."""
    if not boxes:
        return []
    boxes_xyxy = np.asarray([[x, y, x + w, y + h] for x, y, w, h in boxes], np.float32)
    areas = boxes_xyxy[:, 2] * boxes_xyxy[:, 3]
    order = np.argsort(-areas)
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes_xyxy[i, 0], boxes_xyxy[rest, 0])
        yy1 = np.maximum(boxes_xyxy[i, 1], boxes_xyxy[rest, 1])
        xx2 = np.minimum(boxes_xyxy[i, 2], boxes_xyxy[rest, 2])
        yy2 = np.minimum(boxes_xyxy[i, 3], boxes_xyxy[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-9)
        order = rest[iou <= iou_threshold]
    return keep
