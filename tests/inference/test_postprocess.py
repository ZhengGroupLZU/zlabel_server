"""Post-processing tests (NMS, mask upscaling, contour helpers)."""

from __future__ import annotations

import numpy as np
import pytest

from inference.sam_ort.postprocess import (
    contour_filter,
    nms,
    nms_filter,
    offset_polygon,
    pcs_filter_nms,
    pcs_scores,
    reduce_contour_points,
    smooth_contour,
    upscale_mask,
    upscale_mask_pad,
    xywh2xyxy,
)


class TestNms:
    def test_keeps_best_of_overlapping(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], np.float32)
        scores = np.array([0.5, 0.9, 0.8], np.float32)
        keep = nms(boxes, scores, 0.5)
        # box1 (score 0.9) suppresses box0; box2 kept
        assert set(keep.tolist()) == {1, 2}

    def test_all_distinct_kept(self):
        boxes = np.array([[0, 0, 10, 10], [100, 0, 110, 10]], np.float32)
        keep = nms(boxes, np.array([0.5, 0.6]), 0.5)
        assert len(keep) == 2

    def test_empty(self):
        keep = nms(np.zeros((0, 4)), np.zeros(0), 0.5)
        assert len(keep) == 0


class TestNmsFilter:
    def test_xywh_overlap(self):
        boxes = [(0, 0, 10, 10), (1, 1, 10, 10)]  # x, y, w, h
        keep = nms_filter(boxes, 0.5)
        assert len(keep) == 1

    def test_empty(self):
        assert nms_filter([], 0.5) == []


class TestPcs:
    def test_xywh2xyxy(self):
        boxes = np.array([[10, 20, 4, 6]], np.float32)  # cx, cy, w, h
        xyxy = xywh2xyxy(boxes)
        np.testing.assert_allclose(xyxy[0], [8, 17, 12, 23])

    def test_pcs_scores(self):
        logits = np.array([[2.0], [-2.0], [0.0]], np.float32)
        presence = np.array([[1.0]], np.float32)
        scores = pcs_scores(logits, presence)

        def s(x):
            return 1.0 / (1.0 + np.exp(-x))

        np.testing.assert_allclose(scores[0], [s(2) * s(1), s(-2) * s(1), s(0) * s(1)], atol=1e-4)

    def test_pcs_filter_nms(self):
        # 2 candidates, one above conf
        boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.05, 0.05]], np.float32)
        masks = np.zeros((1, 2, 288, 288), np.float32)
        scores = np.array([[0.9], [0.1]], np.float32)
        out_masks, out_boxes, out_scores, cls = pcs_filter_nms(boxes, masks, scores, conf=0.25, iou=0.7)
        assert len(out_scores) == 1
        assert out_scores[0] == pytest.approx(0.9)

    def test_pcs_filter_nms_all_below_conf(self):
        boxes = np.zeros((1, 200, 4), np.float32)
        masks = np.zeros((1, 200, 288, 288), np.float32)
        scores = np.full((1, 200), 0.1, np.float32)
        out_masks, out_boxes, out_scores, cls = pcs_filter_nms(boxes, masks, scores, conf=0.25, iou=0.7)
        assert len(out_scores) == 0


class TestUpscaleMask:
    def test_plain_resize(self):
        mask = np.zeros((288, 288), np.float32)
        mask[100:200, 100:200] = 1.0
        up = upscale_mask(mask, (720, 1280))
        assert up.shape == (720, 1280)
        assert up.dtype == np.uint8
        assert up.max() == 1 and up.min() == 0
        # center region positive
        assert up[360, 640] > 0

    def test_threshold(self):
        mask = np.full((288, 288), -0.5, np.float32)
        up = upscale_mask(mask, (720, 1280))
        assert (up == 0).all()


class TestUpscaleMaskPad:
    def test_crop_then_resize(self):
        # 720x1280 -> letterbox 1024x1024 (content 1024x576, pad bottom 448)
        mask = np.zeros((1024, 1024), np.float32)
        mask[:576, :] = 1.0  # content area positive, padding negative
        up = upscale_mask_pad(mask, (720, 1280))
        assert up.shape == (720, 1280)
        assert up.dtype == np.uint8
        # padding cropped: full mask should be positive
        assert (up > 0).all()

    def test_padding_fraction_matches_letterbox(self, img_bgr):
        # build the mask from a letterboxed image region, then verify the crop math
        from inference.sam_ort.preprocess import preprocess_sam_letterbox

        tensor, r = preprocess_sam_letterbox(img_bgr, 1024)
        content_h = round(img_bgr.shape[0] * r)
        content_w = round(img_bgr.shape[1] * r)
        # mask with positive content region and negative padding
        mask = np.full((1024, 1024), -1.0, np.float32)
        mask[:content_h, :content_w] = 1.0
        up = upscale_mask_pad(mask, img_bgr.shape[:2])
        assert up.shape == img_bgr.shape[:2]
        assert (up > 0).all()

    def test_already_target_size(self):
        mask = np.full((100, 100), 1.0, np.float32)
        up = upscale_mask_pad(mask, (100, 100))
        assert (up > 0).all()


class TestContourHelpers:
    def test_contour_filter(self):
        import cv2

        img = np.zeros((1000, 1000), np.uint8)
        img[100:200, 100:200] = 255  # 100x100 = 1% of image area
        img[600:610, 600:610] = 255  # tiny
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kept = contour_filter(contours, min_area_ratio=0.005, img_shape=(1000, 1000))
        assert len(kept) == 1

    def test_reduce_contour_points(self):
        # a 360-gon: approxPolyDP should reduce the point count substantially
        angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        contour = (
            np.stack([200 * np.cos(angles), 200 * np.sin(angles)], axis=1).astype(np.int32).reshape(-1, 1, 2)
        )
        simplified = reduce_contour_points(contour, min_points=5, max_points=100)
        assert 4 <= len(simplified) <= 100

    def test_smooth_contour_resamples_and_smooths_in_pixels(self):
        # a jagged square outline: 2 px steps, sharp corners
        contour = np.array(
            [[0, 0], [0, 4], [0, 8], [4, 8], [8, 8], [8, 4], [8, 0], [4, 0]], np.float32
        ).reshape(-1, 2)
        smoothed = smooth_contour(contour, sigma_px=1.5)
        assert smoothed.dtype == np.float32  # subpixel: no snapping back to the grid
        assert smoothed.ndim == 2 and smoothed.shape[1] == 2
        # resampled at ~1 px of arc length, so the outline keeps its perimeter
        perimeter = float(np.hypot(*(np.roll(smoothed, -1, axis=0) - smoothed).T).sum())
        assert 16 < perimeter < 40
        # smoothing pulls a sharp corner in, but stays inside the shape's box (+/- 1 px)
        assert smoothed.min() >= -1.0 and smoothed.max() <= 9.0
        assert np.isfinite(smoothed).all()

    def test_smooth_contour_keeps_a_smooth_curve_in_place(self):
        angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        circle = np.column_stack([100 + 40 * np.cos(angles), 100 + 40 * np.sin(angles)]).astype(np.float32)
        smoothed = smooth_contour(circle, sigma_px=1.5)
        radius = np.hypot(smoothed[:, 0] - 100, smoothed[:, 1] - 100)
        assert abs(float(radius.mean()) - 40) < 0.6  # no shrinkage of an already smooth edge

    def test_smooth_contour_survives_a_tiny_contour(self):
        """A degenerate 3-point contour is resampled but must not become a blob."""
        import cv2

        contour = np.array([[0, 0], [10, 0], [10, 10]], np.float32)
        smoothed = smooth_contour(contour, sigma_px=1.5)
        assert smoothed.shape[1] == 2 and len(smoothed) >= 3
        assert abs(cv2.contourArea(smoothed) - cv2.contourArea(contour)) < 0.35 * cv2.contourArea(contour)

    def test_offset_polygon_restores_the_pixel_centre_area(self):
        import cv2

        mask = np.zeros((200, 200), np.uint8)
        cv2.circle(mask, (100, 100), 45, 255, -1)
        truth = float(np.count_nonzero(mask))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = smooth_contour(max(contours, key=cv2.contourArea), sigma_px=1.5)
        bare = abs(cv2.contourArea(pts))
        expanded = abs(cv2.contourArea(offset_polygon(pts, 0.5)))
        assert bare < truth  # pixel centres are ~perimeter/2 short of the raster area
        assert abs(expanded / truth - 1) < 0.01  # half a pixel outward puts it back
