"""Model runners: unified set_image + segment_points/segment_box (and SAM3 text) over ONNXRuntime.

- SamRunner: SAM / EdgeSAM (stretched input) and SlimSAM (letterboxed input)
- Sam2Runner: SAM2 (encoder -> decoder), letterboxed input
- Sam3Runner: SAM3 (vision_encoder + pvs for interactive, text/geometry/detector for PCS)

Preprocessing mirrors the ZLabel desktop pipeline:
- SAM / EdgeSAM: direct stretch to the square encoder input, prompts scaled per-axis,
  masks resized directly back to the original image.
- SlimSAM / SAM2: ``LetterBox(imgsz, auto=False, center=False)`` (min-ratio scale, 114 pad
  right/bottom), prompts scaled by the same ratio, masks un-padded (crop) and resized back.
- SAM3: stretch for PCS, letterbox for PVS (separately cached vision features).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.sam_ort.backends import OrtSession
from app.sam_ort.postprocess import (
    pcs_filter_nms,
    pcs_scores,
    upscale_mask,
    upscale_mask_pad,
)
from app.sam_ort.preprocess import preprocess_sam, preprocess_sam3, preprocess_sam_letterbox
from app.sam_ort.tokenize import SimpleTokenizer
from app.ztypes import PvsResult, SamOnnxResult

SAM_IMG = 1024
SAM3_IMG = 1008
SAM3_HW = 72 * 72
SAM3_CTX = 32
SAM3_GEO_MAX_BOXES = 4
SAM3_PVS_MAX_POINTS = 4


def _box_to_points(box_xyxy_px) -> tuple[list[tuple[float, float]], list[float]]:
    x1, y1, x2, y2 = (float(v) for v in box_xyxy_px)
    return [(x1, y1), (x2, y2)], [2.0, 3.0]


class _SamDecoderMixin:
    """Shared prompt->decoder logic for SAM-family (encoder -> image_embeddings -> decoder)."""

    def __init__(self, letterbox: bool = False, img_size: int = SAM_IMG, backend: str = "CPU", threads: int = 4):
        self._letterbox = letterbox
        self._img_size = img_size
        self._backend = backend
        self._threads = threads
        self._ratio = 1.0
        self._orig_shape: tuple[int, int] | None = None

    def _transform_points(self, points, labels):
        """Scale pixel coords into the encoder input space."""
        pts = np.asarray(points, np.float32)
        lbs = np.asarray(labels, np.float32)
        if (lbs == 2).sum() == 0:
            onnx_coord = np.concatenate([pts, np.zeros((1, 2), np.float32)])[None]
            onnx_label = np.append(lbs, -1.0)[None].astype(np.float32)
        else:
            onnx_coord = pts[None]
            onnx_label = lbs[None].astype(np.float32)
        onnx_coord = onnx_coord.copy()
        if self._letterbox:
            onnx_coord *= self._ratio
        else:
            h, w = self._orig_shape
            onnx_coord[..., 0] *= self._img_size / w
            onnx_coord[..., 1] *= self._img_size / h
        return onnx_coord, onnx_label

    def _new_decoder(self) -> OrtSession:
        if self._decoder is None:
            self._decoder = OrtSession(self._decoder_path, backend=self._backend, threads=self._threads)
        return self._decoder

    def _run_decoder(self, point_coords, point_labels) -> list[SamOnnxResult]:
        decoder = self._new_decoder()
        feeds: dict[str, np.ndarray] = {}
        for name in decoder.input_names:
            if name == "image_embeddings":
                feeds[name] = self._image_embeddings
            elif name == "point_coords":
                feeds[name] = point_coords
            elif name == "point_labels":
                feeds[name] = point_labels
            elif name == "mask_input":
                feeds[name] = np.zeros((1, 1, 256, 256), np.float32)
            elif name == "has_mask_input":
                feeds[name] = np.zeros((1,), np.float32)
            elif name == "orig_im_size":
                # SAM/SlimSAM decoders resize masks to orig_im_size. For letterboxed
                # input feed the padded size and crop the padding back afterwards
                # (upscale_mask_pad); for stretched input feed the original size so the
                # decoder resizes directly (upscale_mask then becomes a no-op).
                feeds[name] = np.array(
                    self._orig_shape if not self._letterbox else [self._img_size, self._img_size], np.float32
                )
            else:
                raise ValueError(f"unexpected SAM decoder input: {name}")
        out = decoder.run(feeds)
        masks = out["masks"][0]  # (N, 1024, 1024) for SAM/SlimSAM, (N, 256, 256) for EdgeSAM
        scores = out.get("scores", out.get("iou_predictions"))[0]  # (N,)
        results = []
        for mask, score in zip(masks, scores):
            up = self._upscale(mask)
            results.append(SamOnnxResult(mask=up.astype(np.float32), score=float(score)))
        return sorted(results, key=lambda r: r.score, reverse=True)

    def _upscale(self, mask: np.ndarray) -> np.ndarray:
        if self._letterbox:
            return upscale_mask_pad(mask, self._orig_shape)
        return upscale_mask(mask, self._orig_shape)

    def segment_points(self, points, labels) -> list[SamOnnxResult]:
        coord, lbl = self._transform_points(points, labels)
        return self._run_decoder(coord, lbl)

    def segment_box(self, box_xyxy_px) -> list[SamOnnxResult]:
        pts, lbs = _box_to_points(box_xyxy_px)
        return self.segment_points(pts, lbs)


class SamRunner(_SamDecoderMixin):
    """SAM / EdgeSAM / SlimSAM: encoder -> image_embeddings -> decoder."""

    def __init__(self, encoder_path: str | Path, decoder_path: str | Path, img_size: int = SAM_IMG,
                 letterbox: bool = False, backend: str = "CPU", threads: int = 4):
        super().__init__(letterbox=letterbox, img_size=img_size, backend=backend, threads=threads)
        self._encoder_path = str(encoder_path)
        self._decoder_path = str(decoder_path)
        self._encoder = OrtSession(self._encoder_path, backend=backend, threads=threads)
        self._decoder: OrtSession | None = None
        self._image_embeddings: np.ndarray | None = None

    def set_image(self, image_bgr: np.ndarray):
        self._orig_shape = image_bgr.shape[:2]
        in_name = self._encoder.input_names[0]
        if self._letterbox:
            tensor, self._ratio = preprocess_sam_letterbox(image_bgr, self._img_size)
        else:
            tensor = preprocess_sam(image_bgr, self._img_size)
        self._image_embeddings = self._encoder.run({in_name: tensor})["image_embeddings"]


class Sam2Runner(_SamDecoderMixin):
    """SAM2: encoder -> high_res_feats_0/1 + image_embed -> decoder."""

    def __init__(self, encoder_path: str | Path, decoder_path: str | Path, img_size: int = SAM_IMG,
                 backend: str = "CPU", threads: int = 4):
        super().__init__(letterbox=True, img_size=img_size, backend=backend, threads=threads)
        self._encoder_path = str(encoder_path)
        self._decoder_path = str(decoder_path)
        self._encoder = OrtSession(self._encoder_path, backend=backend, threads=threads)
        self._decoder: OrtSession | None = None
        self._image_embed: np.ndarray | None = None
        self._hr0: np.ndarray | None = None
        self._hr1: np.ndarray | None = None

    def set_image(self, image_bgr: np.ndarray):
        self._orig_shape = image_bgr.shape[:2]
        tensor, self._ratio = preprocess_sam_letterbox(image_bgr, self._img_size)
        out = self._encoder.run({"image": tensor})
        self._image_embed, self._hr0, self._hr1 = out["image_embed"], out["high_res_feats_0"], out["high_res_feats_1"]

    def _run_decoder(self, point_coords, point_labels) -> list[SamOnnxResult]:
        decoder = self._new_decoder()
        feeds: dict[str, np.ndarray] = {}
        for name in decoder.input_names:
            if name == "image_embed":
                feeds[name] = self._image_embed
            elif name == "high_res_feats_0":
                feeds[name] = self._hr0
            elif name == "high_res_feats_1":
                feeds[name] = self._hr1
            elif name == "point_coords":
                feeds[name] = point_coords
            elif name == "point_labels":
                feeds[name] = point_labels
            elif name == "mask_input":
                feeds[name] = np.zeros((1, 1, 256, 256), np.float32)
            elif name == "has_mask_input":
                feeds[name] = np.zeros((1,), np.float32)
            else:
                raise ValueError(f"unexpected SAM2 decoder input: {name}")
        out = decoder.run(feeds)
        masks = out["masks"][0]  # (N, 256, 256)
        scores = out["iou_predictions"][0]
        results = []
        for mask, score in zip(masks, scores):
            up = self._upscale(mask)
            results.append(SamOnnxResult(mask=up.astype(np.float32), score=float(score)))
        return sorted(results, key=lambda r: r.score, reverse=True)


class Sam3Runner:
    """SAM3: vision_encoder + pvs (interactive points) and text/geometry/detector (PCS)."""

    def __init__(
        self,
        model_dir: str | Path,
        conf: float = 0.25,
        iou: float = 0.7,
        backend: str = "CPU",
        threads: int = 4,
    ):
        d = Path(model_dir)
        self.conf = conf
        self.iou = iou
        self._backend = backend
        self._threads = threads
        # the vision encoder's CUDA arena holds ~7GB after one forward (fp32 weights
        # + activations) and never shrinks; keeping it resident together with the
        # detector OOMs 12GB GPUs, so the vision session is created per encode and
        # released afterwards (see _run_vision)
        self._vision_path = str(d / "sam3_vision_encoder.onnx")
        self._pvs_model = OrtSession(str(d / "sam3_pvs.onnx"), backend=backend, threads=threads)
        # the text encoder is a 1.4GB fp32 model whose compute is trivial; keep it on
        # CPU to spare GPU memory for the detector
        self._text_enc_model = OrtSession(str(d / "sam3_text_encoder.onnx"), backend="CPU", threads=threads)
        self._geo_enc_model = OrtSession(str(d / "sam3_geometry_encoder.onnx"), backend=backend, threads=threads)
        self._detector_model = OrtSession(str(d / "sam3_detector.onnx"), backend=backend, threads=threads)
        self._tokenizer = None
        if (d / "vocab.json").exists() and (d / "merges.txt").exists():
            self._tokenizer = SimpleTokenizer.default(d)
        self._orig_shape: tuple[int, int] | None = None
        self._pcs_feats: dict | None = None
        self._pvs_feats: dict | None = None

    def set_image(self, image_bgr: np.ndarray):
        self._img = image_bgr
        self._orig_shape = image_bgr.shape[:2]
        self._pcs_feats = None
        self._pvs_feats = None

    def _run_vision(self, im: np.ndarray) -> dict:
        # transient session: created per encode and released afterwards so its CUDA
        # arena memory is freed before the detector runs (see __init__)
        vision = OrtSession(self._vision_path, backend=self._backend, threads=self._threads)
        out = vision.run({"image": im})
        del vision
        fpn2 = out["fpn2"]
        pos2 = out["pos2"]
        return {
            "fpn0": out["fpn0"],
            "fpn1": out["fpn1"],
            "image_embed": out["image_embed"],
            "hr0": out["hr0"],
            "hr1": out["hr1"],
            "img_feats": fpn2.transpose(0, 2, 3, 1).reshape(SAM3_HW, 256)[:, None, :],
            "img_pos": pos2.transpose(0, 2, 3, 1).reshape(SAM3_HW, 256)[:, None, :],
        }

    def _ensure_pcs(self) -> dict:
        if self._pcs_feats is None:
            self._pcs_feats = self._run_vision(preprocess_sam3(self._img, pad=False))
        return self._pcs_feats

    def _ensure_pvs(self) -> dict:
        if self._pvs_feats is None:
            self._pvs_feats = self._run_vision(preprocess_sam3(self._img, pad=True))
        return self._pvs_feats

    # ------------------------------------------------------------------ interactive
    def segment_points(self, points, labels=None) -> PvsResult:
        points = [tuple(float(v) for v in p) for p in points]
        if labels is None:
            labels = [1] * len(points)
        return self._pvs(points, [int(v) for v in labels])

    def segment_box(self, box_xyxy_px) -> PvsResult:
        pts, lbs = _box_to_points(box_xyxy_px)
        return self._pvs(pts, [int(v) for v in lbs])

    def _pvs(self, points_px, labels) -> PvsResult:
        H, W = self._orig_shape
        f = self._ensure_pvs()
        r = min(SAM3_IMG / H, SAM3_IMG / W)
        pts = np.asarray(points_px, np.float32) * r
        labels = np.asarray(labels, np.int64)
        pad_coords = np.zeros((1, SAM3_PVS_MAX_POINTS, 2), np.float32)
        pad_labels = -np.ones((1, SAM3_PVS_MAX_POINTS), np.int64)
        n = min(len(pts), SAM3_PVS_MAX_POINTS)
        pad_coords[0, :n] = pts[:n]
        pad_labels[0, :n] = labels[:n]
        out = self._pvs_model.run(
            {
                "image_embed": f["image_embed"],
                "hr0": f["hr0"],
                "hr1": f["hr1"],
                "point_coords": pad_coords,
                "point_labels": pad_labels,
                "mask_input": np.zeros((1, 1, 288, 288), np.float32),
                "mask_present": np.zeros((1,), np.int64),
            }
        )
        mask = upscale_mask_pad(out["masks"][0, 0], (H, W))
        ys, xs = np.nonzero(mask)
        box = np.zeros((4,), np.float32)
        if len(xs) > 0:
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)
        return PvsResult(mask=mask, score=float(out["iou"][0]), box=box)

    # ------------------------------------------------------------------ PCS (text / all instances)
    def segment_text(self, texts, bboxes_xyxy_px=None) -> list[SamOnnxResult]:
        if isinstance(texts, str):
            texts = [texts]
        geo_feats, geo_masks = self._geometry(bboxes_xyxy_px)
        all_results: list[SamOnnxResult] = []
        for text in texts:
            tf, tm = self._encode_text(text)
            f = self._ensure_pcs()
            pred = self._detector_model.run(
                {
                    "fpn0": f["fpn0"],
                    "fpn1": f["fpn1"],
                    "img_feats": f["img_feats"],
                    "img_pos": f["img_pos"],
                    "text_features": tf,
                    "text_mask": tm,
                    "geo_feats": geo_feats,
                    "geo_masks": geo_masks,
                }
            )
            scores = pcs_scores(pred["pred_logits"], pred["presence"])
            masks, boxes_n, sc, _cls = pcs_filter_nms(pred["pred_boxes"], pred["pred_masks"], scores, self.conf, self.iou)
            H, W = self._orig_shape
            for m, b, s in zip(masks, boxes_n, sc):
                up = upscale_mask(m, (H, W))
                box_px = (b[0] * W, b[1] * H, b[2] * W, b[3] * H)
                all_results.append(SamOnnxResult(mask=up.astype(np.float32), score=float(s), box=tuple(box_px)))
        return all_results

    def _encode_text(self, text: str):
        if self._tokenizer is None:
            raise RuntimeError("SAM3 text segmentation requires vocab.json + merges.txt in the model folder")
        ids = self._tokenizer(text, context_length=SAM3_CTX)[0]  # (32,) int64, 0-padded
        out = self._text_enc_model.run({"token_ids": ids[None].astype(np.int64)})
        return out["text_features"], out["text_mask"]

    def _geometry(self, bboxes_xyxy_px):
        f = self._ensure_pcs()
        H, W = self._orig_shape
        if not bboxes_xyxy_px:
            geo_feats = np.zeros((SAM3_GEO_MAX_BOXES + 1, 1, 256), np.float32)
            geo_masks = np.ones((1, SAM3_GEO_MAX_BOXES + 1), np.int8)
            return geo_feats, geo_masks
        boxes = []
        for x1, y1, x2, y2 in bboxes_xyxy_px:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            boxes.append([cx / W, cy / H, (x2 - x1) / W, (y2 - y1) / H])
        n = len(boxes)
        pad_boxes = np.zeros((SAM3_GEO_MAX_BOXES, 1, 4), np.float32)
        pad_boxes[:n] = np.asarray(boxes, np.float32)[:, None, :]
        box_mask = np.zeros((1, SAM3_GEO_MAX_BOXES), np.int8)
        box_mask[0, n:] = 1  # 1 = padding
        _geo = self._geo_enc_model.run(
            {
                "img_feats": f["img_feats"],
                "img_pos": f["img_pos"],
                "boxes": pad_boxes,
                "labels": np.ones((SAM3_GEO_MAX_BOXES, 1), np.int64),
                "box_mask": box_mask,
            }
        )
        return _geo["geo_feats"], _geo["geo_masks"]
