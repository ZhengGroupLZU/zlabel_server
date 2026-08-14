"""Unified predictor facade over the ONNXRuntime runners, mirroring the Ultralytics-style API.

predictor = build_runner(model_dir="assets/onnx", model_name="SAM3", backend="CPU")
predictor.set_image(image_bgr)
results = predictor.predict(points=[[x, y]], labels=[1])
results = predictor.predict(bboxes=[[x1, y1, x2, y2]])
results = predictor.predict(text=["person"])        # SAM3 PCS
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from app.sam_ort.runner import Sam2Runner, Sam3Runner, SamRunner
from app.ztypes import PvsResult, SamOnnxResult

MODEL_FILES: dict[str, tuple[str, str]] = {
    "SAM": ("sam_vit_b_encoder.onnx", "sam_vit_b_decoder.onnx"),
    "EdgeSAM": ("edge_sam_3x_encoder.onnx", "edge_sam_3x_decoder.onnx"),
    # fp32-converted encoder: the original fp16 graph triggers an uncatchable
    # onnxruntime optimizer crash (see scripts/fp16_to_fp32.py)
    "SlimSAM": ("slimsam_encoder.onnx", "slimsam_decoder.onnx"),
    "SAM2": ("sam2_hiera_large.encoder.onnx", "sam2_hiera_large.decoder.onnx"),
}


def build_runner(
    model_dir: str | Path,
    model_name: str,
    backend: str = "CPU",
    threads: int = 4,
    conf: float = 0.25,
    iou: float = 0.7,
):
    d = Path(model_dir)
    if model_name == "SAM3":
        return Sam3Runner(d, conf=conf, iou=iou, backend=backend, threads=threads)
    enc, dec = MODEL_FILES.get(model_name, MODEL_FILES["EdgeSAM"])
    if model_name == "SAM2":
        return Sam2Runner(d / enc, d / dec, backend=backend, threads=threads)
    return SamRunner(d / enc, d / dec, backend=backend, threads=threads)


def _to_results(r) -> list[SamOnnxResult]:
    if isinstance(r, SamOnnxResult):
        return [r]
    if isinstance(r, PvsResult):
        return [SamOnnxResult(mask=r.mask.astype(np.float32), score=r.score, box=tuple(r.box))]
    return list(r)


class Predictor:
    """Local ONNXRuntime predictor with an Ultralytics-style API."""

    def __init__(
        self,
        model_dir: str | Path,
        model_name: str = "EdgeSAM",
        backend: str = "CPU",
        threads: int = 4,
        conf: float = 0.25,
        iou: float = 0.7,
    ):
        self.model_dir = str(Path(model_dir))
        self.model_name = model_name
        self.backend = backend
        self.threads = threads
        self.conf = conf
        self.iou = iou
        self._runner = None
        # onnxruntime sessions are not thread-safe across concurrent runs
        self._lock = threading.Lock()

    def setup_model(self):
        if self._runner is None:
            self._runner = build_runner(
                self.model_dir, self.model_name, self.backend, self.threads, self.conf, self.iou
            )
        return self

    def set_image(self, image: np.ndarray) -> Predictor:
        with self._lock:
            self.setup_model()
            self._runner.set_image(image)
        return self

    def reset_image(self):
        with self._lock:
            self._runner = None

    def predict(
        self,
        points: list | None = None,
        labels: list | None = None,
        bboxes: list | None = None,
        text: list | str | None = None,
        conf: float | None = None,
        iou: float | None = None,
    ) -> list[SamOnnxResult]:
        """Run inference and return a flat list of SamOnnxResult (mask/score/box)."""
        with self._lock:
            return self._predict_unsafe(points, labels, bboxes, text, conf, iou)

    def _predict_unsafe(
        self,
        points: list | None = None,
        labels: list | None = None,
        bboxes: list | None = None,
        text: list | str | None = None,
        conf: float | None = None,
        iou: float | None = None,
    ) -> list[SamOnnxResult]:
        self.setup_model()
        if conf is not None:
            self._runner.conf = conf
        if iou is not None:
            self._runner.iou = iou

        if text is not None or bboxes:
            if not hasattr(self._runner, "segment_text"):
                if text is not None:
                    raise ValueError("text prompts require the SAM3 model")
                return self._run_points_boxes(points, labels, bboxes)
            texts = [text] if isinstance(text, str) else text
            texts = texts or ["visual"]
            return list(self._runner.segment_text(texts, bboxes))

        if points is None:
            raise ValueError("provide at least one of points=, bboxes= or text=")
        return self._run_points_boxes(points, labels, bboxes)

    def _run_points_boxes(self, points, labels, bboxes) -> list[SamOnnxResult]:
        results: list[SamOnnxResult] = []
        if bboxes:
            for b in bboxes:
                results.extend(_to_results(self._runner.segment_box([float(v) for v in b])))
        if points:
            pts = [tuple(float(v) for v in p) for p in points]
            lbs = list(labels) if labels is not None else [1] * len(pts)
            results.extend(_to_results(self._runner.segment_points(pts, lbs)))
        return results

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
