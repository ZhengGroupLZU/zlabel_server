"""Real-model smoke tests (CPU). Marked ``slow``: runs full-size models.

Scores are asserted with generous ranges so model re-exports do not break the
suite; the goal is regression detection (e.g. wrong preprocessing, broken
backends), not exact value matching.
"""

from __future__ import annotations

import shutil

import pytest

from inference.sam_ort import Predictor
from inference.sam_ort.runner import Sam3Runner

pytestmark = pytest.mark.slow

# point/box locations on the test image (720x1280)
PERSON_POINT = [(400, 300)]
PERSON_BOX = [(400, 200, 800, 700)]
SAM3_POINT = [(635, 271)]  # PVS uses letterbox scaling (min-ratio 1008/1280=0.7875)
# original stretch mapping was (500, 380) in 1008-space; letterbox maps to (500, 213)


def _predict(img_bgr, name, points=None, bboxes=None, text=None, conf=None, iou=None):
    kwargs = {}
    if conf is not None:
        kwargs["conf"] = conf
    if iou is not None:
        kwargs["iou"] = iou
    p = Predictor(model_dir="assets/onnx", model_name=name, backend="CPU", threads=8, **kwargs)
    p.set_image(img_bgr)
    return p.predict(points=points, labels=[1] * len(points) if points else None, bboxes=bboxes, text=text)


class TestSamFamily:
    @pytest.mark.parametrize(
        ("name", "score_range"),
        [
            ("SAM", (0.8, 1.0)),
            ("EdgeSAM", (0.5, 1.0)),
            ("SlimSAM", (0.8, 1.0)),
            ("SAM2", (0.1, 0.5)),
        ],
    )
    def test_point_prompt(self, img_bgr, name, score_range):
        results = _predict(img_bgr, name, points=PERSON_POINT)
        assert results, "expected at least one mask"
        best = results[0]
        assert score_range[0] <= best.score <= score_range[1], f"{name} score {best.score}"
        assert best.mask.shape == img_bgr.shape[:2]

    @pytest.mark.parametrize(
        ("name", "score_range"),
        [
            ("SAM", (0.7, 1.0)),
            ("EdgeSAM", (0.7, 1.0)),
            ("SlimSAM", (0.7, 1.0)),
            ("SAM2", (0.7, 1.0)),
        ],
    )
    def test_box_prompt(self, img_bgr, name, score_range):
        results = _predict(img_bgr, name, bboxes=PERSON_BOX)
        assert results
        assert score_range[0] <= results[0].score <= score_range[1]
        assert results[0].mask.shape == img_bgr.shape[:2]


class TestSam3:
    def test_point_prompt(self, img_bgr):
        results = _predict(img_bgr, "SAM3", points=SAM3_POINT)
        assert results
        assert 0.3 <= results[0].score <= 0.7
        assert results[0].mask.shape == img_bgr.shape[:2]

    def test_text_prompt(self, img_bgr):
        results = _predict(img_bgr, "SAM3", text=["person"])
        assert results, "text prompt should produce detections"
        assert 0.9 <= results[0].score <= 1.0
        assert results[0].mask.shape == img_bgr.shape[:2]
        assert results[0].box is not None

    def test_box_prompt(self, img_bgr):
        results = _predict(img_bgr, "SAM3", bboxes=PERSON_BOX)
        assert results
        assert 0.5 <= results[0].score <= 1.0
        assert results[0].mask.shape == img_bgr.shape[:2]

    def test_text_and_box_prompt(self, img_bgr):
        results = _predict(img_bgr, "SAM3", bboxes=PERSON_BOX, text=["person"])
        assert results
        assert results[0].mask.shape == img_bgr.shape[:2]

    def test_text_without_tokenizer_files(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        for f in [
            "sam3_vision_encoder.onnx",
            "sam3_pvs.onnx",
            "sam3_text_encoder.onnx",
            "sam3_geometry_encoder.onnx",
            "sam3_detector.onnx",
        ]:
            shutil.copy(f"assets/onnx/{f}", d / f)
        runner = Sam3Runner(d, backend="CPU", threads=4)
        with pytest.raises(RuntimeError, match="vocab.json"):
            runner._encode_text("person")
