"""Predictor policy and routing tests (no real inference)."""

from __future__ import annotations

import pytest

from app.sam_ort.predictor import CUDA_FAST_MODELS, Predictor, _effective_backend


class TestEffectiveBackend:
    @pytest.mark.parametrize("name", ["SlimSAM", "EdgeSAM"])
    def test_cuda_requested_but_forced_cpu(self, name):
        assert _effective_backend(name, "CUDA") == "CPU"

    @pytest.mark.parametrize("name", sorted(CUDA_FAST_MODELS))
    def test_cuda_fast_models(self, name):
        assert _effective_backend(name, "CUDA") == "CUDA"

    @pytest.mark.parametrize("name", ["SAM", "SlimSAM", "EdgeSAM", "SAM2", "SAM3"])
    def test_cpu_always_cpu(self, name):
        assert _effective_backend(name, "CPU") == "CPU"


class TestPredictorRouting:
    def make(self, model_name: str = "SAM") -> Predictor:
        p = Predictor(model_dir="assets/onnx", model_name=model_name, backend="CPU", threads=2)
        p._runner = FakeRunner(has_segment_text=(model_name == "SAM3"))
        return p

    def test_points_route_to_segment_points(self):
        p = self.make("SAM")
        p.predict(points=[(1, 2)], labels=[1])
        assert p._runner.calls == [("segment_points", [(1.0, 2.0)], [1])]

    def test_bboxes_route_to_segment_box(self):
        p = self.make("SAM")
        p.predict(bboxes=[(1, 2, 3, 4)])
        assert p._runner.calls == [("segment_box", [1.0, 2.0, 3.0, 4.0])]

    def test_text_requires_sam3(self):
        p = self.make("SAM")
        with pytest.raises(ValueError, match="text prompts require the SAM3 model"):
            p.predict(text=["person"])

    def test_sam3_text_defaults_to_visual(self):
        p = self.make("SAM3")
        p.predict(bboxes=[(1, 2, 3, 4)])
        assert p._runner.calls[0][0] == "segment_text"
        assert p._runner.calls[0][1] == ["visual"]

    def test_sam3_text_passthrough(self):
        p = self.make("SAM3")
        p.predict(text=["person"])
        assert p._runner.calls[0][1] == ["person"]

    def test_sam3_text_str_single(self):
        p = self.make("SAM3")
        p.predict(text="person")
        assert p._runner.calls[0][1] == ["person"]

    def test_bboxes_and_text_combined(self):
        p = self.make("SAM3")
        p.predict(bboxes=[(1, 2, 3, 4)], text=["person"])
        assert p._runner.calls[0][0] == "segment_text"
        assert p._runner.calls[0][1] == ["person"]

    def test_no_prompts_raises(self):
        p = self.make("SAM")
        with pytest.raises(ValueError, match="points"):
            p.predict()

    def test_points_and_bboxes(self):
        p = self.make("SAM")
        p.predict(points=[(1, 2)], labels=[1], bboxes=[(1, 2, 3, 4)])
        assert p._runner.calls[0][0] == "segment_box"
        assert p._runner.calls[1][0] == "segment_points"


class FakeRunner:
    def __init__(self, has_segment_text: bool):
        self._has_segment_text = has_segment_text
        self.calls: list = []
        self.conf = 0.25
        self.iou = 0.7

    def __getattr__(self, name):
        # emulate SAM3-only segment_text capability (hasattr must be False otherwise)
        if name == "segment_text" and self._has_segment_text:
            def segment_text(texts, bboxes):
                self.calls.append(("segment_text", texts, bboxes))
                return []

            return segment_text
        raise AttributeError(name)

    def segment_points(self, points, labels):
        self.calls.append(("segment_points", points, labels))
        return []

    def segment_box(self, box):
        self.calls.append(("segment_box", box))
        return []
