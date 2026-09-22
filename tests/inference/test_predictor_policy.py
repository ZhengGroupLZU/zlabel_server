"""Predictor policy and routing tests (no real inference)."""

from __future__ import annotations

import pytest

from inference.sam_ort.backends import build_providers
from inference.sam_ort.predictor import Predictor


def _provider_names(backend: str) -> list[str]:
    providers, _ = build_providers(backend, 4)
    return [p if isinstance(p, str) else p[0] for p in providers]


class TestBackendPolicy:
    """The per-model CUDA policy was removed; build_providers uses CUDA for any
    model when requested and falls back to CPU when CUDA is unavailable."""

    def test_cpu_requested(self):
        assert _provider_names("CPU") == ["CPUExecutionProvider"]

    def test_cuda_includes_cpu_fallback(self):
        names = _provider_names("CUDA")
        assert "CPUExecutionProvider" in names

    @pytest.mark.parametrize("name", ["SAM", "SlimSAM", "EdgeSAM", "SAM2", "SAM3"])
    def test_backend_independent_of_model(self, name):
        # requested backend drives provider selection, not the model name
        p = Predictor(model_dir="assets/onnx", model_name=name, backend="CPU", threads=2)
        assert p.backend == "CPU"
        assert p.model_name == name


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

    def test_sam3_bbox_routes_to_segment_box(self):
        # bbox-only prompts go to the interactive PVS path (segment_box), not text
        p = self.make("SAM3")
        p.predict(bboxes=[(1, 2, 3, 4)])
        assert p._runner.calls[0][0] == "segment_box"
        assert p._runner.calls[0][1] == [1.0, 2.0, 3.0, 4.0]

    def test_sam3_text_passthrough(self):
        p = self.make("SAM3")
        p.predict(text=["person"])
        assert p._runner.calls[0][0] == "segment_text"
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
