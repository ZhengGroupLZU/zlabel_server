"""Backend tests: provider selection and session I/O discovery (EdgeSAM, small model)."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest

from inference.sam_ort.backends import OrtSession, build_providers

EDGE_ENCODER = "assets/onnx/edge_sam_3x_encoder.onnx"


class TestBuildProviders:
    def test_cpu(self):
        providers, so = build_providers("CPU", 4)
        assert providers == ["CPUExecutionProvider"]
        assert so.intra_op_num_threads == 4

    def test_cuda_includes_cpu_fallback(self):
        providers, _ = build_providers("CUDA", 4)
        names = [p if isinstance(p, str) else p[0] for p in providers]
        assert "CPUExecutionProvider" in names
        if "CUDAExecutionProvider" in ort.get_available_providers():
            assert "CUDAExecutionProvider" in names


@pytest.fixture(scope="module")
def edge_session() -> OrtSession:
    return OrtSession(EDGE_ENCODER, backend="CPU", threads=4)


class TestOrtSession:
    def test_io_names(self, edge_session):
        assert edge_session.input_names == ["image"]
        assert edge_session.output_names == ["image_embeddings"]

    def test_run(self, edge_session):
        x = np.zeros((1, 3, 1024, 1024), np.float32)
        out = edge_session.run({"image": x})
        assert set(out.keys()) == {"image_embeddings"}
        assert out["image_embeddings"].shape == (1, 256, 64, 64)

    def test_input_ordering_insensitive(self, edge_session):
        x = np.zeros((1, 3, 1024, 1024), np.float32)
        # feeds dict is looked up by name, so any key order works
        out1 = edge_session.run({"image": x})
        out2 = edge_session.run({"image": x})
        np.testing.assert_array_equal(out1["image_embeddings"], out2["image_embeddings"])
