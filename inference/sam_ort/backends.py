"""ONNXRuntime backend: session wrapper with provider (CPU/CUDA) selection.

Unlike the MNN backend there are no shape-based result caching pitfalls:
onnxruntime sessions are stateless and reusable across calls.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


def build_providers(backend: str = "CPU", threads: int = 4) -> tuple[list[str], ort.SessionOptions]:
    """Return (providers, session_options) for the requested backend.

    CUDA is used when available and requested; otherwise falls back to CPU.
    """
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = max(1, int(threads))
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if backend == "CUDA":
        providers = [
            ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]
    return providers, sess_options


class OrtSession:
    """A single ONNX model; run() returns outputs by name (same interface as the MNN backend)."""

    def __init__(self, model_path: str, backend: str = "CPU", threads: int = 4):
        self.path = str(Path(model_path))
        providers, sess_options = build_providers(backend, threads)
        self._session = self._create_session(providers, sess_options)
        self.input_names: list[str] = [i.name for i in self._session.get_inputs()]
        self.output_names: list[str] = [o.name for o in self._session.get_outputs()]

    def _create_session(self, providers: list[str], sess_options: ort.SessionOptions) -> ort.InferenceSession:
        """Create the session; on failure retry with progressively safer settings.

        ORT's ``SimplifiedLayerNormFusion`` optimizer is flaky on some graphs
        (intermittent "GetIndexFromName ... InsertedPrecisionFreeCast" failures),
        and the CUDA provider may be unavailable at runtime, so we degrade
        gracefully instead of crashing at startup.
        """
        try:
            return ort.InferenceSession(self.path, sess_options, providers=providers)
        except Exception:
            if "CUDAExecutionProvider" not in providers:
                raise
        # retry without the flaky LayerNorm fusion optimizer
        try:
            sess_options.disabled_optimizers = ["SimplifiedLayerNormFusion"]
            return ort.InferenceSession(self.path, sess_options, providers=providers)
        except Exception:
            pass
        # retry without graph optimizations, then fall back to CPU
        try:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            sess_options.disabled_optimizers = []
            return ort.InferenceSession(self.path, sess_options, providers=providers)
        except Exception:
            pass
        cpu_options = ort.SessionOptions()
        cpu_options.intra_op_num_threads = max(1, int(sess_options.intra_op_num_threads))
        return ort.InferenceSession(self.path, cpu_options, providers=["CPUExecutionProvider"])

    def run(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outs = self._session.run(self.output_names, feeds)
        return dict(zip(self.output_names, outs))
