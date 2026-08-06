"""CPU vs CUDA consistency tests. Marked ``gpu`` (skipped without CUDA).

Guards against silent backend drift: the ONNXRuntime CUDA EP must produce
masks equivalent to the CPU path (IoU >= 0.99) and be measurably faster for
the models that enable it.
"""

from __future__ import annotations

import gc
import time

import numpy as np
import onnxruntime as ort
import pytest

from app.sam_ort import Predictor

pytestmark = pytest.mark.gpu

CUDA_AVAILABLE = "CUDAExecutionProvider" in ort.get_available_providers()

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDAExecutionProvider not available"),
]

SAM3_POINT = [(635, 271)]
PERSON_POINT = [(400, 300)]


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    union = (a | b).sum()
    return inter / union if union else 0.0


def _run(name: str, backend: str, img_bgr: np.ndarray):
    p = Predictor(model_dir="assets/onnx", model_name=name, backend=backend, threads=8)
    t0 = time.time()
    p.set_image(img_bgr)
    points = SAM3_POINT if name == "SAM3" else PERSON_POINT
    results = p.predict(points=points, labels=[1])
    elapsed = time.time() - t0
    return results[0], elapsed


@pytest.mark.parametrize("name", ["SAM", "SAM2", "SAM3"])
def test_cuda_parity_and_speedup(name, img_bgr):
    cpu_result, cpu_t = _run(name, "CPU", img_bgr)
    gc.collect()  # free CPU-side sessions before the GPU run
    gpu_result, gpu_t = _run(name, "CUDA", img_bgr)
    try:
        iou = _mask_iou(cpu_result.mask > 0, gpu_result.mask > 0)
        assert iou >= 0.99, f"{name}: CPU/CUDA mask IoU {iou:.3f}"
        assert gpu_t < cpu_t, f"{name}: CUDA not faster ({cpu_t:.1f}s vs {gpu_t:.1f}s)"
        speedup = cpu_t / max(gpu_t, 1e-6)
        assert speedup >= 2.0, f"{name}: speedup only {speedup:.1f}x"
    finally:
        gc.collect()


@pytest.mark.parametrize("name", ["EdgeSAM", "SlimSAM"])
def test_cuda_forced_cpu_still_correct(name, img_bgr):
    """Models without CUDA benefit must run on CPU and stay correct."""
    cpu_result, _ = _run(name, "CPU", img_bgr)
    gc.collect()
    gpu_result, _ = _run(name, "CUDA", img_bgr)  # internally forced to CPU
    iou = _mask_iou(cpu_result.mask > 0, gpu_result.mask > 0)
    assert iou >= 0.99, f"{name}: CPU vs forced-CPU IoU {iou:.3f}"
    gc.collect()
