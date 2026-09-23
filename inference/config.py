"""Inference settings (model + worker).

Read by the inference worker process and by the API when it needs model metadata
(e.g. which models exist). Same ``ZLSERVER_`` env prefix as the API, so one env
file (``.env.v2``) configures both processes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceSettings(BaseSettings):
    # --- model --------------------------------------------------------------
    model_name: Literal["SAM", "SlimSAM", "EdgeSAM", "SAM2", "SAM3"] = "EdgeSAM"
    model_dir: str = "assets/onnx"
    # onnxruntime execution provider: CPU / CUDA (falls back to CPU when unavailable)
    ort_backend: Literal["CPU", "CUDA"] = "CPU"
    ort_threads: int = 8
    # SAM3 PCS detection thresholds
    sam3_conf: float = 0.25
    sam3_iou: float = 0.7

    # contour post-processing (kept identical to the desktop client's values)
    contour_min_points: int = 10
    contour_max_points: int = 100
    contour_max_iterations: int = 10
    min_contour_area_ratio: float = 3.0e-5

    # --- worker -------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8001
    # Shared secret with the API, both directions (the API's ``inference_token``):
    # the worker requires it on /infer and presents it when pulling an image.
    # Env var: ZLSERVER_INFERENCE_TOKEN (empty = refuse to serve, dev must set it).
    inference_token: str = ""
    # where the pull path fetches task images from (the API's /api/v2/internal/images)
    api_base_url: str = "http://127.0.0.1:8000"
    # embeddings are cached by image sha256; this bounds that cache
    embedding_cache_size: int = 32
    # how many requests the worker runs concurrently (1 = serialise, safest for
    # the SAM3 CUDA arena; raise only with enough VRAM)
    max_concurrency: int = 1
    # queue capacity (requests waiting for a slot)
    queue_size: int = 64
    request_timeout: float = 300.0

    model_config = SettingsConfigDict(
        env_prefix="ZLSERVER_", env_file=".env.v2", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache(maxsize=1)
def get_inference_settings() -> InferenceSettings:
    return InferenceSettings()
