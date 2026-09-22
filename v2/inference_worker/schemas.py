"""Wire models of the inference worker (the API is the only client)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Prompts(BaseModel):
    points: list[Any] | None = None  # [{x,y}] or [[x,y]]
    labels: list[float] | None = None
    rects: list[Any] | None = None  # [{x,y,w,h}] or [[x,y,w,h]]
    texts: list[str] | None = None


class InferJob(BaseModel):
    """One stateless request: the frame (inline or pullable) + the prompt."""

    job_id: str = ""
    anno_id: str = ""
    image_sha256: str = ""
    image_b64: str | None = None
    image_url: str | None = None
    model: str | None = None
    prompts: Prompts = Field(default_factory=Prompts)
    threshold: int = 100
    mode: int = 1  # AutoMode: 1 SAM, 2 CV, 3 SAM+CV
    return_type: int = 1  # 1 RECT, 2 POLYGON, 3 RLE
    crop_box: list[float] | None = None  # (left, top, right, bottom) in image pixels


class CacheOut(BaseModel):
    size: int
    entries: int
    hit_rate: float | None = None


class QueueOut(BaseModel):
    waiting: int
    size: int
    concurrency: int


class HealthOut(BaseModel):
    status: str = "ok"
    model: str
    backend: str
    loaded: bool
    cache: CacheOut
    queue: QueueOut
    jobs: int


class LatencyOut(BaseModel):
    p50: float = 0.0
    p95: float = 0.0
    max: float = 0.0


class MetricsOut(BaseModel):
    jobs: int
    errors: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float | None = None
    latency_ms: LatencyOut
    uptime_s: float
