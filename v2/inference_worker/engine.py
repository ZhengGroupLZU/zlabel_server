"""The serving engine: one model, one embedding snapshot per frame, a small queue.

Why snapshots: the runner keeps the encoded image in a handful of arrays, so the
worker can cache them per image (``EmbeddingCache``) and *restore* one instead of
re-running the encoder. That is what makes repeated clicks on the same frame cheap
and — more importantly — guarantees a job never runs on another frame's encoding
(the v1 bug: a global "current image" shared by every client).
"""

from __future__ import annotations

import base64
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import numpy as np
import requests
from PIL import Image

from inference.config import InferenceSettings
from inference.sam_ort import Predictor
from inference.worker import ZSamWorker
from inference.ztypes import AutoMode, Point, Polygon, Rect, ReturnType, SamReturn
from v2.core.errors import ApiError, InferenceUnavailable, ValidationFailed, WorkerBusy
from v2.core.logging import get_logger
from v2.inference_worker.schemas import InferJob

logger = get_logger("zlabel.v2.worker")

MODES = (0, 1, 2, 3)  # 0 = "SAM & CV" (the legacy both-toggles value), 1 SAM, 2 CV, 3 SAM|CV


@dataclass
class EngineMetrics:
    jobs: int = 0
    errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    latencies: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    started_at: float = field(default_factory=time.time)

    @property
    def hit_rate(self) -> float | None:
        total = self.cache_hits + self.cache_misses
        return round(self.cache_hits / total, 4) if total else None

    def snapshot(self) -> dict[str, Any]:
        latencies = sorted(self.latencies)
        return {
            "jobs": self.jobs,
            "errors": self.errors,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.hit_rate,
            "latency_ms": {
                "p50": self._pct(latencies, 0.5),
                "p95": self._pct(latencies, 0.95),
                "max": round(latencies[-1] * 1000, 2) if latencies else 0.0,
            },
            "uptime_s": round(time.time() - self.started_at, 1),
        }

    @staticmethod
    def _pct(sorted_values: list[float], q: float) -> float:
        if not sorted_values:
            return 0.0
        index = min(len(sorted_values) - 1, int(len(sorted_values) * q))
        return round(sorted_values[index] * 1000, 2)


class EmbeddingCache:
    """LRU of encoded-image snapshots keyed by ``sha256`` (+ crop)."""

    def __init__(self, size: int) -> None:
        self.size = max(1, size)
        self._items: OrderedDict[str, dict] = OrderedDict()

    def get(self, key: str) -> dict | None:
        state = self._items.get(key)
        if state is not None:
            self._items.move_to_end(key)
        return state

    def put(self, key: str, state: dict) -> None:
        self._items[key] = state
        self._items.move_to_end(key)
        while len(self._items) > self.size:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items


def to_bgr(image: Image.Image) -> np.ndarray:
    """PIL (RGB) → OpenCV-style BGR, the layout the runners expect."""
    return np.asarray(image.convert("RGB"), dtype=np.uint8)[..., ::-1].copy()


def shift_results(results, dx: float, dy: float):
    """Move crop-space results back into full-image coordinates."""
    if dx == 0 and dy == 0:
        return results
    shifted = []
    for item in results:
        if isinstance(item, Rect):
            shifted.append(Rect(x=item.x + dx, y=item.y + dy, w=item.w, h=item.h))
        elif isinstance(item, Polygon):
            shifted.append(Polygon(points=[Point(x=p.x + dx, y=p.y + dy) for p in item.points]))
        else:  # RLE strings stay as they are
            shifted.append(item)
    return shifted


class InferenceEngine:
    def __init__(
        self,
        settings: InferenceSettings,
        *,
        predictor: Predictor | None = None,
        cache: EmbeddingCache | None = None,
        fetch: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self.predictor = predictor or Predictor(
            model_dir=settings.model_dir,
            model_name=settings.model_name,
            backend=settings.ort_backend,
            threads=settings.ort_threads,
            conf=settings.sam3_conf,
            iou=settings.sam3_iou,
        )
        self.cache = cache or EmbeddingCache(settings.embedding_cache_size)
        self._fetch = fetch or requests.get
        self.metrics = EngineMetrics()
        self._current_key = ""
        self._admit_lock = threading.Lock()
        self._waiting = 0
        self._slot = threading.Semaphore(max(1, settings.max_concurrency))

    # region serving
    def run(self, job: InferJob) -> dict[str, Any]:
        self._admit()
        started = time.perf_counter()
        try:
            with self._slot:
                payload = self._run_locked(job)
        except ApiError:
            self.metrics.errors += 1
            raise
        except Exception as e:  # noqa: BLE001 - surfaced as a 503 to the API
            self.metrics.errors += 1
            logger.exception(f"inference job failed ({job.job_id}): {e}")
            raise InferenceUnavailable(f"inference failed: {e}") from e
        finally:
            self.metrics.latencies.append(time.perf_counter() - started)
            with self._admit_lock:
                self._waiting -= 1
        self.metrics.jobs += 1
        return payload

    def _run_locked(self, job: InferJob) -> dict[str, Any]:
        content = self._load_image(job)
        image = to_bgr(Image.open(BytesIO(content)))
        crop = self._crop_box(job, image.shape)
        offset = (0.0, 0.0)
        if crop is not None:
            left, top, right, bottom = crop
            image = image[top:bottom, left:right]
            offset = (left, top)

        key = self._cache_key(job, content, crop)
        self._ensure_embedding(key, image)

        mode, return_type = self._modes(job)
        worker = ZSamWorker(
            model=self.predictor,
            anno_id=job.anno_id,
            img=image,
            auto_mode=mode,
            threshold=job.threshold,
            return_type=return_type,
            min_contour_area_ratio=self.settings.min_contour_area_ratio,
            contour_min_points=self.settings.contour_min_points,
            contour_max_points=self.settings.contour_max_points,
            contour_max_iterations=self.settings.contour_max_iterations,
        )
        data = self._dispatch(worker, job, mode)
        if offset != (0.0, 0.0):
            data = shift_results(data, *offset)
        return SamReturn(
            anno_id=job.anno_id,
            status=True,
            mode=mode.name,
            msg="success",
            data=list(data),
        ).model_dump()

    def _dispatch(self, worker: ZSamWorker, job: InferJob, mode: AutoMode):
        prompts = job.prompts
        if prompts.texts:
            if mode is not AutoMode.SAM:
                raise ValidationFailed("text prompts require mode=1 (SAM)")
            return worker.run_text(prompts.texts)
        if prompts.rects:
            return worker.run_rect(self._rects(prompts.rects))
        if prompts.points:
            if mode not in (AutoMode.SAM, AutoMode.CV):
                raise ValidationFailed("point prompts support mode=1 (SAM) or mode=2 (CV)")
            return worker.run_point(
                self._points(prompts.points), list(prompts.labels or [1] * len(prompts.points))
            )
        raise ValidationFailed("the job carries no prompts (points/rects/texts)")

    def _ensure_embedding(self, key: str, image: np.ndarray) -> bool:
        """Restore the cached snapshot or encode once; returns True on a hit."""
        state = self.cache.get(key)
        if state is not None:
            if key != self._current_key:
                self.predictor.import_image_state(state)
                self._current_key = key
            self.metrics.cache_hits += 1
            return True
        self.predictor.set_image(image)
        self.cache.put(key, self.predictor.export_image_state())
        self._current_key = key
        self.metrics.cache_misses += 1
        return False

    # endregion

    # region inputs
    def _load_image(self, job: InferJob) -> bytes:
        if job.image_b64:
            try:
                return base64.b64decode(job.image_b64)
            except Exception as e:  # noqa: BLE001
                raise ValidationFailed(f"image_b64 is not valid base64: {e}") from e
        if job.image_url:
            url = job.image_url
            if not url.startswith("http"):
                url = f"{self.settings.api_base_url.rstrip('/')}/{url.lstrip('/')}"
            headers = {"Authorization": f"Bearer {self.settings.token}"} if self.settings.token else {}
            try:
                resp = self._fetch(url, headers=headers, timeout=60)
            except requests.RequestException as e:
                raise InferenceUnavailable(f"cannot pull the image from the API: {e}") from e
            if int(getattr(resp, "status_code", 0)) != 200:
                raise InferenceUnavailable(f"the API refused the image pull ({resp.status_code})")
            return resp.content
        raise ValidationFailed("the job carries no image (image_b64 or image_url)")

    def _crop_box(self, job: InferJob, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
        if not job.crop_box or len(job.crop_box) != 4:
            return None
        height, width = shape[:2]
        left, top, right, bottom = (int(round(v)) for v in job.crop_box)
        left, top = max(0, left), max(0, top)
        right, bottom = min(width, right), min(height, bottom)
        if right - left < 2 or bottom - top < 2:
            return None
        return left, top, right, bottom

    def _cache_key(self, job: InferJob, content: bytes, crop: tuple[int, int, int, int] | None) -> str:
        digest = job.image_sha256 or __import__("hashlib").sha256(content).hexdigest()
        return digest if crop is None else f"{digest}#{crop[0]},{crop[1]},{crop[2]},{crop[3]}"

    @staticmethod
    def _points(raw: list[Any]) -> list[Point]:
        points = []
        for item in raw:
            if isinstance(item, dict):
                points.append(Point(x=float(item["x"]), y=float(item["y"])))
            else:
                points.append(Point(x=float(item[0]), y=float(item[1])))
        return points

    @staticmethod
    def _rects(raw: list[Any]) -> list[Rect]:
        rects = []
        for item in raw:
            if isinstance(item, dict):
                rects.append(
                    Rect(x=float(item["x"]), y=float(item["y"]), w=float(item["w"]), h=float(item["h"]))
                )
            else:
                rects.append(Rect(x=float(item[0]), y=float(item[1]), w=float(item[2]), h=float(item[3])))
        return rects

    def _modes(self, job: InferJob) -> tuple[AutoMode, ReturnType]:
        mode = job.mode
        if mode not in MODES:
            raise ValidationFailed(f"unknown mode: {job.mode}")
        if job.return_type not in (1, 2, 3):
            raise ValidationFailed(f"unknown return_type: {job.return_type}")
        return AutoMode(mode), ReturnType(job.return_type)

    def _admit(self) -> None:
        with self._admit_lock:
            if self._waiting >= self.settings.queue_size:
                raise WorkerBusy(f"the inference queue is full ({self.settings.queue_size} waiting)")
            self._waiting += 1

    # endregion

    # region observability
    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.settings.model_name,
            "backend": self.settings.ort_backend,
            "loaded": self.predictor.loaded,
            "cache": {
                "size": self.cache.size,
                "entries": len(self.cache),
                "hit_rate": self.metrics.hit_rate,
            },
            "queue": {
                "waiting": self._waiting,
                "size": self.settings.queue_size,
                "concurrency": self.settings.max_concurrency,
            },
            "jobs": self.metrics.jobs,
        }

    def metrics_snapshot(self) -> dict[str, Any]:
        return self.metrics.snapshot()

    # endregion
