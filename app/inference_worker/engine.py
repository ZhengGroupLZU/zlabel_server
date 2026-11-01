"""The serving engine: one model, one embedding snapshot per task image, a small queue.

Why snapshots: the runner keeps the encoded image in a handful of arrays, so the
worker can cache them per image (``EmbeddingCache``) and *restore* one instead of
re-running the encoder. That is what makes repeated clicks on the same task image cheap
and — more importantly — guarantees a job never runs on another task image's encoding
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

from app.core.errors import ApiError, InferenceUnavailable, ValidationFailed, WorkerBusy
from app.core.logging import get_logger
from app.inference_worker.schemas import InferJob
from inference.config import InferenceSettings
from inference.sam_ort import Predictor
from inference.worker import ZSamWorker
from inference.ztypes import AutoMode, Point, Polygon, Rect, ReturnType, SamReturn

logger = get_logger("zlabel.app.worker")

MODES = (0, 1, 2, 3)  # 0 = "SAM & CV" (the legacy both-toggles value), 1 SAM, 2 CV, 3 SAM|CV


@dataclass
class EngineMetrics:
    jobs: int = 0
    errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    #: decoded-image cache (skips re-transferring and re-decoding the stored file)
    image_cache_hits: int = 0
    image_cache_misses: int = 0
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
            "image_cache_hits": self.image_cache_hits,
            "image_cache_misses": self.image_cache_misses,
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


class DecodedImageCache:
    """LRU of decoded task images, keyed by content digest.

    Reading and decoding a 4000x6000 PNG costs ~0.5-1 s; a zoomed prompt crop makes
    that per click (ten seeds per plate), and the bytes never change for a digest. The
    cropped slice itself is cheap, so only the full decoded BGR array is cached.

    Memory: one 4000x6000 BGR array is ~72 MB, hence the small default size.
    """

    def __init__(self, size: int) -> None:
        self.size = max(0, int(size))
        self._items: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, key: str) -> np.ndarray | None:
        state = self._items.get(key)
        if state is not None:
            self._items.move_to_end(key)
        return state

    def put(self, key: str, image: np.ndarray) -> None:
        if self.size <= 0:
            return
        self._items[key] = image
        self._items.move_to_end(key)
        while len(self._items) > self.size:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


LOW_RES_SHAPE = (1, 1, 256, 256)


def decode_mask_input(encoded: str | None) -> np.ndarray | None:
    """base64 float16 (256, 256) logits -> float32 decoder input; None when absent."""
    if not encoded:
        return None
    raw = base64.b64decode(encoded)
    arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if arr.size != LOW_RES_SHAPE[2] * LOW_RES_SHAPE[3]:
        raise ValidationFailed(f"mask_input must hold {LOW_RES_SHAPE[2]}x{LOW_RES_SHAPE[3]} values")
    return arr.reshape(LOW_RES_SHAPE)


def encode_low_res(low_res) -> str | None:
    """float logits -> base64 float16 (half the bytes; logits do not need fp32)."""
    if low_res is None:
        return None
    arr = np.asarray(low_res, np.float32).reshape(LOW_RES_SHAPE[2], LOW_RES_SHAPE[3])
    return base64.b64encode(arr.astype(np.float16).tobytes()).decode("ascii")


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
        self.images = DecodedImageCache(getattr(settings, "decoded_image_cache_size", 2))
        #: per-request model selection: ``job.model`` picks one of these
        self._predictors: dict[str, Predictor] = {}
        self._fetch = fetch or requests.get
        self.metrics = EngineMetrics()
        self._current_key = ""
        self._current_model = ""
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
        predictor = self._predictor_for(job.model)
        # The API already tells us the content digest, so a plate we have decoded before
        # never gets pulled again: ``_load_image`` transfers the whole stored file (a
        # 4000x6000 PNG is ~30 MB) and used to run on *every* job - a zoomed prompt crop
        # makes that per click, and the transfer, not the decode, was the ~1 s.
        digest = (job.image_sha256 or "").strip()
        image = self.images.get(digest) if digest else None
        if image is not None:
            self.metrics.image_cache_hits += 1
        else:
            content = self._load_image(job)
            digest = digest or __import__("hashlib").sha256(content).hexdigest()
            image = self.images.get(digest)
            if image is None:
                image = to_bgr(Image.open(BytesIO(content)))
                self.images.put(digest, image)
                self.metrics.image_cache_misses += 1
        crop = self._crop_box(job, image.shape)
        offset = (0.0, 0.0)
        # a crop is a view, not a copy: the encoder copies what it needs
        region = image
        if crop is not None:
            left, top, right, bottom = crop
            region = image[top:bottom, left:right]
            offset = (left, top)

        key = self._cache_key(job, digest, crop)
        self._ensure_embedding(predictor, key, region)

        mode, return_type = self._modes(job)
        worker = ZSamWorker(
            model=predictor,
            anno_id=job.anno_id,
            img=region,
            auto_mode=mode,
            threshold=job.threshold,
            return_type=return_type,
            min_contour_area_ratio=self.settings.min_contour_area_ratio,
            contour_min_points=self.settings.contour_min_points,
            contour_max_points=self.settings.contour_max_points,
            contour_max_iterations=self.settings.contour_max_iterations,
            contour_smooth_sigma=self.settings.contour_smooth_sigma,
            contour_simplify_epsilon=self.settings.contour_simplify_epsilon,
            contour_offset=self.settings.contour_offset,
        )
        data = self._dispatch(worker, job, mode, offset)
        if offset != (0.0, 0.0):
            data = shift_results(data, *offset)
        payload = SamReturn(
            anno_id=job.anno_id,
            status=True,
            mode=mode.name,
            msg="success",
            data=list(data),
        ).model_dump()
        if job.want_low_res:
            # the client echoes this back as ``mask_input`` on the next click; it is
            # crop-space (the decoder sees the crop), which is fine as long as the
            # client only reuses it for the same crop_box
            payload["low_res"] = encode_low_res(getattr(worker, "last_low_res", None))
            payload["low_res_crop"] = list(crop) if crop is not None else None
        return payload

    def _dispatch(
        self, worker: ZSamWorker, job: InferJob, mode: AutoMode, offset: tuple[float, float] = (0.0, 0.0)
    ):
        prompts = job.prompts
        dx, dy = -offset[0], -offset[1]  # full-image prompts -> crop space
        if prompts.texts:
            if mode is not AutoMode.SAM:
                raise ValidationFailed("text prompts require mode=1 (SAM)")
            return worker.run_text(prompts.texts)
        if prompts.rects:
            return worker.run_rect(
                [Rect(x=r.x + dx, y=r.y + dy, w=r.w, h=r.h) for r in self._rects(prompts.rects)]
            )
        if prompts.points:
            points = self._points(prompts.points)
            if offset != (0.0, 0.0):
                points = [Point(x=p.x + dx, y=p.y + dy) for p in points]
            # A point outside the crop reaches the decoder at a coordinate the crop does
            # not contain, i.e. it is meaningless. The desktop widens its prompt window to
            # the union of all points, so this only fires for an out-of-date client (or a
            # hand-made request) - log it instead of guessing silently.
            height, width = worker.img.shape[:2]
            outside = [(round(p.x), round(p.y)) for p in points if not (0 <= p.x < width and 0 <= p.y < height)]
            if outside:
                logger.warning(
                    f"prompt points outside the crop window ({width}x{height}): {outside[:4]} - "
                    "the masks for these prompts are unreliable"
                )
            if mode not in (AutoMode.SAM, AutoMode.CV):
                raise ValidationFailed("point prompts support mode=1 (SAM) or mode=2 (CV)")
            return worker.run_point(
                points,
                list(prompts.labels or [1] * len(points)),
                mask_input=decode_mask_input(job.mask_input),
            )
        raise ValidationFailed("the job carries no prompts (points/rects/texts)")

    def _ensure_embedding(self, predictor: Predictor, key: str, image: np.ndarray) -> bool:
        """Restore the cached snapshot or encode once; returns True on a hit."""
        state = self.cache.get(key)
        if state is not None:
            if key != self._current_key:
                predictor.import_image_state(state)
                self._current_key = key
            self.metrics.cache_hits += 1
            return True
        predictor.set_image(image)
        self.cache.put(key, predictor.export_image_state())
        self._current_key = key
        self.metrics.cache_misses += 1
        return False

    def _predictor_for(self, name: str | None) -> Predictor:
        """The predictor for one request (``job.model``), built lazily and kept warm.

        The model used to be fixed at start-up, so the client's model choice only
        worked in local mode. A plate of seeds wants the 3x-downsampled EdgeSAM even
        less than a person does, and the machine has a GPU, so the choice has to travel
        with the request.
        """
        wanted = (name or self.settings.model_name or "").strip()
        if not wanted or wanted == self.settings.model_name:
            return self.predictor
        cached = self._predictors.get(wanted)
        if cached is not None:
            return cached
        logger.info(f"loading inference model {wanted} on demand")
        built = Predictor(
            model_dir=self.settings.model_dir,
            model_name=wanted,
            backend=self.settings.ort_backend,
            threads=self.settings.ort_threads,
            conf=self.settings.sam3_conf,
            iou=self.settings.sam3_iou,
        )
        self._predictors[wanted] = built
        return built

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
            headers = (
                {"Authorization": f"Bearer {self.settings.inference_token}"}
                if self.settings.inference_token
                else {}
            )
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

    def _cache_key(self, job: InferJob, digest: str, crop: tuple[int, int, int, int] | None) -> str:
        """Embeddings belong to one (model, image bytes, crop) triple.

        The model name is part of the key: two models produce different features for the
        same pixels, and a shared snapshot would silently feed one to the other.
        """
        model = (job.model or self.settings.model_name or "").strip() or "default"
        crop_part = "" if crop is None else "#" + ",".join(str(int(v)) for v in crop)
        return f"{model}|{digest}{crop_part}"

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
            "models": sorted({self.settings.model_name, *self._predictors}),
            "backend": self.settings.ort_backend,
            "loaded": self.predictor.loaded,
            "cache": {
                "size": self.cache.size,
                "entries": len(self.cache),
                "hit_rate": self.metrics.hit_rate,
                "decoded_images": len(self.images),
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
