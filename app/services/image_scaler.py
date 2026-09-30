"""On-the-fly task-image variants (``GET .../images/{rel_path}?max_side=``).

The desktop draws a task at screen resolution while the storage tree holds camera
originals, so pulling a 20 MB photo to paint it once wastes bandwidth and time.
This service answers such a request with a downscaled copy of the stored file and
reports the **original** size in ``X-Image-Original-Size``: annotation documents
store full-resolution pixels, so the client needs that number to keep its
coordinates right (it stretches the small texture over the original rectangle).

Rules:

- never upscale - a file already within ``max_side`` is served byte-for-byte (the
  endpoint then also keeps the plain ETag),
- a broken/undecodable file falls back to the stored bytes: asking for a smaller
  copy must never turn a readable task into an error,
- the Exif block (orientation included!) and the ICC profile are carried over, so
  the variant shows exactly the pixels the original would - the server never
  rotates, because no client does today,
- a web format is kept (PNG/JPEG/WebP); anything else is re-encoded as PNG,
- variants are cached on disk keyed by ``"{stored etag}|m{max_side}"``: the first
  request pays for the decode/encode, every later one reads a file.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger("zlabel.app.image_scaler")

#: containers we can hand back as-is (everything else becomes PNG)
_FORMAT_MEDIA_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
_FALLBACK_FORMAT = "PNG"


@dataclass(frozen=True)
class ImageVariant:
    """A downscaled copy of a stored task image."""

    content: bytes
    media_type: str
    #: (width, height) of the **stored** file, not of ``content``
    original_size: tuple[int, int]


def _save_kwargs(out_format: str, *, exif: bytes, icc_profile: bytes) -> dict[str, Any]:
    """Metadata to carry into the re-encoded variant (Pillow rejects empty values).

    Exif only survives a JPEG re-encode here: Pillow's PNG writer takes text chunks
    from ``pnginfo`` and would not restore what we pass as ``exif``. Orientation is
    never applied - the stored pixels are what every client shows today.
    """
    kwargs: dict[str, Any] = {}
    if out_format == "JPEG" and exif:
        kwargs["exif"] = exif
    if icc_profile:
        kwargs["icc_profile"] = icc_profile
    return kwargs


def render_variant(
    content: bytes,
    max_side: int,
    *,
    quality: int = 85,
) -> ImageVariant | None:
    """Downscale ``content`` to ``max_side``; ``None`` when the original is fine.

    ``None`` means "serve the stored bytes": the image is already within the limit,
    it could not be decoded, or the result would not be smaller. Pure function - the
    cache and the HTTP details live in :class:`ImageScaler`.
    """
    try:
        with Image.open(io.BytesIO(content)) as image:
            original_size = (image.width, image.height)
            if max(original_size) <= max_side:
                return None
            source_format = (image.format or _FALLBACK_FORMAT).upper()
            exif = bytes(image.info.get("exif") or b"")
            icc_profile = bytes(image.info.get("icc_profile") or b"")
            scale = max_side / max(original_size)
            target = (max(1, round(original_size[0] * scale)), max(1, round(original_size[1] * scale)))
            if source_format == "JPEG":
                # JPEG can decode straight to ~1/2, 1/4, ... of the size: a photo
                # bound for 1280 px should not be decoded at 24 MP first.
                image.draft("RGB", target)
            resized = image.resize(target, Image.Resampling.LANCZOS)
            if source_format in _FORMAT_MEDIA_TYPES:
                out_format = source_format
            else:
                out_format = _FALLBACK_FORMAT
            if out_format == "JPEG" and resized.mode not in ("RGB", "L"):
                resized = resized.convert("RGB")
            buffer = io.BytesIO()
            resized.save(
                buffer,
                format=out_format,
                quality=quality,
                **_save_kwargs(out_format, exif=exif, icc_profile=icc_profile),
            )
            scaled = buffer.getvalue()
    except Exception as e:  # noqa: BLE001 - a broken image is not a server error
        logger.warning(f"cannot downscale a task image ({len(content)} bytes) to {max_side}px: {e}")
        return None
    if len(scaled) >= len(content):
        # a highly compressed original can grow when re-encoded: keep the original
        return None
    return ImageVariant(
        content=scaled, media_type=_FORMAT_MEDIA_TYPES[out_format], original_size=original_size
    )


class ImageScaler:
    """Downscales stored task images and caches the variants on disk."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = Path(settings.image_cache_dir)

    @property
    def limit(self) -> int:
        """Server-side ceiling for a requested size; ``0`` = no ceiling."""
        return max(0, int(self.settings.image_max_side_limit))

    @property
    def quality(self) -> int:
        return min(95, max(1, int(self.settings.image_encode_quality)))

    # region api
    def clamp(self, max_side: int) -> int:
        """The effective long-edge limit; ``0`` = "serve the stored file"."""
        if max_side <= 0:
            return 0
        if self.limit:
            return min(max_side, self.limit)
        return max_side

    def variant(self, content: bytes, *, max_side: int, etag: str) -> ImageVariant | None:
        """The cached/built variant for these exact bytes, or ``None`` to serve them as-is."""
        key = self._key(etag, max_side)
        cached = self._load(key)
        if cached is not None:
            return cached
        variant = render_variant(content, max_side, quality=self.quality)
        if variant is None:
            return None
        self._store(key, variant)
        return variant

    # endregion

    # region cache
    @staticmethod
    def _key(etag: str, max_side: int) -> str:
        return hashlib.sha256(f"{etag}|m{max_side}".encode()).hexdigest()

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.root / key[:2] / f"{key}.bin", self.root / key[:2] / f"{key}.json"

    def _load(self, key: str) -> ImageVariant | None:
        """The cached variant; a half-written or corrupt entry counts as a miss."""
        data_path, meta_path = self._paths(key)
        if not data_path.is_file() or not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return ImageVariant(
                content=data_path.read_bytes(),
                media_type=str(meta["media_type"]),
                original_size=(int(meta["width"]), int(meta["height"])),
            )
        except Exception as e:  # noqa: BLE001 - derived data, dropping it is always safe
            logger.warning(f"dropping unreadable image variant {key[:12]}: {e}")
            for path in (data_path, meta_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            return None

    def _store(self, key: str, variant: ImageVariant) -> None:
        """Write the bytes first, then the metadata: a reader never sees half an entry."""
        data_path, meta_path = self._paths(key)
        meta = {
            "media_type": variant.media_type,
            "width": variant.original_size[0],
            "height": variant.original_size[1],
        }
        try:
            data_path.parent.mkdir(parents=True, exist_ok=True)
            for path, payload in ((data_path, variant.content), (meta_path, json.dumps(meta).encode())):
                # the pid + thread keep two concurrent requests for the same variant
                # from writing (and therefore publishing) each other's temp file
                tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
                tmp.write_bytes(payload)
                os.replace(tmp, path)
        except OSError as e:  # a read-only/full disk must not break the request
            logger.warning(f"cannot cache image variant: {e}")

    # endregion
