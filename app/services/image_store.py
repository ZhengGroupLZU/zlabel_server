"""Content-addressed store for images uploaded by clients.

Task images usually live in the storage tree, but a client working on a *local* dataset
has to hand its task images to the server. Uploads land here keyed by sha256, which also gives
the inference worker a stable identity to cache embeddings by (and lets the API
reference an image the worker already has instead of re-sending it).
"""

from __future__ import annotations

import hashlib
import shutil
from collections import OrderedDict
from pathlib import Path

from app.core.config import Settings
from app.core.errors import NotFound
from app.core.logging import get_logger

logger = get_logger("zlabel.app.images")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class PathDigestCache:
    """``(path, stat etag) -> sha256``, so a repeat predict does not re-read the file.

    Stateless prediction is keyed by the task image's digest, and deriving it used to
    mean reading (over OpenList, fetching) the whole stored file on *every* request - a
    4000x6000 PNG is ~30 MB, and a zoomed prompt crop makes that per click. The etag is
    cheap enough to re-check on each request, so a replaced file still gets a new digest
    while an unchanged one costs nothing.
    """

    def __init__(self, size: int = 128) -> None:
        self.size = max(1, int(size))
        self._items: OrderedDict[tuple[str, str], str] = OrderedDict()

    def get(self, path: str, etag: str) -> str | None:
        key = (path, etag)
        digest = self._items.get(key)
        if digest is not None:
            self._items.move_to_end(key)
        return digest

    def put(self, path: str, etag: str, digest: str) -> None:
        key = (path, etag)
        self._items[key] = digest
        self._items.move_to_end(key)
        while len(self._items) > self.size:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


class ImageStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = Path(settings.upload_dir)

    def path_for(self, digest: str) -> Path:
        if len(digest) < 4 or not all(c in "0123456789abcdef" for c in digest.lower()):
            raise NotFound(f"bad image digest: {digest!r}")
        return self.root / digest[:2] / f"{digest}.bin"

    def put(self, data: bytes) -> tuple[str, int]:
        """Store bytes and return ``(sha256, size)``; identical bytes hit the same file."""
        digest = sha256_bytes(data)
        target = self.path_for(digest)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            logger.debug(f"stored upload {digest[:12]} ({len(data)} bytes)")
        return digest, len(data)

    def get(self, digest: str) -> bytes:
        path = self.path_for(digest)
        if not path.exists():
            raise NotFound(f"unknown image: {digest}")
        return path.read_bytes()

    def has(self, digest: str) -> bool:
        try:
            return self.path_for(digest).exists()
        except NotFound:
            return False

    def size(self) -> int:
        if not self.root.exists():
            return 0
        return sum(p.stat().st_size for p in self.root.rglob("*.bin"))

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
