"""Content-addressed store for images uploaded by clients.

Frames usually live in OpenList, but a client working on a *local* dataset has to
hand its frames to the server. Uploads land here keyed by sha256, which also gives
the inference worker a stable identity to cache embeddings by (and lets the API
reference an image the worker already has instead of re-sending it).
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from v2.core.config import Settings
from v2.core.errors import NotFound
from v2.core.logging import get_logger

logger = get_logger("zlabel.v2.images")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
