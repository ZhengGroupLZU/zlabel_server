"""The storage contract the server relies on.

All file IO goes through one adapter, so the Protocol below is the seam that keeps
the services independent of *where* the bytes live (today: ``LocalDiskBackend``
over ``ZLSERVER_STORAGE_ROOT``; tests may inject their own implementation).

Paths are *virtual* POSIX paths: ``/<project>/<relative path>``. The backend maps
them onto whatever it stores things in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.core.config import Settings

IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")

#: where annotations live inside a project directory (same layout as the desktop's
#: local datasets, so one dataset directory can be used by both sides)
ZL_LABEL_DIR = ".zlabel/annos"


@dataclass(frozen=True)
class FileInfo:
    """Just enough file metadata for HTTP caching / upload checks."""

    path: str
    size: int
    modified: str

    @property
    def etag(self) -> str:
        return f'"{self.size}-{self.modified}"'


def is_image(path: str) -> bool:
    return path.lower().endswith(IMAGE_EXTENSIONS)


@runtime_checkable
class StorageBackend(Protocol):
    """File storage + the path conventions the services rely on."""

    kind: str

    # region paths
    @property
    def root(self) -> str: ...

    def project_dir(self, project: str) -> str: ...

    def zlabel_dir(self, project: str) -> str: ...

    def anno_path(self, project: str, anno_id: str) -> str: ...

    def history_path(self, project: str, anno_id: str, version: int) -> str: ...

    def image_path(self, project: str, rel_path: str) -> str: ...

    # endregion

    # region io
    def list_dirs(self, path: str) -> list[str]: ...

    def file_info(self, path: str) -> FileInfo: ...

    def exists(self, path: str) -> bool: ...

    def is_dir(self, path: str) -> bool: ...

    def get_bytes(self, path: str) -> bytes: ...

    def put_bytes(self, path: str, data: bytes) -> None: ...

    def glob_files(self, path: str) -> list[str]: ...

    def glob_images(self, path: str) -> list[str]: ...

    def ensure_dir(self, path: str) -> None: ...

    # endregion


def build_storage(settings: Settings):
    """Build the storage backend (kept as a function so tests can inject one)."""
    from app.adapters.local_disk import LocalDiskBackend

    return LocalDiskBackend(settings)
