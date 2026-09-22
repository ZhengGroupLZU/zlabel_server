"""The storage contract every backend must satisfy.

v2 already funnelled all file IO through one adapter, so swapping OpenList for
something else is a matter of implementing this Protocol. ``token`` stays in the
signatures because the OpenList backend needs the caller's credential; the local
backend ignores it (the server owns the disk).

Paths are *virtual* POSIX paths: ``/<project>/<relative path>``. The backend maps
them onto whatever it stores things in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from v2.core.config import Settings

IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")

#: where annotations live inside a project directory (same layout as the desktop's
#: local datasets, so one dataset directory can be used by both sides)
ZL_LABEL_DIR = ".zlabel/annos"


@dataclass(frozen=True)
class RemoteFile:
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
    #: True when a project is discovered by a marker file (OpenList); False when
    #: the server owns the tree and any image-bearing directory is a project
    uses_marker_discovery: bool

    # region paths
    @property
    def root(self) -> str: ...

    def project_dir(self, project: str) -> str: ...

    def zlabel_dir(self, project: str) -> str: ...

    def anno_path(self, project: str, anno_id: str) -> str: ...

    def history_path(self, project: str, anno_id: str, version: int) -> str: ...

    def image_path(self, project: str, rel_path: str) -> str: ...

    def marker_path(self, project: str) -> str: ...

    # endregion

    # region io
    def list_dirs(self, path: str, token: str = "") -> list[str]: ...

    def file_info(self, path: str, token: str = "") -> RemoteFile: ...

    def exists(self, path: str, token: str = "") -> bool: ...

    def get_bytes(self, path: str, token: str = "") -> bytes: ...

    def put_bytes(self, path: str, data: bytes, token: str = "") -> None: ...

    def glob_files(self, path: str, token: str = "") -> list[str]: ...

    def glob_images(self, path: str, token: str = "") -> list[str]: ...

    def ensure_dir(self, path: str, token: str = "") -> None: ...

    # endregion


def build_storage(settings: Settings):
    """Pick the configured backend (kept as a function so tests can inject one)."""
    backend: Literal["openlist", "local"] = settings.storage_backend
    if backend == "local":
        from v2.adapters.local_disk import LocalDiskBackend

        return LocalDiskBackend(settings)
    from v2.adapters.openlist import OpenListAdapter

    return OpenListAdapter(settings)
