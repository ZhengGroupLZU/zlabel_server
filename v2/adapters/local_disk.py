"""Filesystem storage backend: the server owns a directory tree.

Layout (deliberately identical to the desktop's local dataset mode, so the same
directory can be opened by the desktop as a dataset or served by the API):

    <ZLSERVER_STORAGE_ROOT>/<project>/                 # user's own structure
        images/...                                     # frames, any nesting
        .zlabel/
            project.json                               # optional metadata
            annos/<anno_id>.zlabel                     # annotations
            annos/_history/<anno_id>/v<n>.zlabel       # version history

No database server, no HTTP hop: listing and scanning are plain filesystem calls.
Every top-level directory is a project (the server owns the whole tree).
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

from v2.adapters.storage import FileInfo, is_image
from v2.core.config import Settings
from v2.core.errors import NotFound, UpstreamError, ValidationFailed
from v2.core.logging import get_logger

logger = get_logger("zlabel.v2.storage")


class LocalDiskBackend:
    """``StorageBackend`` over ``ZLSERVER_STORAGE_ROOT``."""

    kind = "local"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root_dir = Path(settings.storage_root).expanduser()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    # region paths (virtual, POSIX, rooted at "/")
    @property
    def root(self) -> str:
        return "/"

    def project_dir(self, project: str) -> str:
        return f"/{self._clean(project)}"

    def zlabel_dir(self, project: str) -> str:
        return f"{self.project_dir(project)}/{self.settings.anno_dir_clean}"

    def anno_path(self, project: str, anno_id: str) -> str:
        return f"{self.zlabel_dir(project)}/{self._clean(anno_id)}.zlabel"

    def history_path(self, project: str, anno_id: str, version: int) -> str:
        return f"{self.zlabel_dir(project)}/_history/{self._clean(anno_id)}/v{int(version)}.zlabel"

    def image_path(self, project: str, rel_path: str) -> str:
        return f"{self.project_dir(project)}/{rel_path.lstrip('/')}"

    # endregion

    # region io
    def list_dirs(self, path: str) -> list[str]:
        directory = self._resolve(path)
        if not directory.is_dir():
            raise NotFound(f"not found: {path}")
        return sorted(
            entry.name for entry in directory.iterdir() if entry.is_dir() and not entry.name.startswith(".")
        )

    def file_info(self, path: str) -> FileInfo:
        target = self._resolve(path)
        if not target.is_file():
            raise NotFound(f"not found: {path}")
        stat = target.stat()
        modified = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
        return FileInfo(path=path, size=stat.st_size, modified=modified)

    def exists(self, path: str) -> bool:
        try:
            return self._resolve(path).is_file()
        except ValidationFailed:
            return False

    def is_dir(self, path: str) -> bool:
        try:
            return self._resolve(path).is_dir()
        except ValidationFailed:
            return False

    def get_bytes(self, path: str) -> bytes:
        target = self._resolve(path)
        if not target.is_file():
            raise NotFound(f"not found: {path}")
        try:
            return target.read_bytes()
        except OSError as e:  # permissions, vanished mid-read, ...
            raise UpstreamError(f"cannot read {path}: {e}") from e

    def put_bytes(self, path: str, data: bytes) -> None:
        """Write atomically: a reader never sees a half-written annotation."""
        target = self._resolve(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
            tmp.write_bytes(data)
            os.replace(tmp, target)
        except OSError as e:
            raise UpstreamError(f"cannot write {path}: {e}") from e

    def glob_files(self, path: str) -> list[str]:
        directory = self._resolve(path)
        if not directory.is_dir():
            raise NotFound(f"not found: {path}")
        found: list[str] = []
        for entry in sorted(directory.rglob("*")):
            if entry.is_file() and not any(
                part.startswith(".") for part in entry.relative_to(directory).parts
            ):
                found.append(f"{path.rstrip('/')}/{entry.relative_to(directory).as_posix()}")
        return found

    def glob_images(self, path: str) -> list[str]:
        return [p for p in self.glob_files(path) if is_image(p)]

    def ensure_dir(self, path: str) -> None:
        try:
            self._resolve(path).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise UpstreamError(f"cannot create {path}: {e}") from e

    # endregion

    # region extras (housekeeping the API exposes)
    def delete(self, path: str) -> None:
        """Remove a file or an empty tree (used by the admin API)."""
        target = self._resolve(path)
        if not target.exists():
            raise NotFound(f"not found: {path}")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    def usage(self) -> dict[str, int]:
        """Total bytes/files under the root (admin dashboard)."""
        files = size = 0
        for entry in self.root_dir.rglob("*"):
            if entry.is_file():
                files += 1
                size += entry.stat().st_size
        return {"files": files, "bytes": size}

    def disk_path(self, path: str) -> Path:
        """The on-disk location of a virtual path (admin tooling/tests)."""
        return self._resolve(path)

    # endregion

    # region internals
    @staticmethod
    def _clean(name: str) -> str:
        clean = str(name).strip().strip("/")
        if not clean or clean in (".", "..") or "/" in clean or "\\" in clean:
            raise ValidationFailed(f"invalid path segment: {name!r}")
        return clean

    def _resolve(self, path: str) -> Path:
        """Map a virtual path onto the root, refusing anything that escapes it."""
        raw = str(path or "/").replace("\\", "/")
        if "\x00" in raw:
            raise ValidationFailed("invalid path")
        relative = raw.lstrip("/")
        parts = [part for part in relative.split("/") if part not in ("", ".")]
        if any(part == ".." for part in parts):
            raise ValidationFailed(f"path escapes the storage root: {path!r}")
        target = self.root_dir.joinpath(*parts)
        # a symlink pointing outside the root would sidestep the check above
        resolved_root = self.root_dir.resolve()
        resolved = target.resolve()
        if resolved != resolved_root and resolved_root not in resolved.parents:
            raise ValidationFailed(f"path escapes the storage root: {path!r}")
        return target

    # endregion
