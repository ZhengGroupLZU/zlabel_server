"""The only module that talks to OpenList.

Wraps the vendored SDK (``v2/vendor/openlist_api``) and translates its errors into
the v2 error model. Every method takes an explicit ``token``: the SDK client holds
a mutable token and is not thread-safe, so a client is created per call (cheap)
instead of sharing one across requests — the bug v1 had with its global client.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import requests

from v2.core.config import Settings
from v2.core.errors import ApiError, Forbidden, NotFound, SessionStale, Unauthorized, UpstreamError
from v2.core.logging import get_logger
from v2.vendor.openlist_api import OpenListAPIError, OpenListClient

logger = get_logger("zlabel.v2.openlist")

IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")

ClientFactory = Callable[[str], OpenListClient]


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


class OpenListAdapter:
    def __init__(self, settings: Settings, *, client_factory: ClientFactory | None = None) -> None:
        self.settings = settings
        self._client_factory: ClientFactory = client_factory or (lambda host: OpenListClient(host))
        # the scanner logs in once and reuses the token string (not a client)
        self._service_token: str = ""

    # region clients / auth
    def _client(self, token: str = "") -> OpenListClient:
        client = self._client_factory(self.settings.oplist_host)
        if token:
            client.set_token(token)
        return client

    def login(self, username: str, password: str) -> str:
        """Authenticate a user; returns their OpenList token."""
        try:
            resp = self._client().auth.login(username, password)
        except Exception as e:  # noqa: BLE001 - translated below
            if getattr(e, "status_code", None) in (401, 403):
                raise Unauthorized("OpenList rejected the credentials") from e
            raise self._translate(e, "login") from e
        token = str(getattr(getattr(resp, "data", None), "token", "") or "")
        if not token:
            raise Unauthorized("OpenList rejected the credentials")
        return token

    def current_user(self, token: str) -> dict[str, Any]:
        """``{"id","name","email"}`` of the token's owner."""
        data = self._data(self._client(token).auth.get_current_user(), "get_current_user")
        return {
            "id": str(getattr(data, "id", "") or ""),
            "name": str(getattr(data, "username", "") or getattr(data, "name", "") or ""),
            "email": str(getattr(data, "email", "") or ""),
        }

    def service_token(self) -> str:
        """Token for background work: static token first, then service credentials."""
        if self._service_token:
            return self._service_token
        if self.settings.oplist_token:
            self._service_token = self.settings.oplist_token
            return self._service_token
        if not self.settings.oplist_username or not self.settings.oplist_password:
            raise UpstreamError("no OpenList token and no service credentials configured")
        self._service_token = self.login(self.settings.oplist_username, self.settings.oplist_password)
        return self._service_token

    def reset_service_token(self) -> None:
        self._service_token = ""

    # endregion

    # region paths
    @property
    def root(self) -> str:
        return self.settings.oplist_proj_dir.rstrip("/") or "/"

    def project_dir(self, project: str) -> str:
        return f"{self.root}/{project.strip('/')}"

    def zlabel_dir(self, project: str) -> str:
        return f"{self.project_dir(project)}/zlabel"

    def anno_path(self, project: str, anno_id: str) -> str:
        """Where an annotation lives (shared with the desktop client)."""
        return f"{self.zlabel_dir(project)}/{anno_id}.zlabel"

    def history_path(self, project: str, anno_id: str, version: int) -> str:
        return f"{self.zlabel_dir(project)}/_history/{anno_id}/v{version}.zlabel"

    def image_path(self, project: str, rel_path: str) -> str:
        return f"{self.project_dir(project)}/{rel_path.lstrip('/')}"

    def marker_path(self, project: str) -> str:
        return f"{self.project_dir(project)}/{self.settings.project_marker}"

    # endregion

    # region filesystem
    def list_dirs(self, path: str, token: str) -> list[str]:
        try:
            resp = self._client(token).fs.dirs(path)
        except Exception as e:  # noqa: BLE001
            raise self._translate(e, f"dirs({path})") from e
        return [str(entry.name) for entry in (resp.data or [])]

    def file_info(self, path: str, token: str) -> RemoteFile:
        """Metadata of one object; a missing object raises ``NotFound``."""
        try:
            resp = self._client(token).fs.get(path)
        except Exception as e:  # noqa: BLE001
            raise self._translate(e, f"get({path})") from e
        data = getattr(resp, "data", None)
        if data is None:
            raise NotFound(f"not found: {path}")
        return RemoteFile(
            path=path,
            size=int(getattr(data, "size", 0) or 0),
            modified=str(getattr(data, "modified", "") or ""),
        )

    def exists(self, path: str, token: str) -> bool:
        try:
            self.file_info(path, token)
        except NotFound:
            return False
        return True

    def get_bytes(self, path: str, token: str) -> bytes:
        try:
            return self._client(token).fs.get_file_bytes(path)
        except Exception as e:  # noqa: BLE001
            raise self._translate(e, f"get_file_bytes({path})") from e

    def put_bytes(self, path: str, data: bytes, token: str) -> None:
        try:
            resp = self._client(token).fs.stream_upload(path, BytesIO(data), as_task=False)
        except Exception as e:  # noqa: BLE001
            raise self._translate(e, f"upload({path})") from e
        if int(getattr(resp, "code", 200) or 200) != 200:
            raise UpstreamError(f"OpenList refused the upload of {path}: {getattr(resp, 'message', '')}")

    def glob_files(self, path: str, token: str) -> list[str]:
        """Every file under ``path`` (recursive); the caller filters by extension."""
        try:
            return list(self._client(token).fs.glob(path, "*"))
        except Exception as e:  # noqa: BLE001
            raise self._translate(e, f"glob({path})") from e

    def glob_images(self, path: str, token: str) -> list[str]:
        return [p for p in self.glob_files(path, token) if is_image(p)]

    def ensure_dir(self, path: str, token: str) -> None:
        """Create ``path`` unless it already exists (OpenList mkdir is not recursive)."""
        if not path or path == "/":
            return
        marker = f"{path}/.keep"
        try:
            self._client(token).fs.mkdir(path)
        except Exception as e:  # noqa: BLE001 - "already exists" is fine
            translated = self._translate(e, f"mkdir({path})")
            if isinstance(translated, UpstreamError) and "exist" not in translated.message.lower():
                raise translated from e
        _ = marker

    # endregion

    # region internals
    @staticmethod
    def _data(resp: Any, what: str) -> Any:
        data = getattr(resp, "data", None)
        if data is None:
            raise NotFound(f"{what} returned no data")
        return data

    @staticmethod
    def _translate(exc: Exception, what: str) -> ApiError:
        """Map vendored-SDK / transport errors onto the v2 error model."""
        status = getattr(exc, "status_code", None)
        message = str(exc) or exc.__class__.__name__
        if status == 404:
            return NotFound(message)
        if status in (401, 403):
            return SessionStale(f"{what}: {message}") if status == 401 else Forbidden(f"{what}: {message}")
        if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            return UpstreamError(f"cannot reach OpenList: {message}")
        if isinstance(exc, OpenListAPIError):
            return UpstreamError(f"{what}: {message}")
        if isinstance(exc, requests.exceptions.HTTPError):
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code == 404:
                return NotFound(message)
            if code in (401, 403):
                return SessionStale(f"{what}: {message}")
            return UpstreamError(f"{what}: {message}")
        if isinstance(exc, ApiError):
            return exc
        logger.warning(f"unexpected OpenList error in {what}: {message!r}")
        return UpstreamError(f"{what}: {message}")

    # endregion
