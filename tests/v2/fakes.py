"""An in-memory OpenList: the v2 tests never touch the network.

Mirrors the parts of the vendored SDK the adapter uses (``auth.login``,
``auth.get_current_user``, ``fs.dirs/get/get_file_bytes/glob/stream_upload/mkdir``)
and raises the same exception types, so the adapter's error translation is
exercised for real.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from types import SimpleNamespace

from v2.vendor.openlist_api import NotFoundError, OpenListAPIError


@dataclass(frozen=True)
class Entry:
    name: str
    is_dir: bool


class FakeOpenList:
    def __init__(
        self,
        users: dict[str, str] | None = None,
        root: str = "/zlabel_server/projects",
        service_token: str = "",
    ) -> None:
        self.root = root
        self.service_token = service_token
        self.files: dict[str, bytes] = {}
        self.dirs: set[str] = {root}
        self.users: dict[str, str] = dict(users or {"rainy": "secret"})
        self.tokens: dict[str, str] = {}
        self.calls: list[tuple[str, str]] = []
        self.failures: dict[str, Exception] = {}
        self._seq = 0

    # region test-side helpers
    def add_file(self, path: str, data: bytes = b"data") -> str:
        self._ensure_parents(path)
        self.files[path] = data
        return path

    def add_dir(self, path: str) -> str:
        self._ensure_parents(path)
        self.dirs.add(path.rstrip("/"))
        return path

    def fail_on(self, path: str, exc: Exception) -> None:
        self.failures[path] = exc

    def images(self) -> list[str]:
        return sorted(self.files)

    def client(self, _host: str = ""):
        return FakeClient(self)

    # endregion

    def _ensure_parents(self, path: str) -> None:
        parts = path.strip("/").split("/")
        for i in range(1, len(parts)):
            self.dirs.add("/" + "/".join(parts[:i]))

    def _check(self, path: str) -> None:
        if path in self.failures:
            raise self.failures[path]

    def issue_token(self, username: str) -> str:
        self._seq += 1
        token = f"ol-token-{self._seq}-{username}"
        self.tokens[token] = username
        return token

    def user_of(self, token: str) -> str:
        if token and self.service_token and token == self.service_token:
            return "service"
        if not token or token not in self.tokens:
            raise OpenListAPIError("invalid token", status_code=401)
        return self.tokens[token]


class FakeAuth:
    def __init__(self, ol: FakeOpenList, client: FakeClient) -> None:
        self.ol = ol
        self.client = client

    def login(self, username: str, password: str, otp_code: str | None = None):
        self.ol.calls.append(("login", username))
        if self.ol.users.get(username) != password:
            raise OpenListAPIError("failed find user: record not found", status_code=401)
        token = self.ol.issue_token(username)
        self.client.set_token(token)
        return SimpleNamespace(data=SimpleNamespace(token=token))

    def get_current_user(self):
        user = self.ol.user_of(self.client.token)  # raises 401 when invalid
        self.ol.calls.append(("me", user))
        return SimpleNamespace(
            data=SimpleNamespace(id=f"id-{user}", username=user, email=f"{user}@example.com")
        )


class FakeFs:
    def __init__(self, ol: FakeOpenList, client: FakeClient) -> None:
        self.ol = ol
        self.client = client

    def _auth(self) -> None:
        self.ol.user_of(self.client.token)

    def dirs(self, path: str, password: str = "", force_root: bool = False):
        self._auth()
        self.ol.calls.append(("dirs", path))
        self.ol._check(path)
        prefix = path.rstrip("/") + "/"
        names = sorted(
            {p[len(prefix) :].split("/")[0] for p in self.ol.dirs if p.startswith(prefix) and p != path}
        )
        return SimpleNamespace(data=[SimpleNamespace(name=n) for n in names])

    def ls(self, path: str, password: str = "", refresh: bool = False):
        self._auth()
        self.ol.calls.append(("ls", path))
        self.ol._check(path)
        if path.rstrip("/") not in self.ol.dirs:
            raise NotFoundError(f"object not found: {path}", status_code=404)
        prefix = path.rstrip("/") + "/"
        entries: list[Entry] = []
        for p in sorted(self.ol.dirs):
            rest = p[len(prefix) :]
            if p.startswith(prefix) and "/" not in rest:
                entries.append(Entry(rest, True))
        for p in sorted(self.ol.files):
            rest = p[len(prefix) :]
            if p.startswith(prefix) and "/" not in rest:
                entries.append(Entry(rest, False))
        return SimpleNamespace(data=SimpleNamespace(content=entries))

    def glob(self, path: str, pattern: str, password: str = "", refresh: bool = False) -> list[str]:
        self._auth()
        self.ol.calls.append(("glob", path))
        self.ol._check(path)
        prefix = path.rstrip("/") + "/"
        return [
            p
            for p in sorted(self.ol.files)
            if p.startswith(prefix) and fnmatch.fnmatch(p.rsplit("/", 1)[-1], pattern)
        ]

    def get(self, path: str, password: str = "", page: int = 1, per_page: int = 0, refresh: bool = False):
        self._auth()
        self.ol.calls.append(("get", path))
        self.ol._check(path)
        if path in self.ol.files:
            content = self.ol.files[path]
            return SimpleNamespace(
                data=SimpleNamespace(
                    size=len(content), modified="2026-01-01T00:00:00Z", raw_url=f"raw://{path}"
                )
            )
        if path in self.ol.dirs:
            return SimpleNamespace(
                data=SimpleNamespace(size=0, modified="2026-01-01T00:00:00Z", raw_url=None)
            )
        # OpenList reports a missing object as 200 + empty data; the SDK maps it to 404
        raise NotFoundError(f"failed to getobj: object not found: {path}", status_code=404)

    def get_file_bytes(
        self, path: str, password: str = "", page: int = 1, per_page: int = 0, refresh: bool = False
    ) -> bytes:
        self._auth()
        self.ol.calls.append(("read", path))
        self.ol._check(path)
        if path not in self.ol.files:
            raise NotFoundError(f"failed to getobj: object not found: {path}", status_code=404)
        return self.ol.files[path]

    def stream_upload(self, file_path: str, file, as_task: bool = True):
        self._auth()
        self.ol.calls.append(("upload", file_path))
        self.ol._check(file_path)
        data = file.read() if hasattr(file, "read") else bytes(file)
        self.ol.add_file(file_path, data)
        return SimpleNamespace(code=200, message="success")

    def mkdir(self, path: str):
        self._auth()
        self.ol.calls.append(("mkdir", path))
        self.ol._check(path)
        if path.rstrip("/") in self.ol.dirs:
            raise OpenListAPIError("directory already exists", status_code=500)
        self.ol.add_dir(path)
        return SimpleNamespace(code=200, message="success")


class FakeClient:
    def __init__(self, ol: FakeOpenList) -> None:
        self.ol = ol
        self.token = ""
        self.fs = FakeFs(ol, self)
        self.auth = FakeAuth(ol, self)

    def set_token(self, token: str) -> None:
        self.token = token
