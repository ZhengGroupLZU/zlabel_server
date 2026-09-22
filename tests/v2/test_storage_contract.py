"""Every storage backend must behave the same.

The same contract runs against the OpenList backend (over the in-memory fake) and
the local disk backend, so switching backends cannot silently change what the
services see. Add a case here whenever the storage surface grows.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.v2.fakes import FakeOpenList
from v2.adapters.local_disk import LocalDiskBackend
from v2.adapters.openlist import OpenListAdapter
from v2.adapters.storage import StorageBackend
from v2.core.config import Settings
from v2.core.errors import NotFound, ValidationFailed

PROJ = "projA"
IMAGE = "images/dish01/D1.png"


@pytest.fixture(params=["local", "openlist"])
def backend(request, tmp_path) -> StorageBackend:
    """Both implementations, side by side."""
    if request.param == "local":
        settings = Settings(
            storage_backend="local", storage_root=str(tmp_path / "storage"), anno_dir=".zlabel/annos"
        )
        settings.ensure_dirs()
        return LocalDiskBackend(settings)
    ol = FakeOpenList(service_token="service-token")
    ol.add_file(f"/zlabel_server/projects/{PROJ}/{IMAGE}", b"png")
    ol.add_file(f"/zlabel_server/projects/{PROJ}/notes.txt", b"text")
    request.node.fake_openlist = ol
    return OpenListAdapter(
        Settings(
            oplist_proj_dir="/zlabel_server/projects",
            oplist_token="service-token",
            anno_dir=".zlabel/annos",  # pinned: both backends on the same layout
        ),
        client_factory=ol.client,
    )


def put(backend: StorageBackend, path: str, data: bytes = b"data") -> None:
    backend.put_bytes(path, data, backend.service_token())


def test_anno_dir_defaults_per_backend():
    """An existing OpenList deployment keeps its historical directory.

    Switching it to the shared ``.zlabel/annos`` layout is an explicit choice (set
    ZLSERVER_ANNO_DIR) made after migrating the files, so a restart can never make
    saved annotations invisible.
    """
    from v2.core.config import Settings as _Settings

    assert _Settings(storage_backend="openlist").anno_dir_clean == "zlabel"
    assert _Settings(storage_backend="local").anno_dir_clean == ".zlabel/annos"
    assert _Settings(storage_backend="local", anno_dir="custom/annos").anno_dir_clean == "custom/annos"
    unsafe = _Settings(anno_dir="../escape")
    with pytest.raises(ValueError):
        _ = unsafe.anno_dir_clean


def test_backend_declares_its_discovery_mode(backend: StorageBackend):
    assert backend.kind in ("local", "openlist")
    assert isinstance(backend.uses_marker_discovery, bool)
    assert backend.service_token() == backend.service_token()  # cached/stable
    assert backend.uses_marker_discovery is (backend.kind == "openlist")


def test_path_helpers_agree(backend: StorageBackend):
    project_dir = backend.project_dir(PROJ)
    assert project_dir.endswith(PROJ)
    assert backend.anno_path(PROJ, "a1").endswith("/.zlabel/annos/a1.zlabel")
    assert backend.history_path(PROJ, "a1", 3).endswith("/.zlabel/annos/_history/a1/v3.zlabel")
    assert backend.image_path(PROJ, IMAGE).endswith(f"/{IMAGE}")
    assert backend.marker_path(PROJ).endswith(".zlabel-server-project-root")


def test_put_get_roundtrip_and_etag(backend: StorageBackend):
    path = backend.anno_path(PROJ, "a1")
    put(backend, path, b'{"v": 1}')
    assert backend.get_bytes(path, backend.service_token()) == b'{"v": 1}'

    info = backend.file_info(path, backend.service_token())
    assert info.size == len(b'{"v": 1}') and info.etag.startswith('"8-')
    assert backend.exists(path, backend.service_token()) is True


def test_missing_objects_are_not_found(backend: StorageBackend):
    missing = backend.anno_path(PROJ, "nope")
    with pytest.raises(NotFound):
        backend.get_bytes(missing, backend.service_token())
    with pytest.raises(NotFound):
        backend.file_info(missing, backend.service_token())
    assert backend.exists(missing, backend.service_token()) is False
    # annotating a frame of a project that does not exist yet is fine (mkdir -p)
    put(backend, backend.anno_path("brand_new", "a1"))
    assert backend.exists(backend.anno_path("brand_new", "a1"), backend.service_token())


def test_ensure_dir_is_idempotent_and_listed(backend: StorageBackend):
    put(backend, backend.image_path(PROJ, IMAGE))
    backend.ensure_dir(backend.zlabel_dir(PROJ), backend.service_token())
    backend.ensure_dir(backend.zlabel_dir(PROJ), backend.service_token())  # must not raise
    assert PROJ in backend.list_dirs(backend.root, backend.service_token())


def test_glob_images_only_returns_images(backend: StorageBackend):
    put(backend, backend.image_path(PROJ, IMAGE))
    put(backend, backend.image_path(PROJ, "notes.txt"))
    put(backend, backend.image_path(PROJ, "zlabel_should_be_skipped.junk"))
    found = backend.glob_images(backend.project_dir(PROJ), backend.service_token())
    assert [p.rsplit("/", 1)[-1] for p in found] == ["D1.png"]
    # frames and annotations may sit next to each other without interfering
    put(backend, backend.anno_path(PROJ, "a1"), b"{}")
    assert backend.glob_images(backend.project_dir(PROJ), backend.service_token()) == found


def test_paths_cannot_escape_the_root(backend: StorageBackend):
    """A traversal must never read outside the root.

    The local backend refuses the path up front (ValidationFailed); the OpenList
    backend cannot resolve it either, so it answers NotFound - either way nothing
    outside the storage is reachable.
    """
    for bad in ("/../outside.png", f"/{PROJ}/../../outside.png", "/a/../../../etc/passwd"):
        with pytest.raises((ValidationFailed, NotFound)):
            backend.get_bytes(bad, backend.service_token())


def test_writes_are_atomic_and_replaced(backend: StorageBackend):
    path = backend.anno_path(PROJ, "a1")
    put(backend, path, b"first")
    put(backend, path, b"second")
    assert backend.get_bytes(path, backend.service_token()) == b"second"
    # a local backend must not leave temp files behind
    if backend.kind == "local":
        left = [p.name for p in Path(backend.root_dir).rglob("*") if ".tmp-" in p.name]
        assert left == []


def test_delete_is_local_only(backend: StorageBackend):
    put(backend, backend.image_path(PROJ, IMAGE))
    if backend.kind != "local":
        pytest.skip("deleting is an admin feature of the local backend")
    backend.delete(backend.image_path(PROJ, IMAGE))
    assert backend.exists(backend.image_path(PROJ, IMAGE), backend.service_token()) is False
    assert backend.usage()["bytes"] == 0
