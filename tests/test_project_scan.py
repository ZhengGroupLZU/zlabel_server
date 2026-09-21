"""Hermetic tests for the OpenList project auto-discovery logic.

No network or real OpenList calls; ``discover_projects`` is driven by a fake
``fs`` that mirrors the tiny subset of the API it touches (``dirs``, ``get``,
``glob``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests

from app.openlist_api import OpenListAPIError
from app.project_scan import ALLOWED_IMAGE_EXT, discover_projects

MARKER = ".zlabel-server-project-root"


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _DirItem:
    def __init__(self, name: str):
        self.name = name


class _DirsResponse:
    def __init__(self, dirs: list[str]):
        self.data = [_DirItem(d) for d in dirs]


class _Resp:
    def __init__(self, status: int):
        self.status_code = status


class FakeFS:
    """Minimal mimic of ``FileSystemAPI`` used by discovery."""

    def __init__(self, dirs, marker_present, glob_files, explode_dirs=False):
        self._dirs = dirs  # top-level project names under /projects
        self._marker_present = set(marker_present)  # project dirs that have the marker
        self._glob_files = glob_files  # project_dir -> list of full file paths
        self._explode_dirs = explode_dirs

    def dirs(self, path):
        if self._explode_dirs:
            raise RuntimeError("oplist unavailable")
        return _DirsResponse(self._dirs)

    def get(self, path):
        # expected: /projects/<name>/.zlabel-server-project-root
        parts = path.rstrip("/").split("/")
        if parts[-1] != MARKER:
            raise AssertionError(f"unexpected get path: {path}")
        project_name = parts[-2]  # the dir segment before the marker
        if project_name in self._marker_present:
            return object()  # exists; value is ignored
        raise requests.HTTPError("not found", response=_Resp(404))

    def glob(self, path, pattern):
        return self._glob_files.get(path, [])


class _ErrFS(FakeFS):
    """A ``get`` that fails with a machine error rather than a 404."""

    def __init__(self, dirs, glob_files=None):
        super().__init__(dirs, marker_present=[], glob_files=glob_files or {})
        self._explode_dirs = False

    def get(self, path):
        raise requests.ConnectionError("oplist unreachable")


class FakeClient:
    def __init__(self, fs: FakeFS):
        self.fs = fs


def make_settings(marker=MARKER, proj_dir="/projects"):
    return SimpleNamespace(oplist_proj_dir=proj_dir, project_marker=marker)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_project_requires_marker():
    """Only dirs whose marker exists are projects; others are missing-marker."""
    fs = FakeFS(
        dirs=["projA", "projB"],
        marker_present=["projA"],
        glob_files={"/projects/projA": ["/projects/projA/a.png", "/projects/projA/b.jpg"]},
    )
    result = discover_projects(FakeClient(fs), make_settings())

    assert result.present_dirs == {"projA", "projB"}
    assert result.confirmed_missing == {"projB"}
    assert result.projects == [
        {"name": "projA", "files": ["/projects/projA/a.png", "/projects/projA/b.jpg"]}
    ]


def test_image_ext_filtering_recursive():
    """Only png/jpg/jpeg survive; non-image files are dropped."""
    fs = FakeFS(
        dirs=["projX"],
        marker_present=["projX"],
        glob_files={
            "/projects/projX": [
                "/projects/projX/.zlabel-server-project-root",
                "/projects/projX/sub/a.png",
                "/projects/projX/note.txt",
                "/projects/projX/b.JPEG",
                "/projects/projX/c.gif",
            ]
        },
    )
    result = discover_projects(FakeClient(fs), make_settings())
    assert result.projects[0]["files"] == [
        "/projects/projX/sub/a.png",
        "/projects/projX/b.JPEG",
    ]


def test_marker_probe_404_confirmed_missing():
    """A 404 on the marker probe confirms the dir is not a project."""
    fs = FakeFS(
        dirs=["projY"],
        marker_present=[],
        glob_files={},
    )
    result = discover_projects(FakeClient(fs), make_settings())
    assert result.projects == []
    assert result.confirmed_missing == {"projY"}
    assert result.present_dirs == {"projY"}


def test_uninspectable_dir_is_not_confirmed_missing():
    """A dir we cannot inspect (network error) must be left alone, not deactivated."""
    fs = _ErrFS(dirs=["projZ"])
    result = discover_projects(FakeClient(fs), make_settings())
    assert result.projects == []
    assert result.confirmed_missing == set()
    assert result.present_dirs == {"projZ"}


def test_openlist_api_error_404_is_confirmed_missing():
    """Some OpenList responses carry the status in the JSON body (OpenListAPIError)."""
    class _APIErrFS(FakeFS):
        def get(self, path):
            raise OpenListAPIError("not found", status_code=404)

    fs = _APIErrFS(dirs=["projW"], marker_present=[], glob_files={})
    result = discover_projects(FakeClient(fs), make_settings())
    assert result.confirmed_missing == {"projW"}


def test_root_listing_failure_propagates():
    """If the root listing fails the caller must skip any DB sync entirely."""
    fs = FakeFS(dirs=[], marker_present=[], glob_files={}, explode_dirs=True)
    with pytest.raises(RuntimeError, match="oplist unavailable"):
        discover_projects(FakeClient(fs), make_settings())


def test_allowed_extensions_constant():
    assert all(e in ALLOWED_IMAGE_EXT for e in (".png", ".jpg", ".jpeg"))
