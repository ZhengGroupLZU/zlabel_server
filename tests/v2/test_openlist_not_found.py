"""A missing object is a 404, never an opaque 500 (the desktop relies on it)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from v2.vendor.openlist_api.exceptions import NotFoundError
from v2.vendor.openlist_api.fs import FileSystemAPI


def test_get_file_bytes_missing_object_raises_not_found(monkeypatch):
    api = FileSystemAPI.__new__(FileSystemAPI)  # skip __init__ (no client needed)
    monkeypatch.setattr(api, "get", lambda *a, **k: SimpleNamespace(data=None))

    with pytest.raises(NotFoundError) as exc:
        api.get_file_bytes("/zlabel_server/projects/proj/zlabel/x.zlabel")

    assert exc.value.status_code == 404


def test_get_file_bytes_without_raw_url_raises_not_found(monkeypatch):
    api = FileSystemAPI.__new__(FileSystemAPI)
    monkeypatch.setattr(api, "get", lambda *a, **k: SimpleNamespace(data=SimpleNamespace(raw_url="")))

    with pytest.raises(NotFoundError) as exc:
        api.get_file_bytes("/missing.zlabel")

    assert exc.value.status_code == 404
