"""save_zlabel contract: DB state only advances when the upload really worked."""

from __future__ import annotations

import importlib
import json

import pytest
from fastapi.testclient import TestClient


def _install_auth(appmod, monkeypatch) -> dict[str, str]:
    """Stub the OpenList token check used by ``require_user``."""
    class _UserData:
        id = 1

    class _UserResponse:
        data = _UserData()

    monkeypatch.setattr(appmod.oplist_client, "set_token", lambda token: None)
    monkeypatch.setattr(appmod.oplist_client.auth, "get_current_user", lambda: _UserResponse())
    appmod._auth_cache.clear()
    return {"Authorization": "test-token"}


def _not_found():
    """OpenList 404 for detect_conflict (no annotation stored yet)."""
    from app.openlist_api import OpenListAPIError

    exc = OpenListAPIError("not found", status_code=404)
    raise exc


@pytest.fixture(scope="module")
def appmod():
    from app import config

    old_name = config.SETTINGS.model_name
    config.SETTINGS.model_name = "EdgeSAM"
    try:
        yield importlib.import_module("app.app")
    finally:
        config.SETTINGS.model_name = old_name


class _UploadResponse:
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        self.data = None


@pytest.fixture
def client(appmod, fake_predictor, monkeypatch):
    appmod.SAM_MODEL = fake_predictor
    headers = _install_auth(appmod, monkeypatch)
    return TestClient(appmod.app, headers=headers)


@pytest.fixture
def recorded(appmod, monkeypatch):
    """Record upload calls and DB inserts; run a fake oplist client."""
    from app import db

    calls = {"uploads": [], "links": []}

    def fake_upload(file_path, *_args, **_kwargs):
        calls["uploads"].append(file_path)
        return _UploadResponse(*calls["upload_result"])

    def fake_link(anno_id, *_args, **kwargs):
        calls["links"].append((anno_id, kwargs.get("user_name", ""), kwargs.get("label_names")))

    monkeypatch.setattr(appmod.oplist_client, "set_token", lambda token: None)
    monkeypatch.setattr(appmod.oplist_client.fs, "stream_upload", fake_upload)
    monkeypatch.setattr(appmod.oplist_client.fs, "get_file_bytes", lambda *a, **k: _not_found())
    monkeypatch.setattr(db, "insert_link_table", fake_link)
    calls["upload_result"] = (200, "success")
    calls["stored"] = None  # optionally: the annotation already on the server
    return calls


def test_upload_success_persists_link(client, recorded):
    r = client.put(
        "/api/v1/save_zlabel",
        data={"username": "rainy", "zlabel": '{"id": "abc123"}', "filename": "abc123.zlabel"},
    )
    assert r.status_code == 200
    assert r.json()["message"] == "success"
    assert recorded["uploads"]  # uploaded to the zlabel dir
    assert recorded["links"] == [("abc123", "rainy", [])]  # task marked finished


def test_upload_failure_keeps_task_unfinished(client, recorded):
    recorded["upload_result"] = (500, "disk full")

    r = client.put(
        "/api/v1/save_zlabel",
        data={"username": "rainy", "zlabel": '{"id": "abc123"}', "filename": "abc123.zlabel"},
    )
    assert r.status_code == 502
    assert r.json()["message"] == "disk full"
    assert recorded["links"] == []  # DB untouched: the task stays unfinished


def test_invalid_json_is_rejected_without_link(client, recorded):
    r = client.put(
        "/api/v1/save_zlabel",
        data={"username": "rainy", "zlabel": "not json", "filename": "abc123.zlabel"},
    )
    assert r.status_code == 422
    assert recorded["links"] == []


def test_upload_links_the_annotation_labels(client, recorded):
    anno = {
        "id": "abc123",
        "results": {
            "r1": {"labels": [{"name": "Root"}, {"name": "Shoot"}]},
            "r2": {"labels": [{"name": "Root"}]},
        },
    }
    r = client.put(
        "/api/v1/save_zlabel",
        data={"username": "rainy", "zlabel": json.dumps(anno), "filename": "abc123.zlabel"},
    )
    assert r.status_code == 200
    assert recorded["links"] == [("abc123", "rainy", ["Root", "Shoot"])]


def _stored(updated_at: str, updated_by: str = "someone-else"):
    return json.dumps({
        "id": "abc123",
        "updated_at": updated_at,
        "updated_by": {"name": updated_by},
    }).encode("utf-8")


def test_newer_server_annotation_returns_conflict(client, recorded, appmod, monkeypatch):
    """Optimistic lock: never overwrite a newer annotation on the server."""
    monkeypatch.setattr(appmod.oplist_client.fs, "get_file_bytes", lambda *a, **k: _stored("2099-01-01T00:00:00"))
    incoming = {"id": "abc123", "updated_at": "2020-01-01T00:00:00"}

    r = client.put(
        "/api/v1/save_zlabel",
        data={"username": "rainy", "zlabel": json.dumps(incoming), "filename": "abc123.zlabel"},
    )

    assert r.status_code == 409
    body = r.json()
    assert body["message"] == "annotation conflict"
    assert body["data"]["updated_at"] == "2099-01-01T00:00:00"
    assert body["data"]["updated_by"] == "someone-else"
    assert recorded["uploads"] == []  # nothing was written
    assert recorded["links"] == []


def test_newer_local_annotation_wins(client, recorded, appmod, monkeypatch):
    monkeypatch.setattr(appmod.oplist_client.fs, "get_file_bytes", lambda *a, **k: _stored("2020-01-01T00:00:00"))
    incoming = {"id": "abc123", "updated_at": "2030-01-01T00:00:00"}

    r = client.put(
        "/api/v1/save_zlabel",
        data={"username": "rainy", "zlabel": json.dumps(incoming), "filename": "abc123.zlabel"},
    )

    assert r.status_code == 200
    assert recorded["uploads"]


def test_force_overwrites_a_newer_server_annotation(client, recorded, appmod, monkeypatch):
    monkeypatch.setattr(appmod.oplist_client.fs, "get_file_bytes", lambda *a, **k: _stored("2099-01-01T00:00:00"))
    incoming = {"id": "abc123", "updated_at": "2020-01-01T00:00:00"}

    r = client.put(
        "/api/v1/save_zlabel",
        data={
            "username": "rainy",
            "zlabel": json.dumps(incoming),
            "filename": "abc123.zlabel",
            "force": "true",
        },
    )

    assert r.status_code == 200
    assert recorded["uploads"]


def test_legacy_annotation_without_timestamps_still_saves(client, recorded, appmod, monkeypatch):
    monkeypatch.setattr(appmod.oplist_client.fs, "get_file_bytes", lambda *a, **k: json.dumps({"id": "abc123"}).encode())
    r = client.put(
        "/api/v1/save_zlabel",
        data={"username": "rainy", "zlabel": json.dumps({"id": "abc123"}), "filename": "abc123.zlabel"},
    )
    assert r.status_code == 200
    assert recorded["uploads"]
