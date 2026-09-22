"""FastAPI endpoint tests with a fake model (fast, hermetic).

Covers the image-cache flow (set_image -> predict cache hit), direct image
upload, the get_image bytes-serialization regression, and prompt routing.
"""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def appmod():
    # use the smallest model at import time so module-level SAM_MODEL creation is cheap
    from app import config

    old_name = config.SETTINGS.model_name
    config.SETTINGS.model_name = "EdgeSAM"
    try:
        yield importlib.import_module("app.app")
    finally:
        config.SETTINGS.model_name = old_name


def _install_auth(appmod, monkeypatch) -> dict[str, str]:
    """Stub the OpenList token check used by ``require_user``.

    Also makes ``get_file_bytes`` report "not stored yet" so the save endpoint's
    optimistic-lock lookup does not hit a real server.
    """
    from app.openlist_api import OpenListAPIError

    class _UserData:
        id = 1

    class _UserResponse:
        data = _UserData()

    def _not_found(*_args, **_kwargs):
        raise OpenListAPIError("not found", status_code=404)

    monkeypatch.setattr(appmod.oplist_client, "set_token", lambda token: None)
    monkeypatch.setattr(appmod.oplist_client.auth, "get_current_user", lambda: _UserResponse())
    monkeypatch.setattr(appmod.oplist_client.fs, "get_file_bytes", _not_found)
    appmod._auth_cache.clear()
    return {"Authorization": "test-token"}


@pytest.fixture
def client(appmod, fake_predictor, monkeypatch):
    appmod.SAM_MODEL = fake_predictor
    headers = _install_auth(appmod, monkeypatch)
    return TestClient(appmod.app, headers=headers)


@pytest.fixture
def img_bytes(img_path) -> bytes:
    return img_path.read_bytes()


class TestSetImage:
    def test_without_image_name(self, client, appmod, fake_predictor, img_bytes):
        r = client.post("/api/v1/set_image", files={"image": ("t.jpg", img_bytes, "image/jpeg")})
        assert r.status_code == 200
        assert fake_predictor.set_image_calls, "model image should be set"
        assert "no_name" not in appmod.IMAGE_CACHE

    def test_with_image_name_caches(self, client, appmod, fake_predictor, img_bytes):
        r = client.post(
            "/api/v1/set_image",
            files={"image": ("t.jpg", img_bytes, "image/jpeg")},
            data={"image_name": "cached_img"},
        )
        assert r.status_code == 200
        assert "cached_img" in appmod.IMAGE_CACHE
        assert appmod.IMAGE_CACHE["cached_img"] == img_bytes

    def test_cache_eviction(self, client, appmod, fake_predictor, img_bytes):
        appmod.IMAGE_CACHE.clear()
        appmod.IMAGE_CACHE.update({f"img_{i}": b"x" for i in range(appmod.SETTINGS.image_cache_size)})
        client.post(
            "/api/v1/set_image",
            files={"image": ("t.jpg", img_bytes, "image/jpeg")},
            data={"image_name": "new_img"},
        )
        assert "new_img" in appmod.IMAGE_CACHE
        assert len(appmod.IMAGE_CACHE) <= appmod.SETTINGS.image_cache_size


class TestGetImage:
    def test_cache_hit(self, client, appmod, img_bytes):
        appmod.IMAGE_CACHE["cached_img"] = img_bytes
        r = client.get("/api/v1/get_image?name=cached_img")
        assert r.status_code == 200
        assert r.content == img_bytes

    def test_cache_miss_returns_json_error(self, client, appmod, monkeypatch):
        def boom(*_args, **_kwargs):  # noqa: ARG001
            raise RuntimeError("oplist unavailable")

        monkeypatch.setattr(appmod.oplist_client.fs, "get_file_bytes", boom)
        r = client.get("/api/v1/get_image?name=missing.png")
        assert r.status_code == 500
        body = r.json()
        assert isinstance(body.get("message"), str)


def test_get_image_missing_is_404(client, appmod, monkeypatch):
    """A missing image must pass the OpenList 404 through (not a 500)."""
    from app.openlist_api.exceptions import NotFoundError

    def _missing(*_args, **_kwargs):
        raise NotFoundError("not found: /missing.png", status_code=404)

    monkeypatch.setattr(appmod.oplist_client.fs, "get_file_bytes", _missing)
    appmod.IMAGE_CACHE.clear()

    r = client.get("/api/v1/get_image", params={"name": "/missing.png"})

    assert r.status_code == 404


class TestPredict:
    def _predict(self, client, anno: str, image_name: str = "img", image_bytes=None):
        files = {"image": (image_name, image_bytes, "image/jpeg")} if image_bytes else None
        data = {
            "data": anno,
            "threshold": "100",
            "mode": "1",
            "image_name": image_name,
            "return_type": "1",
        }
        return client.post("/api/v1/predict", data=data, files=files)

    def test_direct_image_upload(self, client, appmod, monkeypatch, img_bytes):
        called = []

        def boom(*_args, **_kwargs):  # noqa: ARG001
            called.append(True)
            raise AssertionError("oplist must not be called")

        monkeypatch.setattr(appmod.oplist_client.fs, "get_file_bytes", boom)
        r = self._predict(client, '{"id":"p1","points":[{"x":400,"y":300}],"labels":[1]}', image_bytes=img_bytes)
        assert r.status_code == 200
        d = r.json()["data"]
        assert d["status"] is True
        assert len(d["data"]) > 0
        assert called == []

    def test_cached_image_name(self, client, appmod, monkeypatch, img_bytes):
        called = []

        def boom(*_args, **_kwargs):  # noqa: ARG001
            called.append(True)
            raise AssertionError("oplist must not be called")

        monkeypatch.setattr(appmod.oplist_client.fs, "get_file_bytes", boom)
        client.post(
            "/api/v1/set_image",
            files={"image": ("t.jpg", img_bytes, "image/jpeg")},
            data={"image_name": "cached_predict"},
        )
        r = self._predict(client, '{"id":"p1","points":[{"x":400,"y":300}],"labels":[1]}', image_name="cached_predict")
        assert r.status_code == 200
        assert r.json()["data"]["status"] is True
        assert called == []

    def test_text_prompt(self, client, img_bytes):
        r = self._predict(client, '{"id":"t1","texts":["person"]}', image_bytes=img_bytes)
        assert r.status_code == 200
        assert r.json()["data"]["status"] is True

    def test_rect_prompt(self, client, img_bytes):
        r = self._predict(client, '{"id":"r1","rects":[{"x":100,"y":100,"w":400,"h":400}]}', image_bytes=img_bytes)
        assert r.status_code == 200
        assert r.json()["data"]["status"] is True

    def test_missing_image_readable_error(self, client, appmod, monkeypatch):
        # regression: resp.body (bytes) must not be stuffed into JSONResponse content
        def boom(*_args, **_kwargs):  # noqa: ARG001
            raise RuntimeError("no such file")

        monkeypatch.setattr(appmod.oplist_client.fs, "get_file_bytes", boom)
        r = self._predict(client, '{"id":"x1","points":[{"x":1,"y":1}],"labels":[1]}', image_name="missing.png")
        assert r.status_code == 500
        body = r.json()
        assert isinstance(body.get("message"), str)
        assert "no such file" in body["message"].lower()

    def test_invalid_annotation(self, client, img_bytes):
        # all prompt fields None -> graceful failure (200 with status False)
        r = self._predict(client, '{"id":"x1"}', image_bytes=img_bytes)
        assert r.status_code == 200
        assert r.json()["data"]["status"] is False


class TestSaveZLabel:
    def _save(self, client, data: dict):
        zlabel = json.dumps({"id": "anno1", "slots": []}).encode("utf-8")
        # filename=None -> sent as a bytes form field (matches `zlabel: bytes = Form`)
        files = {"zlabel": (None, zlabel, "application/json")}
        return client.put("/api/v1/save_zlabel", files=files, data=data)

    def test_uses_project_dir(self, client, appmod, monkeypatch):
        captured = {}

        def fake_upload(file_path, file_data, as_task=True):  # noqa: ARG001
            captured["path"] = file_path
            return SimpleNamespace(code=200, message="success")

        monkeypatch.setattr(appmod.oplist_client.fs, "stream_upload", fake_upload)
        monkeypatch.setattr(appmod.db, "insert_link_table", lambda *a, **k: None)
        r = self._save(client, {"username": "alice", "filename": "label.json", "project": "projX"})
        assert r.status_code == 200
        assert captured["path"] == f"{appmod.SETTINGS.oplist_proj_dir}/projX/zlabel/label.json"

    def test_defaults_to_config_project(self, client, appmod, monkeypatch):
        captured = {}

        def fake_upload(file_path, file_data, as_task=True):  # noqa: ARG001
            captured["path"] = file_path
            return SimpleNamespace(code=200, message="success")

        monkeypatch.setattr(appmod.oplist_client.fs, "stream_upload", fake_upload)
        monkeypatch.setattr(appmod.db, "insert_link_table", lambda *a, **k: None)
        # no ``project`` provided -> falls back to configured oplist_proj_name
        r = self._save(client, {"username": "alice", "filename": "label.json"})
        assert r.status_code == 200
        assert captured["path"] == (
            f"{appmod.SETTINGS.oplist_proj_dir}/{appmod.SETTINGS.oplist_proj_name}/zlabel/label.json"
        )

    def test_upload_failure_is_a_502_and_skips_the_db(self, client, appmod, monkeypatch):
        def boom(*_args, **_kwargs):  # noqa: ARG001
            return SimpleNamespace(code=500, message="push error")

        linked: list[tuple] = []
        monkeypatch.setattr(appmod.oplist_client.fs, "stream_upload", boom)
        monkeypatch.setattr(appmod.db, "insert_link_table", lambda *a, **k: linked.append(a))
        r = self._save(client, {"username": "alice", "filename": "label.json", "project": "projX"})
        assert r.status_code == 502
        assert r.json()["message"] == "push error"
        assert linked == []  # the task must stay unfinished


class TestGetZLabel:
    def test_uses_project_dir(self, client, appmod, monkeypatch):
        captured = {}

        def fake_get(path):
            captured["path"] = path
            return b'{"id":"anno1","slots":[]}'

        monkeypatch.setattr(appmod.oplist_client.fs, "get_file_bytes", fake_get)
        r = client.get("/api/v1/get_zlabel", params={"name": "label.json", "project": "projX"})
        assert r.status_code == 200
        assert captured["path"] == f"{appmod.SETTINGS.oplist_proj_dir}/projX/zlabel/label.json"
        assert r.json() == {"id": "anno1", "slots": []}

    def test_defaults_to_config_project(self, client, appmod, monkeypatch):
        captured = {}

        def fake_get(path):
            captured["path"] = path
            return b'{"id":"anno1"}'

        monkeypatch.setattr(appmod.oplist_client.fs, "get_file_bytes", fake_get)
        r = client.get("/api/v1/get_zlabel", params={"name": "label.json"})
        assert r.status_code == 200
        assert captured["path"] == (
            f"{appmod.SETTINGS.oplist_proj_dir}/{appmod.SETTINGS.oplist_proj_name}/zlabel/label.json"
        )

    def test_missing_annotation_is_404(self, client):
        """The desktop keys "not annotated yet" off 404; it must not be a 500."""
        r = client.get("/api/v1/get_zlabel", params={"name": "x.zlabel", "project": "projA"})
        assert r.status_code == 404
