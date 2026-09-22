"""predict / set_image are token-gated (batch B5)."""

from __future__ import annotations

import importlib
import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def appmod():
    from app import config

    old_name = config.SETTINGS.model_name
    config.SETTINGS.model_name = "EdgeSAM"
    try:
        yield importlib.import_module("app.app")
    finally:
        config.SETTINGS.model_name = old_name


class _UserData:
    id = 1


class _UserResponse:
    data = _UserData()


@pytest.fixture
def anon_client(appmod, fake_predictor):
    """No Authorization header at all."""
    appmod.SAM_MODEL = fake_predictor
    appmod._auth_cache.clear()
    return TestClient(appmod.app)


@pytest.fixture
def token_client(appmod, monkeypatch):
    monkeypatch.setattr(appmod.oplist_client, "set_token", lambda token: None)
    monkeypatch.setattr(appmod.oplist_client.auth, "get_current_user", lambda: _UserResponse())
    appmod._auth_cache.clear()
    return TestClient(appmod.app, headers={"Authorization": "test-token"})


def test_set_image_without_token_is_rejected(anon_client, img_path):
    r = anon_client.post(
        "/api/v1/set_image",
        files={"image": ("x.png", img_path.read_bytes(), "image/png")},
        data={"image_name": "x.png"},
    )
    assert r.status_code == 401


def test_predict_without_token_is_rejected(anon_client, img_path):
    r = anon_client.post(
        "/api/v1/predict",
        files={"image": ("x.png", img_path.read_bytes(), "image/png")},
        data={
            "data": json.dumps({"id": "abc", "rects": [{"x": 400, "y": 200, "w": 120, "h": 120}]}),
            "image_name": "x.png",
            "project": "projA",
        },
    )
    assert r.status_code == 401


def test_invalid_token_is_rejected(appmod, anon_client, monkeypatch, img_path):
    def boom():
        raise RuntimeError("bad token")

    monkeypatch.setattr(appmod.oplist_client, "set_token", lambda token: None)
    monkeypatch.setattr(appmod.oplist_client.auth, "get_current_user", boom)
    appmod._auth_cache.clear()

    r = anon_client.post(
        "/api/v1/set_image",
        headers={"Authorization": "bad-token"},
        files={"image": ("x.png", img_path.read_bytes(), "image/png")},
        data={"image_name": "x.png"},
    )
    assert r.status_code == 401


def test_valid_token_is_verified_once(appmod, token_client, monkeypatch, img_path):
    calls: list[int] = []

    def counting_get_current_user():
        calls.append(1)
        return _UserResponse()

    monkeypatch.setattr(appmod.oplist_client.auth, "get_current_user", counting_get_current_user)
    appmod._auth_cache.clear()

    for _ in range(2):
        r = token_client.post(
            "/api/v1/set_image",
            files={"image": ("x.png", img_path.read_bytes(), "image/png")},
            data={"image_name": "x.png"},
        )
        assert r.status_code == 200

    assert len(calls) == 1  # second call served from the token cache
