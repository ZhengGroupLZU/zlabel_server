"""Config plumbing: one env prefix, one shared secret for API and worker."""

from __future__ import annotations

from inference.config import InferenceSettings
from v2.core.config import Settings


def test_env_prefix_is_zlabelserver(monkeypatch):
    monkeypatch.setenv("ZLSERVER_DATABASE_URL", "sqlite+pysqlite:///./data/x.db")
    monkeypatch.setenv("ZLSERVER_LEASE_MINUTES", "7")
    monkeypatch.setenv("ZLSERVER_MODEL_NAME", "SAM3")
    monkeypatch.setenv("ZLSERVER_INFERENCE_TOKEN", "shared-secret")
    monkeypatch.setenv("ZLSERVER_INFERENCE_URL", "http://worker:8001")

    api = Settings()
    worker = InferenceSettings()

    assert api.database_url.endswith("x.db") and api.lease_minutes == 7
    assert api.inference_url == "http://worker:8001"
    # both processes read the *same* variable: a mismatch here means the worker
    # would reject every request with "token not configured"
    assert api.inference_token == worker.inference_token == "shared-secret"
    assert worker.model_name == "SAM3"


def test_worker_refuses_to_serve_without_a_secret():
    from fastapi.testclient import TestClient

    from v2.inference_worker.main import create_worker_app

    app = create_worker_app(InferenceSettings(inference_token=""))
    resp = TestClient(app).post("/infer", json={"anno_id": "a", "prompts": {}})
    assert resp.status_code == 403
    assert "ZLSERVER_INFERENCE_TOKEN" in resp.json()["message"]


def test_the_env_file_is_the_default_v2_file(shipped_env_file):
    """A stale v1 ``.env.onnx`` must never be read (see tests/conftest.py)."""
    from inference.config import InferenceSettings
    from v2.core.config import Settings

    assert shipped_env_file["api"] == shipped_env_file["worker"] == ".env.v2"
    assert Settings.model_config["env_prefix"] == "ZLSERVER_"
    assert InferenceSettings.model_config["env_prefix"] == "ZLSERVER_"
