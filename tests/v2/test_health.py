"""``GET /api/v2/health`` — capability probe used by the desktop client."""

from __future__ import annotations

from v2.api.v2 import health as health_module


def test_health_reports_capabilities(client):
    resp = client.get("/api/v2/health")
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "ok"
    assert body["version"]
    assert body["checks"]["db"]["status"] == "ok"
    # capability probe is how the client decides which UI to show
    assert "tasks.claim" in body["capabilities"]
    assert "tasks.review" in body["capabilities"]


def test_health_is_cheap_by_default(client):
    """The shallow probe must not touch OpenList / the inference worker."""
    body = client.get("/api/v2/health").json()
    assert set(body["checks"]) == {"db"}


def test_health_deep_reports_dependencies(client, monkeypatch):
    monkeypatch.setattr(health_module, "_check_inference", lambda s: {"status": "unavailable"})
    body = client.get("/api/v2/health", params={"deep": "true"}).json()

    assert set(body["checks"]) == {"db", "openlist", "inference"}
    assert body["checks"]["openlist"]["status"] == "unconfigured"
    assert body["checks"]["inference"]["status"] == "unavailable"
    assert body["status"] == "degraded"


def test_request_id_is_echoed(client):
    resp = client.get("/api/v2/health", headers={"X-Request-ID": "abc123"})
    assert resp.headers["X-Request-ID"] == "abc123"
