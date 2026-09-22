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
    """Deep probes are aggregated and only a failure marks the app degraded."""
    monkeypatch.setattr(health_module, "_check_openlist", lambda s: {"status": "ok"})
    monkeypatch.setattr(health_module, "_check_inference", lambda s: {"status": "unavailable"})
    body = client.get("/api/v2/health", params={"deep": "true"}).json()

    assert set(body["checks"]) == {"db", "openlist", "inference"}
    assert body["checks"]["openlist"]["status"] == "ok"
    assert body["checks"]["inference"]["status"] == "unavailable"
    assert body["status"] == "degraded"


def test_health_probes_are_unconfigured_without_urls(settings):
    """Places that never configure OpenList/inference report "unconfigured"."""
    from v2.api.v2 import health as mod

    plain = type(settings)(database_url="sqlite+pysqlite:///:memory:", oplist_host="", inference_url="")
    assert mod._check_openlist(plain) == {"status": "unconfigured"}
    assert mod._check_inference(plain) == {"status": "unconfigured"}


def test_request_id_is_echoed(client):
    resp = client.get("/api/v2/health", headers={"X-Request-ID": "abc123"})
    assert resp.headers["X-Request-ID"] == "abc123"


def test_v1_era_clients_are_told_to_upgrade(client):
    """Old desktop builds must get an explanation, not a bare 404."""
    resp = client.get("/api/v1/get_tasks")
    assert resp.status_code == 410
    body = resp.json()
    assert body["code"] == "api_version_removed"
    assert "v2" in body["message"] and body["detail"]["path"] == "/api/v1/get_tasks"

    # and the endpoints that do exist are untouched
    assert client.get("/api/v2/health").status_code == 200
