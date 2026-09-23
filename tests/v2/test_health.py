"""``GET /api/v2/health`` — capability probe used by the desktop client."""

from __future__ import annotations

from v2.services.status_service import StatusService


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
    """The shallow probe must not touch the disk / the inference worker."""
    body = client.get("/api/v2/health").json()
    assert set(body["checks"]) == {"db"}


def test_health_deep_reports_dependencies(client, monkeypatch):
    """Deep probes are aggregated and only a failure marks the app degraded."""
    status = client.app.state.services.status
    monkeypatch.setattr(status, "check_storage", lambda: {"status": "ok"})
    monkeypatch.setattr(status, "check_inference", lambda: {"status": "unavailable"})
    body = client.get("/api/v2/health", params={"deep": "true"}).json()

    assert set(body["checks"]) == {"db", "storage", "inference"}
    assert body["checks"]["storage"]["status"] == "ok"
    assert body["checks"]["inference"]["status"] == "unavailable"
    assert body["status"] == "degraded"


def test_health_probes_are_unconfigured_without_urls(settings, db):
    """An unconfigured inference worker reports "unconfigured"."""
    plain = type(settings)(
        database_url="sqlite+pysqlite:///:memory:",
        storage_root=settings.storage_root,
        inference_url="",
    )
    status = StatusService(plain, db)
    assert status.check_storage()["status"] == "ok"
    assert status.check_inference() == {"status": "unconfigured"}


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
