"""StatusService: probes, tolerant parsing and the ok/degraded/error rules."""

from __future__ import annotations

from datetime import UTC, datetime

import requests

from v2.services import status_service as mod


class FakeResponse:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _fake_worker(monkeypatch, health=200, metrics=200, health_payload=None, metrics_payload=None):
    payloads = {
        "http://worker/health": FakeResponse(health, health_payload if health_payload is not None else {}),
        "http://worker/metrics": FakeResponse(metrics, metrics_payload),
    }

    def fake_get(url, **_kwargs):
        return payloads[url]

    monkeypatch.setattr(mod.requests, "get", fake_get)


def test_snapshot_unconfigured_inference_is_neutral(services):
    started = datetime(2026, 1, 1, tzinfo=UTC)
    snap = services.status.snapshot(started_at=started)

    assert snap["server"]["status"] == "ok"
    assert snap["server"]["checks"]["db"]["status"] == "ok"
    assert snap["server"]["checks"]["storage"]["status"] == "ok"
    assert snap["inference"] == {"configured": False, "status": "unconfigured", "health": None, "metrics": None}
    assert snap["server"]["started_at"] == "2026-01-01T00:00:00+00:00"
    assert snap["server"]["uptime_s"] > 0
    assert snap["checked_at"]
    assert isinstance(snap["duration_ms"], int)


def test_db_error_aggregates_to_error(services, monkeypatch):
    monkeypatch.setattr(services.status, "check_db", lambda: {"status": "error", "message": "boom"})
    snap = services.status.snapshot()
    assert snap["server"]["status"] == "error"


def test_storage_error_aggregates_to_degraded(services, monkeypatch):
    monkeypatch.setattr(services.status, "check_storage", lambda: {"status": "error", "message": "readonly"})
    snap = services.status.snapshot()
    assert snap["server"]["status"] == "degraded"


def test_inference_unavailable_aggregates_to_degraded(services, monkeypatch):
    services.settings.inference_url = "http://worker"
    monkeypatch.setattr(
        mod.requests,
        "get",
        lambda url, **kwargs: (_ for _ in ()).throw(requests.RequestException("refused")),
    )
    snap = services.status.snapshot()
    assert snap["server"]["status"] == "degraded"
    assert snap["inference"]["status"] == "unavailable"
    assert "refused" in snap["inference"]["message"]


def test_worker_health_and_metrics_are_parsed_tolerantly(services, monkeypatch):
    services.settings.inference_url = "http://worker"
    _fake_worker(
        monkeypatch,
        health_payload={"status": "ok", "model": "SAM2", "backend": "CPU", "loaded": True},
        metrics_payload={"jobs": 7, "cache_hit_rate": 0.5, "latency_ms": {"p50": 3}, "uptime_s": 60},
    )
    snap = services.status.snapshot()

    assert snap["inference"]["status"] == "ok"
    assert snap["inference"]["health"]["model"] == "SAM2"
    assert snap["inference"]["health"]["queue"] == {}
    assert snap["inference"]["metrics"]["cache_hit_rate"] == 0.5
    assert snap["inference"]["metrics"]["latency_ms"] == {"p50": 3}
    assert snap["inference"]["metrics"]["uptime_s"] == 60
    assert snap["inference"]["metrics_error"] == ""


def test_metrics_failure_keeps_worker_ok(services, monkeypatch):
    services.settings.inference_url = "http://worker"
    _fake_worker(
        monkeypatch,
        health_payload={"status": "ok", "model": "SAM2"},
        metrics=401,
        metrics_payload={"detail": "unauthorized"},
    )
    snap = services.status.snapshot()

    assert snap["server"]["status"] == "ok"
    assert snap["inference"]["status"] == "ok"
    assert snap["inference"]["metrics"] is None
    assert snap["inference"]["metrics_error"] == "HTTP 401"


def test_health_contract_stays_stable(services, monkeypatch):
    services.settings.inference_url = "http://worker"
    monkeypatch.setattr(services.status, "check_storage", lambda: {"status": "ok"})
    monkeypatch.setattr(services.status, "check_inference", lambda: {"status": "unavailable"})
    body = services.status.health(deep=True, capabilities=["tasks.claim"])

    assert body["name"] == services.settings.app_name
    assert body["version"] == services.settings.version
    assert body["status"] == "degraded"
    assert body["capabilities"] == ["tasks.claim"]
    assert set(body["checks"]) == {"db", "storage", "inference"}
