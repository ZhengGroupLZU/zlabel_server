"""Server status probes and aggregation for the admin dashboard.

Both surfaces that need a health picture use this service:

- ``GET /api/v2/health`` (the desktop's capability probe, contract unchanged)
- ``GET /admin/status`` (the dashboard's JSON poll endpoint)

The probes run sequentially and are never cached: DB/storage checks are
milliseconds, and the two inference HTTP calls are each capped at
:data:`PROBE_TIMEOUT`. The dashboard polls this asynchronously, so the
worst-case ~4-5 s only affects the JSON call, not page rendering.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from sqlalchemy import text

from app.core.logging import get_logger

PROBE_TIMEOUT = 2.0
CHECK_FAILED = ("error", "unavailable")

logger = get_logger("zlabel.app.status")


class StatusService:
    """Probe the server's dependencies and shape the results for the UI."""

    def __init__(self, settings: Any, db: Any) -> None:
        self.settings = settings
        self.db = db

    # region checks
    def check_db(self) -> dict[str, Any]:
        try:
            with self.db.session_scope() as session:
                session.execute(text("SELECT 1"))
            return {"status": "ok"}
        except Exception as e:  # noqa: BLE001 - a probe must never raise
            logger.warning(f"db probe failed: {e}")
            return {"status": "error", "message": str(e)}

    def check_storage(self) -> dict[str, Any]:
        """The storage tree must exist and be writable: everything else depends on it."""
        root = Path(self.settings.storage_root).expanduser()
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return {"status": "error", "root": str(root), "message": str(e)}
        if not root.is_dir():
            return {"status": "error", "root": str(root), "message": "not a directory"}
        probe = root / ".zlabel-health-probe"
        try:
            probe.write_bytes(b"ok")
            probe.unlink()
        except OSError as e:
            return {"status": "error", "root": str(root), "message": f"not writable: {e}"}
        return {"status": "ok", "root": str(root)}

    def check_inference(self) -> dict[str, Any]:
        """Probe the worker's ``/health``; the payload rides in ``detail``.

        ``unconfigured`` is not a failure (inference is optional), ``error``
        means the worker answered with a bad status, ``unavailable`` means the
        HTTP call itself failed.
        """
        if not self.settings.inference_url:
            return {"status": "unconfigured"}
        url = f"{self.settings.inference_url.rstrip('/')}/health"
        status, payload, message = self._probe(url, with_token=False)
        if message:
            return {"status": "unavailable", "message": message}
        if status != 200:
            return {"status": "error", "http": status}
        if not isinstance(payload, dict):
            return {"status": "unavailable", "message": "invalid worker /health payload"}
        return {"status": "ok", "detail": payload}

    # endregion

    # region shaping
    def health(self, deep: bool = False, capabilities: list[str] | None = None) -> dict[str, Any]:
        """The desktop's ``GET /api/v2/health`` payload (shape kept stable)."""
        checks: dict[str, Any] = {"db": self.check_db()}
        if deep:
            checks["storage"] = self.check_storage()
            checks["inference"] = self.check_inference()
        degraded = any(str(c.get("status")) in CHECK_FAILED for c in checks.values())
        return {
            "name": self.settings.app_name,
            "version": self.settings.version,
            "status": "degraded" if degraded else "ok",
            "capabilities": list(capabilities or []),
            "checks": checks,
        }

    def snapshot(self, started_at: datetime | None = None) -> dict[str, Any]:
        """The dashboard's structured status payload.

        Aggregation follows the agreed rules: a failed DB check is an ``error``,
        a failed storage check or an unavailable inference worker is
        ``degraded``, and an unconfigured worker is neutral.
        """
        started = time.perf_counter()
        db = self.check_db()
        storage = self.check_storage()
        inference = self.check_inference()
        server_status = "ok"
        if db.get("status") == "error":
            server_status = "error"
        elif storage.get("status") == "error":
            server_status = "degraded"
        elif inference.get("status") in CHECK_FAILED:
            server_status = "degraded"
        payload = {
            "server": {
                "name": self.settings.app_name,
                "version": self.settings.version,
                "status": server_status,
                "started_at": started_at.isoformat() if started_at else None,
                "uptime_s": int(time.time() - started_at.timestamp()) if started_at else None,
                "checks": {"db": db, "storage": storage},
            },
            "inference": self._inference_view(inference),
            "checked_at": datetime.now(UTC).isoformat(),
            "duration_ms": int((time.perf_counter() - started) * 1000),
        }
        return payload

    def _inference_view(self, check: dict[str, Any]) -> dict[str, Any]:
        configured = bool(self.settings.inference_url)
        if not configured:
            return {"configured": False, "status": "unconfigured", "health": None, "metrics": None}
        if check.get("status") != "ok":
            return {
                "configured": True,
                "status": str(check.get("status") or "unknown"),
                "health": None,
                "metrics": None,
                "message": str(
                    check.get("message") or (f"HTTP {check.get('http')}" if check.get("http") else "unknown")
                ),
            }
        health = self._parse_worker_health(check.get("detail") or {})
        metrics, metrics_error = self._worker_metrics()
        return {
            "configured": True,
            "status": "ok",
            "health": health,
            "metrics": metrics,
            "metrics_error": metrics_error,
        }

    # endregion

    # region worker http
    def _probe(self, url: str, *, with_token: bool) -> tuple[int, dict[str, Any] | None, str]:
        headers: dict[str, str] = {}
        if with_token and self.settings.inference_token:
            headers["Authorization"] = f"Bearer {self.settings.inference_token}"
        try:
            resp = requests.get(url, headers=headers, timeout=PROBE_TIMEOUT)
        except requests.RequestException as e:
            return 0, None, str(e)
        try:
            payload = resp.json()
        except Exception:
            payload = None
        return resp.status_code, payload if isinstance(payload, dict) else None, ""

    def _worker_metrics(self) -> tuple[dict[str, Any] | None, str]:
        """``/metrics`` is best-effort: a failure degrades the panel, not the worker."""
        if not self.settings.inference_url:
            return None, ""
        url = f"{self.settings.inference_url.rstrip('/')}/metrics"
        status, payload, message = self._probe(url, with_token=True)
        if message:
            return None, message
        if status != 200:
            return None, f"HTTP {status}"
        if not isinstance(payload, dict):
            return None, "invalid worker /metrics payload"
        return self._parse_worker_metrics(payload), ""

    @staticmethod
    def _parse_worker_health(payload: dict[str, Any]) -> dict[str, Any]:
        """Tolerant parse: unknown/missing fields keep defaults, never raise."""
        cache = payload.get("cache")
        queue = payload.get("queue")
        return {
            "status": str(payload.get("status") or "unknown"),
            "model": str(payload.get("model") or ""),
            "backend": str(payload.get("backend") or ""),
            "loaded": bool(payload.get("loaded", False)),
            "cache": cache if isinstance(cache, dict) else {},
            "queue": queue if isinstance(queue, dict) else {},
            "jobs": int(payload.get("jobs") or 0),
        }

    @staticmethod
    def _parse_worker_metrics(payload: dict[str, Any]) -> dict[str, Any]:
        latency = payload.get("latency_ms")
        return {
            "jobs": int(payload.get("jobs") or 0),
            "errors": int(payload.get("errors") or 0),
            "cache_hits": int(payload.get("cache_hits") or 0),
            "cache_misses": int(payload.get("cache_misses") or 0),
            "cache_hit_rate": payload.get("cache_hit_rate"),
            "latency_ms": latency if isinstance(latency, dict) else {},
            "uptime_s": int(payload.get("uptime_s") or 0),
        }

    # endregion
