"""Liveness / capability probe.

``GET /api/v2/health`` is what the desktop client calls right after login: it
reports what the server can do (``capabilities``) so the UI can hide the claim /
review / version features when it talks to an older server, and it aggregates the
dependency checks (db / OpenList / inference worker) without failing the request.
"""

from __future__ import annotations

import asyncio
from typing import Any

import requests
from fastapi import APIRouter, Request
from sqlalchemy import text

from v2.core.config import Settings
from v2.core.logging import get_logger

logger = get_logger("zlabel.v2.health")

router = APIRouter(tags=["health"])

CAPABILITIES = [
    "auth.session",
    "projects.scan",
    "tasks.claim",
    "tasks.review",
    "tasks.sequence",
    "annotations.versions",
    "labels.write",
    "predict.stateless",
]

_PROBE_TIMEOUT = 2.0


def _check_db(request: Request) -> dict[str, Any]:
    try:
        db = request.app.state.db
        with db.session_scope() as session:
            session.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"db probe failed: {e}")
        return {"status": "error", "message": str(e)}


def _check_openlist(settings: Settings) -> dict[str, Any]:
    if not settings.oplist_host:
        return {"status": "unconfigured"}
    try:
        resp = requests.get(f"{settings.oplist_host.rstrip('/')}/api/public/settings", timeout=_PROBE_TIMEOUT)
        return {"status": "ok" if resp.status_code < 500 else "error", "http": resp.status_code}
    except Exception as e:
        return {"status": "unavailable", "message": str(e)}


def _check_inference(settings: Settings) -> dict[str, Any]:
    if not settings.inference_url:
        return {"status": "unconfigured"}
    try:
        headers = {"Authorization": f"Bearer {settings.inference_token}"} if settings.inference_token else {}
        resp = requests.get(
            f"{settings.inference_url.rstrip('/')}/health", timeout=_PROBE_TIMEOUT, headers=headers
        )
        if resp.status_code != 200:
            return {"status": "error", "http": resp.status_code}
        return {"status": "ok", "detail": resp.json()}
    except Exception as e:
        return {"status": "unavailable", "message": str(e)}


@router.get("/health")
async def health(request: Request, deep: bool = False) -> dict[str, Any]:
    """``deep=false`` only checks the local database (fast, used on every login)."""
    settings: Settings = request.app.state.settings
    checks: dict[str, Any] = {"db": await asyncio.to_thread(_check_db, request)}
    if deep:
        checks["openlist"] = await asyncio.to_thread(_check_openlist, settings)
        checks["inference"] = await asyncio.to_thread(_check_inference, settings)
    degraded = any(c.get("status") in ("error", "unavailable") for c in checks.values())
    return {
        "name": settings.app_name,
        "version": settings.version,
        "status": "degraded" if degraded else "ok",
        "capabilities": CAPABILITIES,
        "checks": checks,
    }
