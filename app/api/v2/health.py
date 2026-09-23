"""Liveness / capability probe.

``GET /api/v2/health`` is what the desktop client calls right after login: it
reports what the server can do (``capabilities``) so the UI can hide the claim /
review / version features when it talks to an older server, and it aggregates the
dependency checks (db / storage / inference worker) without failing the request.

The checks themselves live in :class:`app.services.status_service.StatusService`,
which is also the data source of the admin dashboard's ``/admin/status``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request

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


@router.get("/health")
async def health(request: Request, deep: bool = False) -> dict[str, Any]:
    """``deep=false`` only checks the local database (fast, used on every login)."""
    status = request.app.state.services.status
    return await asyncio.to_thread(status.health, deep, CAPABILITIES)
