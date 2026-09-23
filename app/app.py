"""FastAPI application factory for v2.

Everything stateful is created inside :func:`create_app` (and stored on
``app.state``), so tests build an isolated app with an in-memory database instead
of monkeypatching module globals — the main structural fix over v1.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.v2 import (
    admin,
    annotations,
    auth,
    health,
    images,
    instances,
    internal,
    labels,
    members,
    predict,
    projects,
    tasks,
)
from app.core.config import Settings, get_settings
from app.core.errors import install_error_handlers
from app.core.logging import get_logger, set_request_id
from app.db.base import Database
from app.services.container import Services

API_PREFIX = "/api/v2"

logger = get_logger("zlabel.app.app")


def create_app(
    settings: Settings | None = None,
    database: Database | None = None,
    services: Services | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_dirs()
    db = database or Database(settings.database_url)

    def _bootstrap_admin() -> None:
        """Create the first admin from the environment (first run only)."""
        if not settings.bootstrap_admin:
            return
        if not settings.bootstrap_password:
            logger.warning(
                f"ZLSERVER_BOOTSTRAP_ADMIN={settings.bootstrap_admin!r} is set but "
                "ZLSERVER_BOOTSTRAP_PASSWORD is empty: keeping the database as it is"
            )
            return
        try:
            created = app.state.services.auth.identity.create_user(
                settings.bootstrap_admin, settings.bootstrap_password, admin=True, only_if_missing=True
            )
            logger.info(f"bootstrap admin {created['name']!r} is ready (role={created['role']})")
        except Exception as e:  # noqa: BLE001 - startup must not die on this
            logger.error(f"bootstrap admin failed: {e}")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        logger.info(f"v{settings.version} starting (db={db.url})")
        _app.state.started_at = datetime.now(UTC)
        _app.state.services.started_at = _app.state.started_at
        _bootstrap_admin()
        try:
            yield
        finally:
            db.dispose()

    app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan)
    app.state.settings = settings
    app.state.db = db
    app.state.services = services or Services.build(settings, db)

    install_error_handlers(app)

    # web administration UI (cookie sessions, admin role only)
    from app.admin import mount_admin

    mount_admin(app, app.state.services)

    @app.middleware("http")
    async def _request_id(request: Request, call_next):
        rid = set_request_id(request.headers.get("X-Request-ID"))
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    @app.api_route(
        "/api/v1/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"], include_in_schema=False
    )
    async def api_v1_removed(rest: str):
        """A v1-era client gets a clear "upgrade me", not a bare 404."""
        return JSONResponse(
            status_code=410,
            content={
                "code": "api_version_removed",
                "message": "This server only speaks /api/v2 (the v1 API was removed). Please update the desktop client.",
                "detail": {"path": f"/api/v1/{rest}"},
            },
        )

    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(auth.router, prefix=API_PREFIX)
    app.include_router(projects.router, prefix=API_PREFIX)
    app.include_router(labels.router, prefix=API_PREFIX)
    app.include_router(members.router, prefix=API_PREFIX)
    app.include_router(instances.router, prefix=API_PREFIX)
    app.include_router(admin.router, prefix=API_PREFIX)
    app.include_router(tasks.router, prefix=API_PREFIX)
    app.include_router(annotations.router, prefix=API_PREFIX)
    app.include_router(images.router, prefix=API_PREFIX)
    app.include_router(predict.router, prefix=API_PREFIX)
    app.include_router(internal.router, prefix=API_PREFIX)
    return app
