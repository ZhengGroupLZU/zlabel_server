"""FastAPI application factory for v2.

Everything stateful is created inside :func:`create_app` (and stored on
``app.state``), so tests build an isolated app with an in-memory database instead
of monkeypatching module globals — the main structural fix over v1.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from v2.api.v2 import annotations, auth, health, images, internal, labels, predict, projects, tasks
from v2.core.config import Settings, get_settings
from v2.core.errors import install_error_handlers
from v2.core.logging import get_logger, set_request_id
from v2.db.base import Database
from v2.services.container import Services

API_PREFIX = "/api/v2"

logger = get_logger("zlabel.v2.app")


def create_app(
    settings: Settings | None = None,
    database: Database | None = None,
    services: Services | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    settings.ensure_dirs()
    db = database or Database(settings.database_url)

    async def _scan(app: FastAPI) -> None:
        try:
            projects = app.state.services.projects
            await asyncio.to_thread(lambda: projects.scan_and_sync(force=True))
        except Exception as e:  # noqa: BLE001 - startup must never die on OpenList
            logger.warning(f"project scan unavailable: {e}")

    async def _periodic_scan(app: FastAPI) -> None:
        while True:
            await asyncio.sleep(settings.project_scan_interval)
            await _scan(app)

    def _bootstrap_admin() -> None:
        """Create the first local admin from the environment (first run only)."""
        if settings.identity != "local" or not settings.bootstrap_admin:
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
    async def lifespan(app: FastAPI):
        logger.info(f"v{settings.version} starting (db={db.url})")
        _bootstrap_admin()
        tasks: list[asyncio.Task] = []
        if settings.scan_on_startup:
            tasks.append(asyncio.create_task(_scan(app)))
        if settings.project_scan_interval > 0:
            tasks.append(asyncio.create_task(_periodic_scan(app)))
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with suppress(asyncio.CancelledError):
                    await task
            db.dispose()

    app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan)
    app.state.settings = settings
    app.state.db = db
    app.state.services = services or Services.build(settings, db)

    install_error_handlers(app)

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
    app.include_router(tasks.router, prefix=API_PREFIX)
    app.include_router(annotations.router, prefix=API_PREFIX)
    app.include_router(images.router, prefix=API_PREFIX)
    app.include_router(predict.router, prefix=API_PREFIX)
    app.include_router(internal.router, prefix=API_PREFIX)
    return app
