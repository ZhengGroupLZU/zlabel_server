"""FastAPI application factory for v2.

Everything stateful is created inside :func:`create_app` (and stored on
``app.state``), so tests build an isolated app with an in-memory database instead
of monkeypatching module globals — the main structural fix over v1.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request

from v2.api.v2 import annotations, auth, health, images, labels, predict, projects, tasks
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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(f"v{settings.version} starting (db={db.url})")
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

    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(auth.router, prefix=API_PREFIX)
    app.include_router(projects.router, prefix=API_PREFIX)
    app.include_router(labels.router, prefix=API_PREFIX)
    app.include_router(tasks.router, prefix=API_PREFIX)
    app.include_router(annotations.router, prefix=API_PREFIX)
    app.include_router(images.router, prefix=API_PREFIX)
    app.include_router(predict.router, prefix=API_PREFIX)
    return app
