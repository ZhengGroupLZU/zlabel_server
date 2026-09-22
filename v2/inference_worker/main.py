"""The inference worker: its own process, its own model.

    uv run fastapi run v2/inference_worker/main.py --port 8001

The API talks to it over ``POST /infer`` with the shared secret
(``ZLSERVER_INFERENCE_TOKEN``). Keeping the model out of the API process means an API
restart never reloads a multi-GB model and the GPU queue never blocks HTTP
workers.
"""

from __future__ import annotations

import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header

from inference.config import InferenceSettings, get_inference_settings
from v2.api.deps import bearer_token
from v2.core.errors import Forbidden, Unauthorized, install_error_handlers
from v2.core.logging import get_logger
from v2.inference_worker.engine import InferenceEngine
from v2.inference_worker.schemas import HealthOut, InferJob, MetricsOut

logger = get_logger("zlabel.v2.worker")


def create_worker_app(
    settings: InferenceSettings | None = None,
    engine: InferenceEngine | None = None,
) -> FastAPI:
    settings = settings or get_inference_settings()
    engine = engine or InferenceEngine(settings)

    app = FastAPI(title="ZLabel Inference Worker", version="2.0.0")
    app.state.settings = settings
    app.state.engine = engine
    install_error_handlers(app)

    def require_worker_token(authorization: str | None = Header(None)) -> None:
        """Both directions share one secret; an unconfigured secret is a hard stop."""
        expected = settings.inference_token
        if not expected:
            raise Forbidden("ZLSERVER_INFERENCE_TOKEN is not configured on the worker")
        if not secrets.compare_digest(bearer_token(authorization), expected):
            raise Unauthorized("bad worker token")

    @app.post("/infer")
    def infer(job: InferJob, _token: None = Depends(require_worker_token)) -> dict:
        return engine.run(job)

    @app.get("/health", response_model=HealthOut)
    def health() -> dict:
        """Unauthenticated liveness probe (no secrets in the payload)."""
        return engine.health()

    @app.get("/metrics", response_model=MetricsOut)
    def metrics(_token: None = Depends(require_worker_token)) -> dict:
        return engine.metrics_snapshot()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        logger.info(
            f"inference worker up: model={settings.model_name} backend={settings.ort_backend} "
            f"cache={settings.embedding_cache_size} concurrency={settings.max_concurrency}"
        )
        yield

    app.router.lifespan_context = lifespan
    return app


app = create_worker_app()
