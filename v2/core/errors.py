"""Unified error model for v2.

Success responses are plain resources; failures are always
``{"code": <machine readable>, "message": <human>, "detail": <optional>}``.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ApiError(Exception):
    """Base class for all v2 errors."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str = "", detail: Any = None) -> None:
        super().__init__(message or self.code)
        self.message = message or self.code
        self.detail = detail

    def payload(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "detail": self.detail}


class Unauthorized(ApiError):
    status_code = 401
    code = "unauthorized"


class SessionStale(ApiError):
    """The session exists locally but the upstream (OpenList) token is gone."""

    status_code = 401
    code = "session_stale"


class Forbidden(ApiError):
    status_code = 403
    code = "forbidden"


class NotFound(ApiError):
    status_code = 404
    code = "not_found"


class Conflict(ApiError):
    status_code = 409
    code = "conflict"


class LeaseConflict(Conflict):
    code = "lease_conflict"


class ValidationFailed(ApiError):
    status_code = 422
    code = "validation_error"


class UpstreamError(ApiError):
    status_code = 502
    code = "upstream_error"


class InferenceUnavailable(ApiError):
    status_code = 503
    code = "inference_unavailable"


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_request: Request, exc: ApiError) -> JSONResponse:  # noqa: RUF029
        return JSONResponse(status_code=exc.status_code, content=exc.payload())

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "code": "validation_error",
                "message": "request validation failed",
                "detail": exc.errors(),
            },
        )
