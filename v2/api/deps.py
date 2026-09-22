"""FastAPI dependencies shared by the v2 routers."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Header, Request

from v2.core.errors import Forbidden
from v2.services.auth_service import AuthContext
from v2.services.container import Services


def get_services(request: Request) -> Services:
    return request.app.state.services


def bearer_token(authorization: str | None) -> str:
    """``Authorization: Bearer <token>`` (a bare token is tolerated)."""
    if not authorization:
        return ""
    value = authorization.strip()
    if value.lower().startswith("bearer "):
        return value[7:].strip()
    return value


def get_token(authorization: str | None = Header(None)) -> str:
    return bearer_token(authorization)


def get_auth(
    request: Request,
    authorization: str | None = Header(None),
) -> AuthContext:
    """Resolve the session; raises ``unauthorized`` when it is missing/invalid."""
    return get_services(request).auth.resolve(bearer_token(authorization))


def require_roles(*roles: str) -> Callable[..., AuthContext]:
    """Dependency factory: only the given roles may call the endpoint."""

    def _dependency(request: Request, authorization: str | None = Header(None)) -> AuthContext:
        ctx = get_services(request).auth.resolve(bearer_token(authorization))
        if roles and ctx.role not in roles:
            raise Forbidden(f"role {'/'.join(roles)} required")
        return ctx

    return _dependency
