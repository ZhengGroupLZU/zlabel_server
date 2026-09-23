"""``/api/v2/auth`` — login, current user, logout, user administration."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from app.api.deps import get_auth, get_services, get_token, require_roles
from app.schemas.auth import LoginRequest, LoginResponse, UserOut
from app.services.auth_service import AuthContext
from app.services.container import Services

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, services: Services = Depends(get_services)) -> LoginResponse:
    token, ctx, expires_at = services.auth.login(payload.username, payload.password, payload.client)
    with services.db.session_scope() as session:
        from app.db.models import User

        user = session.get(User, ctx.user_id)
        return LoginResponse(token=token, expires_at=expires_at, user=UserOut.of(user))


@router.get("/me", response_model=UserOut)
def me(auth: AuthContext = Depends(get_auth), services: Services = Depends(get_services)) -> UserOut:
    with services.db.session_scope() as session:
        from app.db.models import User

        return UserOut.of(session.get(User, auth.user_id))


@router.post("/logout", status_code=204)
def logout(token: str = Depends(get_token), services: Services = Depends(get_services)) -> Response:
    services.auth.logout(token)
    return Response(status_code=204)


@router.get("/users", response_model=list[UserOut])
def list_users(
    _auth: AuthContext = Depends(require_roles("admin")),
    services: Services = Depends(get_services),
) -> list[UserOut]:
    return [UserOut.of(u) for u in services.auth.list_users()]


@router.put("/users/{user_id}/role", response_model=UserOut)
def set_role(
    user_id: int,
    role: str,
    auth: AuthContext = Depends(require_roles("admin")),
    services: Services = Depends(get_services),
) -> UserOut:
    return UserOut.of(services.auth.set_role(auth, user_id, role))
