"""Admin UI login: the same sessions as the API, carried in a cookie.

The desktop speaks ``Authorization: Bearer``; the admin UI needs cookies, so this
provider logs the admin in through ``AuthService`` and stores the session token in
an HttpOnly cookie. Authorization stays in one place (``AuthService`` + roles) - the
only extra rule is that the admin UI requires the ``admin`` role.
"""

from __future__ import annotations

import asyncio

from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette_admin.auth import AdminUser, AuthProvider, LoginFailed

from v2.core.errors import ApiError
from v2.core.logging import get_logger
from v2.services.auth_service import AuthContext
from v2.services.container import Services

logger = get_logger("zlabel.v2.admin")

COOKIE_NAME = "zl_admin"
COOKIE_MAX_AGE = 12 * 3600  # the session itself lives longer; the cookie is a work day


def admin_context(services: Services, request: Request) -> AuthContext | None:
    """The logged-in admin's context, for views that write through the services.

    The admin UI carries the same session token as the API, only in a cookie; the
    write hooks resolve it so audit rows name the admin who clicked the button.
    """
    token = str(request.cookies.get(COOKIE_NAME) or "")
    if not token:
        return None
    try:
        return services.auth.resolve(token)
    except ApiError:
        return None


class ZLabelAuthProvider(AuthProvider):
    """Cookie sessions for ``/admin``, restricted to the admin role."""

    def __init__(
        self, services: Services, *, base_url: str = "/admin", cookie_name: str = COOKIE_NAME
    ) -> None:
        super().__init__()
        self.services = services
        self.base_url = base_url.rstrip("/") or "/admin"
        self.cookie_name = cookie_name

    # region helpers
    def _token(self, request: Request) -> str:
        return str(request.cookies.get(self.cookie_name) or "")

    async def authenticate(self, request: Request) -> AdminUser | None:
        """Called on every admin request; ``None`` sends the visitor to the login."""
        token = self._token(request)
        if not token:
            return None
        try:
            ctx = await asyncio.to_thread(self.services.auth.resolve, token)
        except ApiError:
            return None
        if not ctx.is_admin:
            logger.warning(f"non-admin {ctx.name!r} tried to open the admin UI")
            return None
        return AdminUser(username=ctx.name)

    def get_admin_user(self, request: Request) -> str:
        """The logged-in admin's name (used by the templates)."""
        token = self._token(request)
        if not token:
            return ""
        try:
            return self.services.auth.resolve(token).name
        except ApiError:
            return ""

    # endregion

    # region login / logout
    async def login(
        self, username: str, password: str, remember_me: bool, request: Request
    ) -> Response | None:
        """Verify against the identity provider; ``None`` keeps the form open."""
        try:
            token, ctx, _expires = await asyncio.to_thread(
                self.services.auth.login, username, password, "admin-ui"
            )
        except ApiError as e:
            logger.info(f"admin login rejected: {e.message}")
            raise LoginFailed("invalid credentials") from e
        if not ctx.is_admin:
            logger.warning(f"admin login refused for {ctx.name!r}: role={ctx.role}")
            self.services.auth.logout(token)  # do not leave a usable session behind
            raise LoginFailed("this account is not an administrator")

        # the login form carries ?next=...; fall back to the admin index. Using a
        # path keeps this independent of starlette-admin's route names.
        target = str(request.query_params.get("next") or self.base_url)
        response = RedirectResponse(target, status_code=303)
        response.set_cookie(
            self.cookie_name,
            token,
            max_age=COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
            # enable this once /admin is served over https
            secure=request.url.scheme == "https",
            path=str(request.scope.get("root_path", "")) or "/",
        )
        logger.info(f"admin UI login: {ctx.name!r}")
        return response

    async def logout(self, request: Request) -> Response:
        token = self._token(request)
        if token:
            await asyncio.to_thread(self.services.auth.logout, token)
        response = RedirectResponse(f"{self.base_url}/login", status_code=303)
        response.delete_cookie(self.cookie_name, path=str(request.scope.get("root_path", "")) or "/")
        return response

    # endregion
