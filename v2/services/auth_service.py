"""Login, sessions and role checks.

OpenList stays the identity source: credentials are verified there, the user's
OpenList token is kept *inside* the server session (so file access keeps that
user's own ACLs) and the client receives our own opaque session token. Passwords
are never stored.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from sqlalchemy import func, select

from v2.adapters.identity import IdentityProvider
from v2.core.config import Settings
from v2.core.errors import Forbidden, Unauthorized
from v2.core.logging import get_logger
from v2.db.base import Database
from v2.db.models import ROLE_ADMIN, ROLE_ANNOTATOR, ROLES, Session, User, utcnow
from v2.services import audit

logger = get_logger("zlabel.v2.auth")

TOKEN_BYTES = 32


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AuthContext:
    """A resolved session: who is calling and which OpenList token to use."""

    session_id: int
    user_id: int
    name: str
    role: str
    oplist_token: str
    expires_at: datetime

    @property
    def is_reviewer(self) -> bool:
        return self.role in ("reviewer", "admin")

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def with_role(self, role: str) -> AuthContext:
        """The same session, but with the role that applies *here*.

        Project membership (P4) can differ from the global role, so routers resolve
        the effective role once and hand the service layer a context that carries
        it - the service's own ``require_reviewer`` then means "reviewer of this
        project" without knowing about memberships.
        """
        return replace(self, role=role or self.role) if role else self

    def require_reviewer(self) -> None:
        if not self.is_reviewer:
            raise Forbidden("reviewer role required")

    def require_admin(self) -> None:
        if not self.is_admin:
            raise Forbidden("admin role required")


class SessionCache:
    """Short-lived positive cache so hot endpoints skip the DB lookup.

    Lives on the app (not the module) so every app instance / test gets its own.
    """

    def __init__(self, ttl_seconds: float, max_entries: int = 1024) -> None:
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._entries: dict[str, tuple[float, AuthContext]] = {}

    def get(self, token_hash: str) -> AuthContext | None:
        entry = self._entries.get(token_hash)
        if entry is None:
            return None
        stored_at, ctx = entry
        if time.monotonic() - stored_at > self.ttl:
            self._entries.pop(token_hash, None)
            return None
        return ctx

    def put(self, token_hash: str, ctx: AuthContext) -> None:
        if len(self._entries) >= self.max_entries:
            self._entries.clear()
        self._entries[token_hash] = (time.monotonic(), ctx)

    def drop(self, token_hash: str) -> None:
        self._entries.pop(token_hash, None)

    def clear(self) -> None:
        self._entries.clear()


class AuthService:
    def __init__(
        self,
        db: Database,
        identity: IdentityProvider,
        settings: Settings,
        cache: SessionCache | None = None,
    ) -> None:
        self.db = db
        self.identity = identity
        self.settings = settings
        self.cache = cache or SessionCache(settings.session_cache_seconds)

    # region login / logout
    def login(self, username: str, password: str, client_info: str = "") -> tuple[str, AuthContext, datetime]:
        """Verify through the identity provider, upsert the user, issue a session.

        ``LocalIdentity`` checks our own scrypt hash; ``OpenListIdentity`` proxies
        the external service and hands back the user's token (kept in the session
        so per-user file ACLs still work while OpenList is in use).
        """
        remote = self.identity.verify(username, password)
        if remote is None:
            raise Unauthorized("the server rejected these credentials")
        oplist_token = str(remote.get("token") or "")
        name = (remote.get("name") or username).strip()

        with self.db.session_scope() as session:
            user = self._upsert_user(session, remote, name)
            if not user.active:
                raise Unauthorized("this account is disabled")
            token = secrets.token_urlsafe(TOKEN_BYTES)
            expires_at = utcnow() + timedelta(days=self.settings.session_ttl_days)
            row = Session(
                token_hash=hash_token(token),
                user_id=user.id,
                oplist_token=oplist_token,
                client_info=client_info[:128],
                expires_at=expires_at,
            )
            session.add(row)
            session.flush()  # assigns row.id for the response payload
            user.last_login_at = utcnow()
            audit.record(session, action="login", user_id=user.id, target_type="user", target_id=user.id)
            ctx = self._context(user, row, expires_at)
        self.cache.put(hash_token(token), ctx)
        logger.info(f"login ok user={ctx.name!r} role={ctx.role}")
        return token, ctx, expires_at

    def resolve(self, token: str) -> AuthContext:
        """Session token → context (cached for ``session_cache_seconds``)."""
        if not token:
            raise Unauthorized("missing session token")
        token_hash = hash_token(token)
        cached = self.cache.get(token_hash)
        if cached is not None:
            return cached

        with self.db.session_scope() as session:
            row = session.scalar(select(Session).where(Session.token_hash == token_hash))
            if row is None or row.revoked_at is not None:
                raise Unauthorized("unknown or revoked session")
            if row.expires_at <= utcnow():
                raise Unauthorized("session expired")
            user = session.get(User, row.user_id)
            if user is None or not user.active:
                raise Unauthorized("account unavailable")
            ctx = self._context(user, row, row.expires_at)
        self.cache.put(token_hash, ctx)
        return ctx

    def logout(self, token: str) -> None:
        token_hash = hash_token(token)
        with self.db.session_scope() as session:
            row = session.scalar(select(Session).where(Session.token_hash == token_hash))
            if row is not None and row.revoked_at is None:
                row.revoked_at = utcnow()
                audit.record(
                    session, action="logout", user_id=row.user_id, target_type="session", target_id=row.id
                )
        self.cache.drop(token_hash)

    def revoke_all(self, user_id: int) -> int:
        """Revoke every session of a user (role change / disable)."""
        with self.db.session_scope() as session:
            rows = session.scalars(
                select(Session).where(Session.user_id == user_id, Session.revoked_at.is_(None))
            ).all()
            for row in rows:
                row.revoked_at = utcnow()
            return len(rows)

    # endregion

    # region users / roles
    def set_role(self, actor: AuthContext, user_id: int, role: str) -> User:
        """Admin-only role change; revokes the target's sessions."""
        actor.require_admin()
        if role not in ROLES:
            raise Forbidden(f"unknown role: {role}")
        with self.db.session_scope() as session:
            user = session.get(User, user_id)
            if user is None:
                raise Unauthorized(f"unknown user: {user_id}")
            user.role = role
            audit.record(
                session,
                action="set_role",
                user_id=actor.user_id,
                target_type="user",
                target_id=user.id,
                detail={"role": role},
            )
        self.revoke_all(user_id)
        self.cache.clear()
        return user

    def list_users(self) -> list[User]:
        with self.db.session_scope() as session:
            return list(session.scalars(select(User).order_by(User.name)).all())

    # endregion

    # region internals
    def _upsert_user(self, session, remote: dict, name: str) -> User:
        """Find or create the local row for the authenticated account.

        ``oplist_user_id`` holds ``"<provider>:<subject>"`` (the column name is kept
        for compatibility with existing databases).
        """
        subject = str(remote.get("id") or name.lower())
        oplist_user_id = subject if ":" in subject else f"{self.identity.kind}:{subject}"
        user = None
        if oplist_user_id:
            user = session.scalar(select(User).where(User.oplist_user_id == oplist_user_id))
        if user is None:
            user = session.scalar(select(User).where(func.lower(User.name) == name.lower()))
        if user is None:
            is_first = session.scalar(select(func.count()).select_from(User)) == 0
            role = ROLE_ADMIN if (is_first or self._is_bootstrap_admin(name)) else ROLE_ANNOTATOR
            user = User(
                oplist_user_id=oplist_user_id,
                name=name.lower(),
                email=remote.get("email") or "",
                role=role,
            )
            session.add(user)
            session.flush()
            logger.info(f"created user {user.name!r} with role {role}")
        else:
            user.oplist_user_id = oplist_user_id or user.oplist_user_id
            user.email = remote.get("email") or user.email
            if self._is_bootstrap_admin(user.name) and user.role != ROLE_ADMIN:
                user.role = ROLE_ADMIN
        return user

    def _is_bootstrap_admin(self, name: str) -> bool:
        configured = (self.settings.bootstrap_admin or "").strip().lower()
        return bool(configured) and name.strip().lower() == configured

    @staticmethod
    def _context(user: User, row: Session, expires_at: datetime) -> AuthContext:
        return AuthContext(
            session_id=row.id,
            user_id=user.id,
            name=user.name,
            role=user.role,
            oplist_token=row.oplist_token,
            expires_at=expires_at,
        )

    # endregion
