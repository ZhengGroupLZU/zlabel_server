"""Who the users are, independent of where the bytes live.

``LocalIdentity`` stores the accounts itself: scrypt hashes in our ``users`` table
(stdlib only, no extra dependency), so a deployment needs no external service.

The ``IdentityProvider`` Protocol is the seam: a future SSO/LDAP provider only has
to answer "are these credentials valid, and who is it" — the session token is ours
in any case.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from typing import Any, Protocol, runtime_checkable

from v2.core.config import Settings
from v2.core.errors import ValidationFailed
from v2.core.logging import get_logger

logger = get_logger("zlabel.v2.identity")

# scrypt parameters (RFC 7914): ~16 MB, a few dozen ms on a laptop CPU
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
SALT_BYTES = 16
MIN_PASSWORD_LENGTH = 6


def hash_password(password: str) -> str:
    """``scrypt$n$r$p$salt$hash`` (base64). Self-describing for future upgrades."""
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise ValidationFailed(f"a password needs at least {MIN_PASSWORD_LENGTH} characters")
    salt = secrets.token_bytes(SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=SCRYPT_DKLEN
    )
    return "$".join(
        (
            "scrypt",
            str(SCRYPT_N),
            str(SCRYPT_R),
            str(SCRYPT_P),
            base64.b64encode(salt).decode(),
            base64.b64encode(digest).decode(),
        )
    )


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check of a stored hash (never raises for bad input)."""
    if not stored or not password:
        return False
    try:
        scheme, n, r, p, salt_b64, hash_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=base64.b64decode(salt_b64),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(base64.b64decode(hash_b64)),
        )
        return hmac.compare_digest(digest, base64.b64decode(hash_b64))
    except Exception as e:  # noqa: BLE001 - a broken hash is "not this password"
        logger.warning(f"password verification failed: {e}")
        return False


@runtime_checkable
class IdentityProvider(Protocol):
    kind: str

    def verify(self, username: str, password: str) -> dict[str, Any] | None:
        """``{"id","name","email"}`` when the credentials are valid, else None."""

    def set_password(self, username: str, password: str) -> bool:
        """Set/replace a password (used by the admin CLI/API)."""


class LocalIdentity:
    """Accounts stored in our own ``users`` table (no external service)."""

    kind = "local"

    def __init__(self, db, settings: Settings) -> None:
        self.db = db
        self.settings = settings

    def verify(self, username: str, password: str) -> dict[str, Any] | None:
        """Return the account for valid credentials; ``None`` otherwise."""
        from sqlalchemy import func, select

        from v2.db.models import User

        name = (username or "").strip().lower()
        if not name or not password:
            return None
        with self.db.session_scope() as session:
            user = session.scalar(select(User).where(func.lower(User.name) == name))
            if user is None or not user.active or not user.password_hash:
                return None
            if not verify_password(password, user.password_hash):
                return None
            return {"id": f"local:{user.id}", "name": user.name, "email": user.email or ""}

    def set_password(self, username: str, password: str) -> bool:
        """Set a password for an existing account (admin CLI/API)."""
        from sqlalchemy import func, select

        from v2.db.models import User

        hashed = hash_password(password)
        with self.db.session_scope() as session:
            user = session.scalar(select(User).where(func.lower(User.name) == username.strip().lower()))
            if user is None:
                return False
            user.password_hash = hashed
            logger.info(f"password updated for {user.name!r}")
            return True

    def create_user(
        self,
        username: str,
        password: str,
        *,
        role: str = "annotator",
        email: str = "",
        admin: bool = False,
        only_if_missing: bool = False,
    ):
        """Create or update a local account.

        ``only_if_missing`` is the bootstrap mode: an existing account keeps its
        password (the env var cannot silently reset it on every restart).
        """
        from sqlalchemy import func, select

        from v2.db.models import ROLE_ADMIN, ROLE_ANNOTATOR, ROLES, User

        name = (username or "").strip().lower()
        if not name:
            raise ValidationFailed("a user needs a name")
        if role not in ROLES:
            raise ValidationFailed(f"unknown role: {role}")
        hashed = hash_password(password)
        with self.db.session_scope() as session:
            user = session.scalar(select(User).where(func.lower(User.name) == name))
            if user is not None and only_if_missing:
                return {"id": user.id, "name": user.name, "role": user.role}
            if user is None:
                user = User(
                    identity_id=f"local:{name}",
                    name=name,
                    email=email,
                    role=ROLE_ADMIN if admin else role,
                    password_hash=hashed,
                )
                session.add(user)
                session.flush()
                logger.info(f"created local account {name!r} (role={user.role})")
            else:
                user.password_hash = hashed
                user.active = True
                if admin:
                    user.role = ROLE_ADMIN
                elif role:
                    user.role = role or ROLE_ANNOTATOR
            return {"id": user.id, "name": user.name, "role": user.role}


def build_identity(settings: Settings, *, db) -> IdentityProvider:
    """Build the identity provider (one implementation today, kept injectable)."""
    return LocalIdentity(db, settings)
