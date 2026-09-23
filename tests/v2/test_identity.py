"""Local accounts: scrypt hashing, the provider contract, the bootstrap guard."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from v2.adapters.identity import (
    LocalIdentity,
    build_identity,
    hash_password,
    verify_password,
)
from v2.core.config import Settings
from v2.core.errors import ValidationFailed
from v2.db.models import ROLE_ADMIN, ROLE_REVIEWER, User


# region hashing
def test_password_hash_roundtrip_and_rejections():
    stored = hash_password("secret123")
    assert stored.startswith("scrypt$")
    assert verify_password("secret123", stored) is True
    assert verify_password("Secret123", stored) is False  # case matters
    assert verify_password("", stored) is False
    assert verify_password("secret123", "") is False
    assert verify_password("secret123", "garbage") is False
    # a fresh hash uses a fresh salt
    assert hash_password("secret123") != stored


def test_short_passwords_are_refused():
    with pytest.raises(ValidationFailed):
        hash_password("123")


# endregion


# region provider contract


def test_local_identity_creates_verifies_and_rejects(db):
    identity = LocalIdentity(db, Settings())
    identity.create_user("Alice", "secret123", role=ROLE_REVIEWER)
    identity.create_user("root", "secret123", admin=True)

    assert identity.verify("alice", "secret123")["name"] == "alice"  # names are lower-cased
    assert identity.verify("ALICE", "secret123") is not None
    assert identity.verify("alice", "wrong") is None
    assert identity.verify("nobody", "secret123") is None

    with db.session_scope() as session:
        alice = session.scalar(select(User).where(User.name == "alice"))
        root = session.scalar(select(User).where(User.name == "root"))
        assert alice.role == ROLE_REVIEWER and root.role == ROLE_ADMIN
        assert alice.password_hash.startswith("scrypt$")
        assert alice.active is True

    # a disabled account cannot log in
    with db.session_scope() as session:
        session.scalar(select(User).where(User.name == "alice")).active = False
    assert identity.verify("alice", "secret123") is None


def test_password_change_and_bootstrap_mode(db):
    identity = LocalIdentity(db, Settings())
    identity.create_user("alice", "secret123")
    assert identity.set_password("alice", "newsecret") is True
    assert identity.verify("alice", "newsecret") is not None and identity.verify("alice", "secret123") is None

    # bootstrap must never reset an existing password
    identity.create_user("alice", "another-one", admin=True, only_if_missing=True)
    assert identity.verify("alice", "newsecret") is not None
    assert identity.set_password("nobody", "whatever") is False


def test_build_identity_returns_the_local_provider(db):
    provider = build_identity(Settings(), db=db)
    assert isinstance(provider, LocalIdentity) and provider.kind == "local"


# endregion
