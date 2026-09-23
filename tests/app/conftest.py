"""Fixtures for the app test suite (hermetic: in-memory DB, no network)."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.adapters.local_disk import LocalDiskBackend
from app.app import create_app
from app.core.config import Settings
from app.db.base import Database
from app.services.container import Services
from tests.app.fakes import LocalBackendHarness


@pytest.fixture(autouse=True)
def _cheap_password_hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lower the scrypt cost for the tests only.

    The production cost is deliberately expensive (~160 ms per hash/verify here,
    and ``auth_headers`` creates *and* logs in an account per user): it dominated
    the suite while asserting nothing about the KDF. The stored hash carries its
    own parameters (``scrypt$n$r$p$salt$hash``), so ``verify_password`` reads ``n``
    back from the value and the real cost is untouched outside the tests:
    ``tests/app/test_identity.py`` still exercises the real code path.
    """
    from app.adapters import identity

    monkeypatch.setattr(identity, "SCRYPT_N", 2**10)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_url="sqlite+pysqlite:///:memory:",
        upload_dir=str(tmp_path / "uploads"),
        storage_root=str(tmp_path / "storage"),
        anno_dir=".zlabel/annos",
        inference_token="internal-secret",  # shared secret (API <-> worker)
        inference_url="",
    )


@pytest.fixture
def db(settings: Settings) -> Iterator[Database]:
    database = Database(settings.database_url)
    database.create_all()
    try:
        yield database
    finally:
        database.dispose()


@pytest.fixture
def harness(settings: Settings) -> LocalBackendHarness:
    """Seeds the storage tree the backend under test actually reads.

    The harness keeps the old in-memory fake's ``add_file``/``add_dir``/``users``
    surface, but every write lands in ``settings.storage_root``.
    """
    return LocalBackendHarness(settings.storage_root)


@pytest.fixture
def services(settings: Settings, db: Database, harness: LocalBackendHarness) -> Services:
    built = Services.build(settings, db, storage=LocalDiskBackend(settings))
    harness.identity = built.auth.identity
    built.auth.identity.create_user("rainy", "secret", admin=True)
    return built


@pytest.fixture
def app(settings: Settings, db: Database, services: Services):
    return create_app(settings, db, services=services)


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def local_settings(settings: Settings) -> Settings:
    """There is no other backend any more: kept so the older tests keep working."""
    return settings


@pytest.fixture
def local_client(settings: Settings, db: Database, services: Services) -> Iterator[TestClient]:
    """Alias of :func:`client`: there is only one backend now."""
    with TestClient(create_app(settings, db, services=services)) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(harness: LocalBackendHarness) -> callable:
    """``login(client, name, password)`` → Authorization headers.

    Accounts are created on first use (``harness.users[name] = pw`` wins, else the shared
    test password), which is what the old fake did implicitly. ``rainy`` is the
    bootstrap admin; everybody else is an annotator.
    """
    created: set[str] = set()

    def _login(client: TestClient, username: str = "rainy", password: str = "secret") -> dict[str, str]:
        services_: Services = client.app.state.services
        secret = password or harness.users.get(username) or "secret"
        # create_user enforces a minimum length; tests happily pass short ones.
        # "rainy" is the bootstrap admin (created by the services fixture), so its
        # password is used verbatim.
        stored = secret if username == "rainy" or len(secret) >= 8 else secret + "-padding"
        if username not in created:
            if username != "rainy":  # created (as admin) by the services fixture already
                with contextlib.suppress(Exception):
                    services_.auth.identity.create_user(username, stored)
            created.add(username)
        resp = client.post("/api/v2/auth/login", json={"username": username, "password": stored})
        assert resp.status_code == 200, resp.text
        return {"Authorization": f"Bearer {resp.json()['token']}"}

    return _login
