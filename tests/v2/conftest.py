"""Fixtures for the v2 test suite (hermetic: in-memory DB, no network)."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from tests.v2.fakes import LocalBackendHarness
from v2.adapters.local_disk import LocalDiskBackend
from v2.app import create_app
from v2.core.config import Settings
from v2.db.base import Database
from v2.services.container import Services


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
