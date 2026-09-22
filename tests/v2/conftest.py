"""Fixtures for the v2 test suite (hermetic: in-memory DB, no network)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from tests.v2.fakes import FakeOpenList
from v2.adapters.openlist import OpenListAdapter
from v2.app import create_app
from v2.core.config import Settings
from v2.db.base import Database
from v2.services.container import Services


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_url="sqlite+pysqlite:///:memory:",
        upload_dir=str(tmp_path / "uploads"),
        oplist_host="http://openlist.test",  # never dialled: the fake client is injected
        oplist_token="service-token",  # background scan token
        inference_url="",
        inference_token="",
        # deterministic tests: scanners are exercised explicitly
        scan_on_startup=False,
        project_scan_interval=0,
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
def ol() -> FakeOpenList:
    """In-memory OpenList (see tests/v2/fakes.py)."""
    return FakeOpenList(service_token="service-token")


@pytest.fixture
def services(settings: Settings, db: Database, ol: FakeOpenList) -> Services:
    return Services.build(settings, db, openlist=OpenListAdapter(settings, client_factory=ol.client))


@pytest.fixture
def app(settings: Settings, db: Database, services: Services):
    return create_app(settings, db, services=services)


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> callable:
    """``login(client)`` → Authorization headers for a given account."""

    def _login(client: TestClient, username: str = "rainy", password: str = "secret") -> dict[str, str]:
        resp = client.post("/api/v2/auth/login", json={"username": username, "password": password})
        assert resp.status_code == 200, resp.text
        return {"Authorization": f"Bearer {resp.json()['token']}"}

    return _login
