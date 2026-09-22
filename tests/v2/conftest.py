"""Fixtures for the v2 test suite (hermetic: in-memory DB, no network)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from v2.app import create_app
from v2.core.config import Settings
from v2.db.base import Database


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_url="sqlite+pysqlite:///:memory:",
        upload_dir=str(tmp_path / "uploads"),
        oplist_host="",  # probes stay offline
        inference_url="",
        inference_token="",
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
def app(settings: Settings, db: Database):
    return create_app(settings, db)


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
