"""The startup scan has to populate the task table without blocking startup."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient
from sqlalchemy import select

from tests.v2.fakes import LocalBackendHarness
from tests.v2.test_projects import seed
from v2.app import create_app
from v2.db.base import Database
from v2.db.models import Project, Task
from v2.services.container import Services


def test_startup_scan_populates_projects_and_tasks(settings, tmp_path):
    ol = LocalBackendHarness(service_token="service-token")
    seed(ol, "projA", files=("images/dish01/D1.png", "images/dish01/D2.png"))
    # a file-backed DB: the scan runs in a worker thread, and an in-memory
    # StaticPool connection cannot be shared with it
    patched = settings.model_copy(
        update={
            "scan_on_startup": True,
            "project_scan_interval": 0,
            "database_url": f"sqlite+pysqlite:///{(tmp_path / 'startup.db').as_posix()}",
        }
    )
    db = Database(patched.database_url)
    db.create_all()
    services = Services.build(patched, db, openlist=OpenListAdapter(patched, client_factory=ol.client))

    with TestClient(create_app(patched, db, services=services)) as client:
        assert client.get("/api/v2/health").status_code == 200  # startup did not block
        deadline = time.time() + 5
        while time.time() < deadline:
            with db.session_scope() as session:
                if session.scalar(select(Task)) is not None:
                    break
            time.sleep(0.05)
        with db.session_scope() as session:
            assert session.scalar(select(Project.name)) == "projA"
            assert {t.rel_path for t in session.scalars(select(Task))} == {
                "images/dish01/D1.png",
                "images/dish01/D2.png",
            }


def test_startup_survives_an_unreachable_openlist(settings, tmp_path):
    """A dead OpenList must not stop the API from starting."""
    ol = LocalBackendHarness(service_token="service-token")
    ol.fail_on("/zlabel_server/projects", RuntimeError("openlist is down"))
    patched = settings.model_copy(
        update={
            "scan_on_startup": True,
            "project_scan_interval": 0,
            "oplist_token": "",
            "database_url": f"sqlite+pysqlite:///{(tmp_path / 'down.db').as_posix()}",
        }
    )
    db = Database(patched.database_url)
    db.create_all()
    services = Services.build(patched, db, openlist=OpenListAdapter(patched, client_factory=ol.client))

    with TestClient(create_app(patched, db, services=services)) as client:
        time.sleep(0.1)
        assert client.get("/api/v2/health").status_code == 200
