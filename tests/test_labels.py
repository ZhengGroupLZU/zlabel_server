"""Label linking + labels/progress endpoints (batch B4)."""

from __future__ import annotations

import hashlib
import importlib

import pytest
from fastapi.testclient import TestClient


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _img(project: str, rel: str) -> str:
    """The scanner hands out absolute OpenList paths."""
    from app.config import SETTINGS

    return f"{SETTINGS.oplist_proj_dir}/{project}/{rel}"


@pytest.fixture(scope="module")
def appmod():
    from app import config

    old_name = config.SETTINGS.model_name
    config.SETTINGS.model_name = "EdgeSAM"
    try:
        yield importlib.import_module("app.app")
    finally:
        config.SETTINGS.model_name = old_name


@pytest.fixture
def client(appmod, fake_predictor):
    appmod.SAM_MODEL = fake_predictor
    return TestClient(appmod.app)


@pytest.fixture
def mem_db(monkeypatch):
    """Isolated in-memory database (never touches ./zlabel_server.db)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app import db

    # one shared in-memory database: TestClient serves requests from a worker
    # thread, so every connection must see the same schema/data
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    db.Base.metadata.create_all(engine)
    monkeypatch.setattr(db, "session_maker", sessionmaker(engine, expire_on_commit=False))
    return db


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def test_extract_label_names(appmod):
    anno = {
        "id": "x",
        "results": {
            "r1": {"labels": [{"name": "Root"}, {"name": "Shoot"}]},
            "r2": {"labels": [{"name": "Root"}]},  # duplicate name
            "r3": {},  # no labels
        },
    }
    assert appmod.extract_label_names(anno) == ["Root", "Shoot"]


def test_extract_label_names_handles_empty(appmod):
    assert appmod.extract_label_names({}) == []
    assert appmod.extract_label_names({"results": {}}) == []
    assert appmod.extract_label_names({"results": {"r": {"labels": None}}}) == []


# --------------------------------------------------------------------------- #
# db layer
# --------------------------------------------------------------------------- #
def test_insert_link_table_links_labels_and_finishes(mem_db):
    mem_db.create_or_update_projects([{"name": "projA", "files": [_img("projA", "dish/D1.png")]}])
    anno_id = _md5("projA/dish/D1.png")

    mem_db.insert_link_table(anno_id, label_names=["Root", "Shoot"], user_name="Rainy")

    assert [lbl["name"] for lbl in mem_db.get_labels("projA")] == ["Root", "Shoot"]
    assert mem_db.get_progress("projA") == {"finished": 1, "total": 1}

    with mem_db.session_maker() as session:
        user = session.query(mem_db.User).one()
        assert user.name == "rainy"  # normalised, no duplicate "Rainy"/"rainy"
        assert user.finished_count == 1
        assert session.query(mem_db.Task).one().finished is True


def test_insert_link_table_is_idempotent(mem_db):
    mem_db.create_or_update_projects([{"name": "projA", "files": [_img("projA", "a.png")]}])
    anno_id = _md5("projA/a.png")

    mem_db.insert_link_table(anno_id, ["Root"], "rainy")
    mem_db.insert_link_table(anno_id, ["Root"], "rainy")  # re-annotated frame

    with mem_db.session_maker() as session:
        assert session.query(mem_db.Label).count() == 1
        assert len(session.execute(mem_db.link_task_label.select()).all()) == 1
        assert session.query(mem_db.User).one().finished_count == 1  # not double-counted


def test_labels_and_progress_are_scoped_per_project(mem_db):
    mem_db.create_or_update_projects([
        {"name": "projA", "files": [_img("projA", "a.png")]},
        {"name": "projB", "files": [_img("projB", "b.png")]},
    ])
    mem_db.insert_link_table(_md5("projA/a.png"), ["Root"], "rainy")

    assert [lbl["name"] for lbl in mem_db.get_labels("projA")] == ["Root"]
    assert mem_db.get_labels("projB") == []
    assert mem_db.get_progress("projA") == {"finished": 1, "total": 1}
    assert mem_db.get_progress("projB") == {"finished": 0, "total": 1}
    assert mem_db.get_progress("") == {"finished": 1, "total": 2}
    assert mem_db.get_labels("") == [{"id": mem_db.get_labels("projA")[0]["id"], "name": "Root", "color": "#000000"}]


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #
def test_labels_endpoint(client, mem_db):
    mem_db.create_or_update_projects([{"name": "projA", "files": [_img("projA", "a.png")]}])
    mem_db.insert_link_table(_md5("projA/a.png"), ["Root", "Shoot"], "rainy")

    r = client.get("/api/v1/labels", params={"project": "projA"})

    assert r.status_code == 200
    assert [lbl["name"] for lbl in r.json()["data"]] == ["Root", "Shoot"]


def test_progress_endpoint(client, mem_db):
    mem_db.create_or_update_projects([{"name": "projA", "files": [_img("projA", "a.png"), _img("projA", "b.png")]}])
    mem_db.insert_link_table(_md5("projA/a.png"), ["Root"], "rainy")

    r = client.get("/api/v1/how-many-finished", params={"project": "projA"})

    assert r.status_code == 200
    assert r.json()["data"] == {"finished": 1, "total": 2}
