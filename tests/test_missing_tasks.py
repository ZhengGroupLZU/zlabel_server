"""Soft-delete: files that vanish from OpenList stop being offered (batch B6)."""

from __future__ import annotations

import hashlib

import pytest


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _img(project: str, rel: str) -> str:
    """The scanner hands out absolute OpenList paths."""
    from app.config import SETTINGS

    return f"{SETTINGS.oplist_proj_dir}/{project}/{rel}"


@pytest.fixture
def mem_db(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app import db

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    db.Base.metadata.create_all(engine)
    monkeypatch.setattr(db, "session_maker", sessionmaker(engine, expire_on_commit=False))
    return db


def _by_filename(db, project: str) -> dict[str, bool]:
    with db.session_maker() as session:
        tasks = (
            session.query(db.Task)
            .join(db.Project)
            .where(db.Project.name == project)
            .all()
        )
        return {t.filename.rsplit("/", 1)[-1]: bool(t.missing) for t in tasks}


def test_vanished_files_are_marked_missing(mem_db):
    mem_db.create_or_update_projects([{"name": "projA", "files": [_img("projA", "a.png"), _img("projA", "b.png")]}])
    assert _by_filename(mem_db, "projA") == {"a.png": False, "b.png": False}

    # b.png was deleted from the dataset
    mem_db.create_or_update_projects([{"name": "projA", "files": [_img("projA", "a.png")]}])

    assert _by_filename(mem_db, "projA") == {"a.png": False, "b.png": True}


def test_returning_files_clear_the_missing_flag(mem_db):
    mem_db.create_or_update_projects([{"name": "projA", "files": [_img("projA", "a.png")]}])
    mem_db.create_or_update_projects([{"name": "projA", "files": []}])
    assert _by_filename(mem_db, "projA") == {"a.png": True}

    mem_db.create_or_update_projects([{"name": "projA", "files": [_img("projA", "a.png")]}])
    assert _by_filename(mem_db, "projA") == {"a.png": False}


def test_missing_tasks_are_not_offered(mem_db):
    mem_db.create_or_update_projects([{"name": "projA", "files": [_img("projA", "a.png"), _img("projA", "b.png")]}])
    mem_db.create_or_update_projects([{"name": "projA", "files": [_img("projA", "a.png")]}])

    tasks = mem_db.get_tasks(-1, 50, -1, False)
    assert [t.filename.rsplit("/", 1)[-1] for t in tasks] == ["a.png"]
    assert mem_db.get_progress("projA") == {"finished": 0, "total": 1}


def test_labels_ignore_missing_tasks(mem_db):
    mem_db.create_or_update_projects([{"name": "projA", "files": [_img("projA", "a.png"), _img("projA", "b.png")]}])
    mem_db.insert_link_table(_md5("projA/a.png"), ["Root"], "rainy")
    mem_db.insert_link_table(_md5("projA/b.png"), ["Shoot"], "rainy")
    assert [lbl["name"] for lbl in mem_db.get_labels("projA")] == ["Root", "Shoot"]

    mem_db.create_or_update_projects([{"name": "projA", "files": [_img("projA", "a.png")]}])  # b.png gone

    assert [lbl["name"] for lbl in mem_db.get_labels("projA")] == ["Root"]


def test_annotations_of_missing_tasks_still_link(mem_db):
    """A frame may be finished right before the file disappears: keep history."""
    mem_db.create_or_update_projects([{"name": "projA", "files": [_img("projA", "a.png")]}])
    mem_db.create_or_update_projects([{"name": "projA", "files": []}])

    mem_db.insert_link_table(_md5("projA/a.png"), ["Root"], "rainy")

    assert mem_db.get_progress("projA") == {"finished": 0, "total": 0}  # not offered
    with mem_db.session_maker() as session:
        assert session.query(mem_db.Task).one().finished is True  # history kept
