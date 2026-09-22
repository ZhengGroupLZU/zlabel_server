"""anno_id / filename contract shared with the ZLabel desktop client.

The desktop computes ``anno_id_for(project_name, relpath)`` =
``md5(f"{project_name}/{relpath}")`` and writes ``<anno_id>.zlabel`` locally.
The server must produce exactly the same value from its task table, or the
per-project zlabel files would not line up between the two modes.
"""

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
    """Isolated in-memory database (never touches the configured SQLite file)."""
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


def _tasks(db):
    with db.session_maker() as session:
        return session.query(db.Task).order_by(db.Task.anno_id).all()


def test_anno_id_matches_desktop_formula(mem_db):
    mem_db.create_or_update_projects([
        {
            "name": "seed_germ_high_res",
            "files": [
                _img("seed_germ_high_res", "zihuamuxu/2020-z-1004-1/D1.png"),
                _img("seed_germ_high_res", "dish2/D10.png"),
            ],
        }
    ])

    tasks = _tasks(mem_db)
    assert len(tasks) == 2
    for task in tasks:
        rel = task.filename.split("/seed_germ_high_res/", 1)[1]
        assert task.anno_id == _md5(f"seed_germ_high_res/{rel}")


def test_anno_id_uses_posix_separators(mem_db):
    """Windows ``Path.relative_to`` yields backslashes: never let them leak."""
    mem_db.create_or_update_projects([
        {"name": "projA", "files": [_img("projA", "dish/D1.png")]},
    ])

    task = _tasks(mem_db)[0]
    assert task.anno_id == _md5("projA/dish/D1.png")
    assert "\\" not in task.anno_id  # md5 strings cannot contain them anyway


def test_anno_id_is_independent_of_the_deployment_path(mem_db, monkeypatch):
    """Moving OpenList (or changing oplist_proj_dir) must not change ids."""
    from app.config import SETTINGS

    monkeypatch.setattr(SETTINGS, "oplist_proj_dir", "/somewhere/else/projects")
    mem_db.create_or_update_projects([{"name": "p", "files": [_img("p", "x/y.png")]}])
    assert _tasks(mem_db)[0].anno_id == _md5("p/x/y.png")


def test_upsert_is_idempotent(mem_db):
    files = [_img("p", "a.png"), _img("p", "dish/D1.png")]
    mem_db.create_or_update_projects([{"name": "p", "files": files}])
    first = {t.anno_id for t in _tasks(mem_db)}

    mem_db.create_or_update_projects([{"name": "p", "files": files}])

    tasks = _tasks(mem_db)
    assert len(tasks) == 2  # no duplicates
    assert {t.anno_id for t in tasks} == first
