"""Schema behaviour: defaults, constraints and the shared anno_id contract."""

from __future__ import annotations

import hashlib

import pytest
from sqlalchemy.exc import IntegrityError

from v2.contracts.ids import anno_id_for
from v2.db import models as m


def test_anno_id_matches_the_desktop_formula():
    # the desktop computes md5("<project>/<relative posix path>")
    assert anno_id_for("projA", "images/dish/D1.png") == hashlib.md5(b"projA/images/dish/D1.png").hexdigest()


def test_anno_id_normalises_windows_separators():
    assert anno_id_for("projA", r"images\dish\D1.png") == anno_id_for("projA", "images/dish/D1.png")


def _project(db, name: str = "projA") -> m.Project:
    with db.session_scope() as session:
        project = m.Project(name=name)
        session.add(project)
        session.flush()
        return project


def test_task_defaults_and_relations(db):
    project = _project(db)
    with db.session_scope() as session:
        task = m.Task(
            project_id=project.id,
            anno_id=anno_id_for("projA", "images/a/D1.png"),
            path="/zlabel_server/projects/projA/images/a/D1.png",
            rel_path="images/a/D1.png",
            group_name="images/a",
            day=1,
        )
        session.add(task)
    with db.session_scope() as session:
        stored = session.query(m.Task).one()
        assert stored.state == m.STATE_DRAFT
        assert stored.missing is False
        assert stored.claimed_by is None
        assert stored.lease_expires_at is None
        assert stored.group_name == "images/a" and stored.day == 1
        assert stored.project.name == "projA"


def test_anno_id_is_unique(db):
    project = _project(db)
    with db.session_scope() as session:
        session.add(m.Task(project_id=project.id, anno_id="dup", path="p", rel_path="r"))
    with pytest.raises(IntegrityError):
        with db.session_scope() as session:
            session.add(m.Task(project_id=project.id, anno_id="dup", path="p2", rel_path="r2"))


def test_label_unique_per_project_and_task_link(db):
    project = _project(db)
    other = _project(db, "projB")
    with db.session_scope() as session:
        task = m.Task(project_id=project.id, anno_id="a1", path="p", rel_path="r")
        label = m.Label(project_id=project.id, name="Root", color="#112233")
        session.add_all([task, label])
        session.flush()
        task.labels.append(label)
    with db.session_scope() as session:
        assert [lbl.name for lbl in session.query(m.Task).one().labels] == ["Root"]
        # the same name in another project is a different label
        session.add(m.Label(project_id=other.id, name="Root"))
    with pytest.raises(IntegrityError):
        with db.session_scope() as session:
            session.add(m.Label(project_id=project.id, name="Root"))


def test_claim_fields_persist(db):
    project = _project(db)
    with db.session_scope() as session:
        user = m.User(oplist_user_id="7", name="rainy", role=m.ROLE_REVIEWER)
        session.add(user)
        session.flush()
        task = m.Task(project_id=project.id, anno_id="a2", path="p", rel_path="r")
        task.claimer = user
        task.claimed_at = m.utcnow()
        task.lease_expires_at = m.utcnow()
        session.add(task)
    with db.session_scope() as session:
        stored = session.query(m.Task).one()
        assert stored.claimer.name == "rainy"
        assert stored.claimer.is_reviewer is True
        assert stored.claimer.is_admin is False
