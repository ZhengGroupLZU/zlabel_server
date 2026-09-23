"""Schema behaviour: defaults, constraints and the shared anno_id contract."""

from __future__ import annotations

import hashlib

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from v2.contracts.ids import anno_id_for
from v2.db import models as m


def test_anno_id_matches_the_desktop_formula():
    # both sides compute sha256("<project key>/<relative posix path>")
    assert (
        anno_id_for("projKey", "images/dish/D1.png")
        == hashlib.sha256(b"projKey/images/dish/D1.png").hexdigest()
    )


def test_anno_id_normalises_windows_separators():
    assert anno_id_for("projKey", r"images\dish\D1.png") == anno_id_for("projKey", "images/dish/D1.png")


def test_legacy_anno_id_is_the_pre_key_formula():
    from v2.contracts.ids import legacy_anno_id_for

    assert (
        legacy_anno_id_for("projA", "images/dish/D1.png")
        == hashlib.md5(b"projA/images/dish/D1.png").hexdigest()
    )


def _project(db, name: str = "projA") -> m.Project:
    with db.session_scope() as session:
        project = m.Project(name=name, key=f"key-{name}")
        session.add(project)
        session.flush()
        return project


def test_task_defaults_and_relations(db):
    project = _project(db)
    with db.session_scope() as session:
        task = m.Task(
            project_id=project.id,
            anno_id=anno_id_for("key-projA", "images/a/D1.png"),
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


def test_sqlite_foreign_keys_are_enforced(db):
    """``Database`` turns the SQLite ``foreign_keys`` pragma on.

    Without it every ``ondelete="CASCADE"/"SET NULL"`` in the models is a no-op
    and a delete leaves orphan rows behind.
    """
    with db.session_scope() as session:
        project = m.Project(name="projA", key="k1")
        session.add(project)
        session.flush()
        task = m.Task(project_id=project.id, anno_id="a" * 64, path="p", rel_path="r")
        session.add(task)
        session.flush()
        project_id, task_id = project.id, task.id

    # a dangling foreign key is refused instead of silently stored
    with pytest.raises(IntegrityError):
        with db.session_scope() as session:
            session.add(m.Task(project_id=999_999, anno_id="b" * 64, path="p", rel_path="r2"))

    # deleting the parent cascades to the child
    with db.session_scope() as session:
        session.delete(session.get(m.Project, project_id))
    with db.session_scope() as session:
        assert session.get(m.Task, task_id) is None


def test_deleting_a_label_removes_its_task_links(db):
    """The task↔label link rows are cascaded by the database (previously orphaned)."""
    with db.session_scope() as session:
        project = m.Project(name="projA", key="k1")
        session.add(project)
        session.flush()
        task = m.Task(project_id=project.id, anno_id="a" * 64, path="p", rel_path="r")
        label = m.Label(project_id=project.id, name="Root")
        session.add_all([task, label])
        session.flush()
        task.labels.append(label)
        label_id = label.id
    with db.session_scope() as session:
        session.delete(session.get(m.Label, label_id))
    with db.session_scope() as session:
        assert session.execute(select(m.link_task_label)).all() == []


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
        user = m.User(identity_id="7", name="rainy", role=m.ROLE_REVIEWER)
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
