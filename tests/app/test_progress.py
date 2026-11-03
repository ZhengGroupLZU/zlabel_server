"""Per-user workload aggregation (`app/services/progress_service.py`).

Drives a realistic history through the API, because every column is a statement
about *history*: claims (released and re-claimed), authored versions (first vs
later), submits (a rejection makes the second one count twice) and review events.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from app.db.models import AuditLog, Task, User
from tests.app.test_projects import seed

PROJ = "projA"


@pytest.fixture
def workspace(client, auth_headers, harness, services, db):
    """projA with three tasks, two annotators, a reviewer and an idle member."""
    seed(harness, PROJ, files=("a.png", "b.png", "c.png"))
    harness.users["bob"] = "bobsecret"
    harness.users["carol"] = "carolsecret"
    harness.users["dave"] = "davesecret"
    harness.users["rev"] = "revsecret"
    admin = auth_headers(client, "rainy")
    # the session token of a login carries the role, so promote and log in again
    auth_headers(client, "rev", "revsecret")
    with db.session_scope() as session:
        session.scalar(select(User).where(User.name == "rev")).role = "reviewer"
    client.post("/api/v2/projects/scan", headers=admin)
    users = {user.name: user for user in services.auth.list_users()}
    for name in ("bob", "carol", "rev"):
        resp = client.post(
            f"/api/v2/projects/{PROJ}/members",
            json={"user_id": users[name].id, "role": "reviewer" if name == "rev" else "annotator"},
            headers=admin,
        )
        assert resp.status_code == 201, resp.text
    idle = client.post(
        f"/api/v2/projects/{PROJ}/members",
        json={"user_id": users["dave"].id, "role": "annotator"},
        headers=admin,
    )
    assert idle.status_code == 201, idle.text
    return {
        "admin": admin,
        "bob": auth_headers(client, "bob", "bobsecret"),
        "carol": auth_headers(client, "carol", "carolsecret"),
        "rev": auth_headers(client, "rev", "revsecret"),
        "users": users,
    }


def _task(client, headers, rel: str, project: str = PROJ) -> str:
    items = client.get(f"/api/v2/projects/{project}/tasks", headers=headers).json()["items"]
    return next(item["anno_id"] for item in items if item["rel_path"] == rel)


def _workload(client, headers, project: str = PROJ) -> dict[str, dict]:
    resp = client.get(f"/api/v2/projects/{project}/progress", params={"by_user": True}, headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["by_user"]


def _save(client, headers, anno_id: str, project: str = PROJ, base_version: int | None = None) -> None:
    """Store a document (each save is a version, so a later one needs its base)."""
    resp = client.put(
        f"/api/v2/projects/{project}/annotations/{anno_id}",
        params={"base_version": base_version} if base_version is not None else None,
        json={"results": {"r1": {"labels": [{"name": "Seed"}]}}},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text


def _claim(client, headers, anno_id: str) -> None:
    resp = client.post(f"/api/v2/tasks/{anno_id}/claim", headers=headers)
    assert resp.status_code == 200, resp.text


def _submit(client, headers, anno_id: str) -> None:
    resp = client.post(f"/api/v2/tasks/{anno_id}/submit", headers=headers)
    assert resp.status_code == 200, resp.text


def _review(client, headers, anno_id: str, decision: str) -> None:
    body = {"decision": decision, "note": "fix it" if decision == "reject" else ""}
    resp = client.post(f"/api/v2/tasks/{anno_id}/review", json=body, headers=headers)
    assert resp.status_code == 200, resp.text


def test_every_column_counts_its_own_history(client, workspace):
    """bob works one task through reject -> rework -> approve; carol works another."""
    bob = workspace["bob"]
    rev = workspace["rev"]
    a = _task(client, bob, "a.png")
    _claim(client, bob, a)
    _save(client, bob, a)  # version 1 -> created
    _submit(client, bob, a)
    _review(client, rev, a, "reject")
    _claim(client, bob, a)  # re-claim after the rejection: still one task
    _save(client, bob, a, base_version=1)  # version 2 -> modified
    _submit(client, bob, a)  # second submit counts again
    _review(client, rev, a, "approve")

    carol = workspace["carol"]
    b = _task(client, carol, "b.png")
    _claim(client, carol, b)
    _save(client, carol, b)
    client.post(f"/api/v2/tasks/{b}/release", headers=carol)  # released, not submitted
    _claim(client, carol, b)  # claimed twice, still one task
    _save(client, carol, b, base_version=1)
    _submit(client, carol, b)
    _review(client, rev, b, "approve")

    by_user = _workload(client, workspace["admin"])
    assert by_user["bob"] == {
        "role": "annotator",
        "claimed": 1,
        "created": 1,
        "modified": 1,
        "submitted": 2,  # rejected then resubmitted
        "approved": 1,
        "rejected": 1,
        "total": 1,  # one task, touched four ways
    }
    assert by_user["carol"] == {
        "role": "annotator",
        "claimed": 1,  # claim + release + claim is one task
        "created": 1,
        "modified": 1,
        "submitted": 1,
        "approved": 1,
        "rejected": 0,
        "total": 1,
    }
    # the reviewer did the reviewing; the columns are the annotator's output
    assert by_user["rev"] == {
        "role": "reviewer",
        "claimed": 0,
        "created": 0,
        "modified": 0,
        "submitted": 0,
        "approved": 0,
        "rejected": 0,
        "total": 0,
    }
    # an idle member still shows up, with zeros and the project role
    assert by_user["dave"]["total"] == 0 and by_user["dave"]["role"] == "annotator"


def test_total_is_the_union_of_touched_tasks(client, workspace):
    bob = workspace["bob"]
    for rel in ("a.png", "b.png"):
        anno_id = _task(client, bob, rel)
        _claim(client, bob, anno_id)
        _save(client, bob, anno_id)
    only_saved = _task(client, bob, "c.png")  # written without claiming first
    _save(client, bob, only_saved)

    row = _workload(client, workspace["admin"])["bob"]
    assert row["claimed"] == 2
    assert row["created"] == 3
    assert row["total"] == 3  # the union, not the sum
    assert row["submitted"] == 0


def test_review_attribution_survives_a_rename(client, workspace, db):
    """``annotator_id`` in the audit detail wins over the historic name."""
    bob = workspace["bob"]
    rev = workspace["rev"]
    anno_id = _task(client, bob, "a.png")
    _claim(client, bob, anno_id)
    _save(client, bob, anno_id)
    _submit(client, bob, anno_id)
    _review(client, rev, anno_id, "approve")

    with db.session_scope() as session:
        user = session.scalar(select(User).where(User.name == "bob"))
        user.name = "bobby"
        session.flush()

    by_user = _workload(client, workspace["admin"])
    assert by_user["bobby"]["approved"] == 1
    assert by_user["bobby"]["created"] == 1
    assert "bob" not in by_user


def test_old_review_rows_without_an_id_fall_back_to_the_name(client, workspace, db):
    """Rows written before ``annotator_id`` existed still land on the right person."""
    bob = _task(client, workspace["bob"], "a.png")
    with db.session_scope() as session:
        task = session.scalar(select(Task).where(Task.anno_id == bob))
        session.add(
            AuditLog(
                action="review_approve",
                target_type="task",
                target_id=bob,
                detail_json=json.dumps({"note": "", "annotator": "bob"}),
            )
        )
        task.state = "approved"

    assert _workload(client, workspace["admin"])["bob"]["approved"] == 1


def test_the_breakdown_is_project_scoped(client, workspace, harness):
    """Another project's task is not part of this project's numbers."""
    seed(harness, "projB", files=("x.png",))
    admin = workspace["admin"]
    client.post("/api/v2/projects/scan", headers=admin)
    other = _task(client, admin, "x.png", project="projB")
    _claim(client, workspace["bob"], other)  # open access mode: anyone may claim

    assert _workload(client, admin)["bob"]["claimed"] == 0
    assert _workload(client, admin, project="projB")["bob"]["claimed"] == 1
