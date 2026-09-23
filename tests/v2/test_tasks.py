"""Claim/lease + submit/review workflow (the multi-user core)."""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select

from tests.v2.test_projects import seed
from v2.db.models import Annotation, AuditLog, Task, utcnow

ROOT = "/zlabel_server/projects"


def bootstrap(client, auth_headers, harness, files=("a.png", "b.png")) -> dict:
    """One project, synced; returns {"admin": headers, "bob": headers}."""
    seed(harness, "projA", files=files)
    harness.users["bob"] = "pw"
    admin = auth_headers(client, "rainy")
    client.post("/api/v2/projects/scan", headers=admin)
    return {"admin": admin, "bob": auth_headers(client, "bob", "pw")}


def task_id(client, headers, rel="a.png") -> str:
    items = client.get("/api/v2/projects/projA/tasks", headers=headers).json()["items"]
    return next(t["anno_id"] for t in items if t["rel_path"] == rel)


def add_annotation(db, anno_id: str) -> None:
    """Pretend a save already happened (submit requires stored work)."""
    with db.session_scope() as session:
        task = session.scalar(select(Task).where(Task.anno_id == anno_id))
        session.add(Annotation(task_id=task.id, anno_id=anno_id, version=1, path="x", labels_json="[]"))


# region listing
def test_list_tasks_filters_and_pagination(client, auth_headers, harness):
    headers = bootstrap(
        client, auth_headers, harness, files=("images/dish01/D1.png", "images/dish01/D2.png", "z.png")
    )
    body = client.get(
        "/api/v2/projects/projA/tasks", params={"order": "sequence"}, headers=headers["admin"]
    ).json()
    assert body["total"] == 3
    assert [t["rel_path"] for t in body["items"]] == ["images/dish01/D1.png", "images/dish01/D2.png", "z.png"]
    assert body["items"][0]["group"] == "images/dish01"

    page = client.get("/api/v2/projects/projA/tasks", params={"limit": 2}, headers=headers["admin"]).json()
    assert len(page["items"]) == 2 and page["total"] == 3

    grouped = client.get(
        "/api/v2/projects/projA/tasks", params={"group": "images/dish01"}, headers=headers["admin"]
    ).json()
    assert grouped["total"] == 2

    states = client.get(
        "/api/v2/projects/projA/tasks", params={"state": "approved"}, headers=headers["admin"]
    ).json()
    assert states["total"] == 0

    bad = client.get("/api/v2/projects/projA/tasks", params={"state": "nope"}, headers=headers["admin"])
    assert bad.status_code == 422 and bad.json()["code"] == "validation_error"


def test_groups_endpoint_returns_tasks_per_sequence(client, auth_headers, harness):
    headers = bootstrap(
        client, auth_headers, harness, files=("species/dish/D1.png", "species/dish/D2.png", "loose.png")
    )
    groups = client.get("/api/v2/projects/projA/groups", headers=headers["admin"]).json()
    by_name = {g["group"]: g for g in groups}
    assert by_name["species/dish"]["count"] == 2
    assert [f["day"] for f in by_name["species/dish"]["tasks"]] == [1, 2]
    assert by_name[""]["count"] == 1


# endregion


# region claim / lease
def test_claim_then_conflict_for_the_second_annotator(client, auth_headers, harness):
    headers = bootstrap(client, auth_headers, harness)
    anno = task_id(client, headers["admin"])

    first = client.post(f"/api/v2/tasks/{anno}/claim", headers=headers["admin"])
    assert first.status_code == 200
    assert first.json()["claimed_by"] == "rainy"
    assert first.json()["lease_expires_at"] is not None

    second = client.post(f"/api/v2/tasks/{anno}/claim", headers=headers["bob"])
    assert second.status_code == 409
    body = second.json()
    assert body["code"] == "lease_conflict"
    assert body["detail"]["claimed_by"] == "rainy"
    assert body["detail"]["lease_expires_at"]


def test_claim_is_idempotent_for_the_holder(client, auth_headers, harness):
    headers = bootstrap(client, auth_headers, harness)
    anno = task_id(client, headers["admin"])
    client.post(f"/api/v2/tasks/{anno}/claim", headers=headers["admin"])
    again = client.post(f"/api/v2/tasks/{anno}/claim", headers=headers["admin"])
    assert again.status_code == 200 and again.json()["claimed_by"] == "rainy"


def test_expired_lease_can_be_taken_over(client, auth_headers, harness, db):
    headers = bootstrap(client, auth_headers, harness)
    anno = task_id(client, headers["admin"])
    client.post(f"/api/v2/tasks/{anno}/claim", headers=headers["admin"])
    with db.session_scope() as session:
        session.scalar(select(Task)).lease_expires_at = utcnow() - timedelta(minutes=1)

    stolen = client.post(f"/api/v2/tasks/{anno}/claim", headers=headers["bob"])
    assert stolen.status_code == 200 and stolen.json()["claimed_by"] == "bob"


def test_reviewer_can_force_a_claim(client, auth_headers, harness):
    headers = bootstrap(client, auth_headers, harness)
    anno = task_id(client, headers["admin"])
    client.post(f"/api/v2/tasks/{anno}/claim", headers=headers["admin"])

    assert (
        client.post(f"/api/v2/tasks/{anno}/claim", params={"force": True}, headers=headers["bob"]).status_code
        == 403
    )
    forced = client.post(f"/api/v2/tasks/{anno}/claim", params={"force": True}, headers=headers["admin"])
    assert forced.status_code == 200


def test_heartbeat_and_release(client, auth_headers, harness):
    headers = bootstrap(client, auth_headers, harness)
    anno = task_id(client, headers["admin"])
    client.post(f"/api/v2/tasks/{anno}/claim", headers=headers["admin"])

    assert client.post(f"/api/v2/tasks/{anno}/heartbeat", headers=headers["bob"]).status_code == 409
    beats = client.post(f"/api/v2/tasks/{anno}/heartbeat", headers=headers["admin"])
    assert beats.status_code == 200 and beats.json()["lease_expires_at"]

    # another annotator cannot release someone else's claim, a reviewer can
    assert client.post(f"/api/v2/tasks/{anno}/release", headers=headers["bob"]).status_code == 403
    released = client.post(f"/api/v2/tasks/{anno}/release", headers=headers["admin"])
    assert released.status_code == 200 and released.json()["claimed_by"] == ""


# endregion


# region submit / review
def test_submit_requires_the_claim_and_an_annotation(client, auth_headers, harness, db):
    headers = bootstrap(client, auth_headers, harness)
    anno = task_id(client, headers["admin"])
    client.post(f"/api/v2/tasks/{anno}/claim", headers=headers["bob"])

    # not the holder -> lease conflict
    conflict = client.post(f"/api/v2/tasks/{anno}/submit", headers=headers["admin"])
    assert conflict.status_code == 409 and conflict.json()["code"] == "lease_conflict"

    # holder but nothing saved yet -> validation error
    empty = client.post(f"/api/v2/tasks/{anno}/submit", headers=headers["bob"])
    assert empty.status_code == 422 and empty.json()["code"] == "validation_error"

    add_annotation(db, anno)
    submitted = client.post(f"/api/v2/tasks/{anno}/submit", headers=headers["bob"])
    assert submitted.status_code == 200
    assert submitted.json()["state"] == "submitted"
    assert submitted.json()["submitted_at"] is not None
    assert submitted.json()["lease_expires_at"] is None

    assert client.post(f"/api/v2/tasks/{anno}/submit", headers=headers["bob"]).status_code == 409


def test_review_requires_a_reviewer_and_a_note_on_reject(client, auth_headers, harness, db):
    headers = bootstrap(client, auth_headers, harness)
    anno = task_id(client, headers["admin"])
    client.post(f"/api/v2/tasks/{anno}/claim", headers=headers["bob"])
    add_annotation(db, anno)
    client.post(f"/api/v2/tasks/{anno}/submit", headers=headers["bob"])

    forbidden = client.post(
        f"/api/v2/tasks/{anno}/review", json={"decision": "approve"}, headers=headers["bob"]
    )
    assert forbidden.status_code == 403

    note_required = client.post(
        f"/api/v2/tasks/{anno}/review", json={"decision": "reject"}, headers=headers["admin"]
    )
    assert note_required.status_code == 422

    rejected = client.post(
        f"/api/v2/tasks/{anno}/review",
        json={"decision": "reject", "note": "wrong dish"},
        headers=headers["admin"],
    )
    assert rejected.status_code == 200
    body = rejected.json()
    assert body["state"] == "rejected" and body["review_note"] == "wrong dish"
    assert body["reviewed_by"] == "rainy"

    # reviewing again is a conflict (it is no longer submitted)
    assert (
        client.post(
            f"/api/v2/tasks/{anno}/review", json={"decision": "approve"}, headers=headers["admin"]
        ).status_code
        == 409
    )

    # a rejected task is free to pick up again
    again = client.post(f"/api/v2/tasks/{anno}/claim", headers=headers["bob"])
    assert again.status_code == 200 and again.json()["state"] == "rejected"


def test_approve_end_to_end(client, auth_headers, harness, db):
    headers = bootstrap(client, auth_headers, harness)
    anno = task_id(client, headers["admin"])
    client.post(f"/api/v2/tasks/{anno}/claim", headers=headers["admin"])
    add_annotation(db, anno)
    client.post(f"/api/v2/tasks/{anno}/submit", headers=headers["admin"])

    approved = client.post(
        f"/api/v2/tasks/{anno}/review", json={"decision": "approve"}, headers=headers["admin"]
    )
    assert approved.json()["state"] == "approved"
    progress = client.get("/api/v2/projects/projA/progress", headers=headers["admin"]).json()
    assert progress["approved"] == 1 and progress["finished"] == 1 and progress["total"] == 2


def test_approved_task_needs_a_reviewer_to_reopen(client, auth_headers, harness, db):
    headers = bootstrap(client, auth_headers, harness)
    anno = task_id(client, headers["admin"])
    add_annotation(db, anno)
    client.post(f"/api/v2/tasks/{anno}/claim", headers=headers["admin"])
    client.post(f"/api/v2/tasks/{anno}/submit", headers=headers["admin"])
    client.post(f"/api/v2/tasks/{anno}/review", json={"decision": "approve"}, headers=headers["admin"])

    assert client.post(f"/api/v2/tasks/{anno}/reopen", json={}, headers=headers["bob"]).status_code == 403
    reopened = client.post(f"/api/v2/tasks/{anno}/reopen", json={"note": "redo"}, headers=headers["admin"])
    assert reopened.status_code == 200
    assert reopened.json()["state"] == "draft" and reopened.json()["submitted_at"] is None


def test_my_stats_and_audit_trail(client, auth_headers, harness, db):
    headers = bootstrap(client, auth_headers, harness)
    anno = task_id(client, headers["admin"])
    client.post(f"/api/v2/tasks/{anno}/claim", headers=headers["bob"])

    stats = client.get("/api/v2/projects/projA/my-stats", headers=headers["bob"]).json()
    assert stats["draft"] == 1

    with db.session_scope() as session:
        actions = [row.action for row in session.scalars(select(AuditLog).order_by(AuditLog.id))]
    assert "claim" in actions and "login" in actions


# endregion


def test_random_order_returns_the_requested_page(client, auth_headers, harness):
    headers = bootstrap(client, auth_headers, harness, files=("a.png", "b.png", "c.png"))
    page = client.get(
        "/api/v2/projects/projA/tasks", params={"order": "random", "limit": 2}, headers=headers["admin"]
    ).json()
    assert page["total"] == 3 and len(page["items"]) == 2


def test_heartbeat_needs_a_live_lease(client, auth_headers, harness, db):
    """A lapsed or released lease cannot be renewed - re-claim instead."""
    headers = bootstrap(client, auth_headers, harness)
    anno = task_id(client, headers["admin"])
    client.post(f"/api/v2/tasks/{anno}/claim", headers=headers["admin"])

    with db.session_scope() as session:
        session.scalar(select(Task)).lease_expires_at = utcnow() - timedelta(seconds=1)
    expired = client.post(f"/api/v2/tasks/{anno}/heartbeat", headers=headers["admin"])
    assert expired.status_code == 409 and expired.json()["code"] == "lease_conflict"

    # re-claiming works (nobody else took it), and then the heartbeat is fine again
    assert client.post(f"/api/v2/tasks/{anno}/claim", headers=headers["admin"]).status_code == 200
    assert client.post(f"/api/v2/tasks/{anno}/heartbeat", headers=headers["admin"]).status_code == 200

    # submitting clears the lease, so there is nothing left to renew
    add_annotation(db, anno)
    client.post(f"/api/v2/tasks/{anno}/submit", headers=headers["admin"])
    assert client.post(f"/api/v2/tasks/{anno}/heartbeat", headers=headers["admin"]).status_code == 409
