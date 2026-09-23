"""P4: project membership, visibility and project-scoped roles."""

from __future__ import annotations

import pytest

from tests.v2.test_projects import seed
from v2.db.models import Project

MEMBERS = "/api/v2/projects/{project}/members"


@pytest.fixture
def two_users(client, auth_headers, harness):
    """(admin headers, bob's headers, bob's user id)."""
    admin = auth_headers(client)  # first account == admin
    harness.users["bob"] = "pw"
    bob = auth_headers(client, "bob", "pw")
    bob_id = client.get("/api/v2/auth/me", headers=bob).json()["id"]
    return admin, bob, bob_id


def test_membership_crud(two_users, client, harness):
    admin, bob, bob_id = two_users
    seed(harness, "projA", files=("a.png",))
    client.post("/api/v2/projects/scan", headers=admin)

    assert client.get(MEMBERS.format(project="projA"), headers=admin).json() == []
    added = client.post(
        MEMBERS.format(project="projA"), json={"user_id": bob_id, "role": "reviewer"}, headers=admin
    )
    assert added.status_code == 201 and added.json() == {"user_id": bob_id, "name": "bob", "role": "reviewer"}

    listed = client.get(MEMBERS.format(project="projA"), headers=admin).json()
    assert listed == [{"user_id": bob_id, "name": "bob", "role": "reviewer"}]

    # idempotent re-role
    patched = client.patch(
        MEMBERS.format(project="projA") + f"/{bob_id}", json={"role": "annotator"}, headers=admin
    )
    assert patched.status_code == 200 and patched.json()["role"] == "annotator"

    assert client.delete(MEMBERS.format(project="projA") + f"/{bob_id}", headers=admin).status_code == 204
    assert client.get(MEMBERS.format(project="projA"), headers=admin).json() == []
    # removing twice is a 404, not a crash
    assert client.delete(MEMBERS.format(project="projA") + f"/{bob_id}", headers=admin).status_code == 404


def test_open_mode_keeps_everyone_working(two_users, client, harness):
    """Default (open) access: any account can work on any project (pre-P4)."""
    admin, bob, _ = two_users
    seed(harness, "projA", files=("a.png",))
    client.post("/api/v2/projects/scan", headers=admin)

    assert [p["name"] for p in client.get("/api/v2/projects", headers=bob).json()] == ["projA"]
    assert client.get("/api/v2/projects/projA/tasks", headers=bob).status_code == 200
    assert client.get(MEMBERS.format(project="projA"), headers=bob).status_code == 200


def test_strict_mode_hides_and_blocks_foreign_projects(two_users, client, harness, settings):
    admin, bob, bob_id = two_users
    seed(harness, "projA", files=("a.png",))
    seed(harness, "projB", files=("b.png",))
    client.post("/api/v2/projects/scan", headers=admin)

    settings.project_access_mode = "strict"
    # bob belongs to projA only
    client.post(MEMBERS.format(project="projA"), json={"user_id": bob_id, "role": "annotator"}, headers=admin)

    assert [p["name"] for p in client.get("/api/v2/projects", headers=bob).json()] == ["projA"]
    assert client.get("/api/v2/projects/projA/tasks", headers=bob).status_code == 200
    assert client.get("/api/v2/projects/projB/tasks", headers=bob).status_code == 403
    assert client.get(MEMBERS.format(project="projB"), headers=bob).status_code == 403
    # the admin still sees everything
    assert {p["name"] for p in client.get("/api/v2/projects", headers=admin).json()} == {"projA", "projB"}


def test_project_reviewer_role_is_scoped(two_users, client, harness, settings, db):
    """A reviewer of one project must not review another."""
    admin, bob, bob_id = two_users
    seed(harness, "projA", files=("a.png",))
    seed(harness, "projB", files=("b.png",))
    client.post("/api/v2/projects/scan", headers=admin)
    settings.project_access_mode = "strict"

    for project in ("projA", "projB"):
        added = client.post(
            MEMBERS.format(project=project), json={"user_id": bob_id, "role": "annotator"}, headers=admin
        )
        assert added.status_code == 201, added.text
    promoted = client.patch(
        MEMBERS.format(project="projA") + f"/{bob_id}", json={"role": "reviewer"}, headers=admin
    )
    assert promoted.status_code == 200 and promoted.json()["role"] == "reviewer", promoted.text

    with db.session_scope() as session:
        session.query(Project).filter(Project.name == "projA").one()
    anno_a = next(
        t["anno_id"] for t in client.get("/api/v2/projects/projA/tasks", headers=bob).json()["items"]
    )
    anno_b = next(
        t["anno_id"] for t in client.get("/api/v2/projects/projB/tasks", headers=bob).json()["items"]
    )

    # in projA bob is a reviewer -> may reopen; in projB he is an annotator -> 403
    assert client.post(f"/api/v2/tasks/{anno_a}/reopen", json={}, headers=bob).status_code in (200, 409)
    assert client.post(f"/api/v2/tasks/{anno_b}/reopen", json={}, headers=bob).status_code == 403

    # labels follow the same rule
    assert client.post("/api/v2/projects/projA/labels", json={"name": "Root"}, headers=bob).status_code == 201
    assert client.post("/api/v2/projects/projB/labels", json={"name": "Root"}, headers=bob).status_code == 403


def test_membership_needs_a_project_reviewer(two_users, client, harness, settings):
    """An annotator cannot add members (not even to their own project)."""
    admin, bob, bob_id = two_users
    seed(harness, "projA", files=("a.png",))
    client.post("/api/v2/projects/scan", headers=admin)
    settings.project_access_mode = "strict"
    client.post(MEMBERS.format(project="projA"), json={"user_id": bob_id, "role": "annotator"}, headers=admin)

    assert (
        client.post(MEMBERS.format(project="projA"), json={"user_id": bob_id}, headers=bob).status_code == 403
    )
    assert client.get(MEMBERS.format(project="projA"), headers=bob).status_code == 200  # reading is fine


def test_unknown_user_or_role_is_rejected(two_users, client, harness):
    admin, _bob, _bob_id = two_users
    seed(harness, "projA", files=("a.png",))
    client.post("/api/v2/projects/scan", headers=admin)

    assert (
        client.post(MEMBERS.format(project="projA"), json={"user_id": 4242}, headers=admin).status_code == 404
    )
    assert (
        client.post(
            MEMBERS.format(project="projA"), json={"user_id": 1, "role": "wizard"}, headers=admin
        ).status_code
        == 422
    )
