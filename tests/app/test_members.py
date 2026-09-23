"""P4: project membership, visibility and project-scoped roles."""

from __future__ import annotations

import pytest

from app.db.models import Project
from tests.app.test_projects import seed

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


def _png(size=(8, 8), color=(7, 8, 9)) -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_strict_mode_blocks_every_foreign_project_surface(two_users, client, harness, settings):
    """Membership must gate the whole project surface, not just tasks.

    A non-member used to reach the label registry, the task images (read *and*
    upload) and the predict endpoint, while tasks/annotations/instances were
    already 403. Every project-scoped router now resolves the project the same
    way (``get_project(..., auth=...)`` / ``require_access``).
    """
    import json

    admin, bob, bob_id = two_users
    seed(harness, "projA", files=("a.png",))
    harness.add_file("/zlabel_server/projects/projA/images/a.png", _png())
    seed(harness, "projB", files=("b.png",))
    harness.add_file("/zlabel_server/projects/projB/images/b.png", _png())
    client.post("/api/v2/projects/scan", headers=admin)

    settings.project_access_mode = "strict"
    client.post(MEMBERS.format(project="projA"), json={"user_id": bob_id, "role": "annotator"}, headers=admin)

    def call_every_endpoint(
        project: str, headers: dict[str, str], anno_id: str, image_rel: str
    ) -> dict[str, int]:
        """Hit every project-scoped endpoint once; returns ``{label: status}``."""
        predict = {
            "data": (
                None,
                json.dumps({"anno_id": anno_id, "mode": 1, "return_type": 1, "rel_path": image_rel}),
            )
        }
        responses = {
            "list_labels": client.get(f"/api/v2/projects/{project}/labels", headers=headers),
            "list_instances": client.get(f"/api/v2/projects/{project}/instances", headers=headers),
            "instance_statuses": client.get(
                f"/api/v2/projects/{project}/instances/statuses", headers=headers
            ),
            "list_tasks": client.get(f"/api/v2/projects/{project}/tasks", headers=headers),
            "groups": client.get(f"/api/v2/projects/{project}/groups", headers=headers),
            "progress": client.get(f"/api/v2/projects/{project}/progress", headers=headers),
            "my_stats": client.get(f"/api/v2/projects/{project}/my-stats", headers=headers),
            "get_project": client.get(f"/api/v2/projects/{project}", headers=headers),
            "members": client.get(MEMBERS.format(project=project), headers=headers),
            "get_annotation": client.get(
                f"/api/v2/projects/{project}/annotations/{anno_id}", headers=headers
            ),
            "save_annotation": client.put(
                f"/api/v2/projects/{project}/annotations/{anno_id}", json={"id": anno_id}, headers=headers
            ),
            "versions": client.get(
                f"/api/v2/projects/{project}/annotations/{anno_id}/versions", headers=headers
            ),
            "get_image": client.get(f"/api/v2/projects/{project}/images/{image_rel}", headers=headers),
            "upload_image": client.put(
                f"/api/v2/projects/{project}/images/upload.png",
                files={"file": ("x.png", _png())},
                headers=headers,
            ),
            "predict": client.post(f"/api/v2/projects/{project}/predict", files=predict, headers=headers),
            "scan": client.post(f"/api/v2/projects/{project}/scan", headers=headers),
        }
        return {label: response.status_code for label, response in responses.items()}

    foreign = call_every_endpoint("projB", bob, anno_id="deadbeef", image_rel="b.png")
    assert foreign == dict.fromkeys(foreign, 403), foreign

    # a member of projA reaches the same endpoints (the fix is membership, not a
    # blanket block), and the annotator-only limits still apply on top of it.
    a_anno = client.get("/api/v2/projects/projA/tasks", headers=admin).json()["items"][0]["anno_id"]
    mine = call_every_endpoint("projA", bob, anno_id=a_anno, image_rel="a.png")
    assert mine["list_labels"] == 200
    assert mine["list_instances"] == 200
    assert mine["instance_statuses"] == 200
    assert mine["list_tasks"] == 200
    assert mine["groups"] == 200
    assert mine["progress"] == 200
    assert mine["my_stats"] == 200
    assert mine["get_project"] == 200
    assert mine["members"] == 200
    assert mine["get_image"] == 200
    assert mine["upload_image"] == 200
    assert mine["get_annotation"] == 404  # accessible, just not annotated yet
    assert mine["save_annotation"] == 200
    assert mine["versions"] == 200
    assert mine["predict"] == 503  # reachable; the tests configure no inference worker
    assert mine["scan"] == 403  # membership + role: an annotator still cannot scan
