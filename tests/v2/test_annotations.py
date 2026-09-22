"""Annotation save: versioning, optimistic locking, history, label linkage."""

from __future__ import annotations

import json

from sqlalchemy import select

from tests.v2.test_projects import seed
from v2.db.models import Annotation, AnnotationVersion, Label, Task

ROOT = "/zlabel_server/projects"


def bootstrap(client, auth_headers, ol, files=("a.png", "b.png")) -> dict:
    seed(ol, "projA", files=files)
    ol.users["bob"] = "pw"
    admin = auth_headers(client, "rainy")
    client.post("/api/v2/projects/scan", headers=admin)
    return {"admin": admin, "bob": auth_headers(client, "bob", "pw")}


def task_id(client, headers, rel="a.png") -> str:
    items = client.get("/api/v2/projects/projA/tasks", headers=headers).json()["items"]
    return next(t["anno_id"] for t in items if t["rel_path"] == rel)


def document(*labels: str, note: str = "") -> dict:
    return {
        "id": "ignored-by-the-server",
        "note": note,
        "results": {f"r{i}": {"labels": [{"name": name}]} for i, name in enumerate(labels)},
    }


def hand_over(client, admin_headers, bob_headers, anno) -> None:
    """Release the frame as a reviewer, then let the second annotator claim it."""
    assert client.post(f"/api/v2/tasks/{anno}/release", headers=admin_headers).status_code == 200
    assert client.post(f"/api/v2/tasks/{anno}/claim", headers=bob_headers).status_code == 200


def put(client, headers, anno, payload, **params):
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"/api/v2/projects/projA/annotations/{anno}" + (f"?{query}" if query else "")
    return client.put(url, json=payload, headers=headers)


# region basics
def test_save_creates_version_one_and_claims_the_task(client, auth_headers, ol, db):
    headers = bootstrap(client, auth_headers, ol)
    anno = task_id(client, headers["admin"])

    resp = put(client, headers["admin"], anno, document("Root", "Shoot"))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["version"] == 1 and body["state"] == "draft"
    assert body["labels"] == ["Root", "Shoot"]

    # the document landed in OpenList, current + history
    assert json.loads(ol.files[f"{ROOT}/projA/.zlabel/annos/{anno}.zlabel"])["results"]
    assert f"{ROOT}/projA/.zlabel/annos/_history/{anno}/v1.zlabel" in ol.files

    with db.session_scope() as session:
        task = session.scalar(select(Task))
        assert task.claimed_by is not None and task.lease_expires_at is not None
        assert {label.name for label in task.labels} == {"Root", "Shoot"}
        assert session.scalar(select(Annotation)).version == 1
        assert session.scalar(select(AnnotationVersion)).version == 1


def test_get_returns_the_document_and_404_when_missing(client, auth_headers, ol):
    headers = bootstrap(client, auth_headers, ol)
    anno = task_id(client, headers["admin"])

    missing = client.get(f"/api/v2/projects/projA/annotations/{anno}", headers=headers["admin"])
    assert missing.status_code == 404 and missing.json()["code"] == "not_found"

    put(client, headers["admin"], anno, document("Root"))
    got = client.get(f"/api/v2/projects/projA/annotations/{anno}", headers=headers["admin"])
    assert got.status_code == 200
    assert got.headers["X-Anno-Version"] == "1"
    assert got.headers["ETag"] == '"v1"'
    assert got.json()["results"]["r0"]["labels"][0]["name"] == "Root"


def test_second_save_bumps_the_version(client, auth_headers, ol, db):
    headers = bootstrap(client, auth_headers, ol)
    anno = task_id(client, headers["admin"])
    put(client, headers["admin"], anno, document("Root"))
    put(client, headers["admin"], anno, document("Root", "Shoot"), base_version=1)

    with db.session_scope() as session:
        assert session.scalar(select(Annotation)).version == 2
        versions = [
            v.version for v in session.scalars(select(AnnotationVersion).order_by(AnnotationVersion.version))
        ]
        assert versions == [1, 2]
    assert f"{ROOT}/projA/.zlabel/annos/_history/{anno}/v2.zlabel" in ol.files


# endregion


# region optimistic locking
def test_stale_base_version_conflicts(client, auth_headers, ol):
    headers = bootstrap(client, auth_headers, ol)
    anno = task_id(client, headers["admin"])
    put(client, headers["admin"], anno, document("Root"))  # v1 by rainy
    put(client, headers["admin"], anno, document("Root", "Shoot"), base_version=1)  # v2 by rainy

    # bob edits based on v1 (the frame is handed over, so the claim is not the issue)
    hand_over(client, headers["admin"], headers["bob"], anno)
    stale = put(client, headers["bob"], anno, document("Leaf"), base_version=1)
    assert stale.status_code == 409
    body = stale.json()
    assert body["code"] == "conflict"
    assert body["detail"]["server_version"] == 2
    assert body["detail"]["updated_by"] == "rainy"
    assert body["detail"]["updated_at"]


def test_save_without_base_version_needs_a_free_version(client, auth_headers, ol):
    headers = bootstrap(client, auth_headers, ol)
    anno = task_id(client, headers["admin"])
    put(client, headers["admin"], anno, document("Root"))
    # a blind write on top of v1 is refused: nobody may clobber unseen work
    assert put(client, headers["admin"], anno, document("Root2")).status_code == 409
    assert put(client, headers["admin"], anno, document("Root2"), base_version=1).status_code == 200


def test_force_overwrite_is_reviewer_only(client, auth_headers, ol, db):
    headers = bootstrap(client, auth_headers, ol)
    anno = task_id(client, headers["admin"])
    put(client, headers["admin"], anno, document("Root"))
    hand_over(client, headers["admin"], headers["bob"], anno)
    assert put(client, headers["bob"], anno, document("Leaf"), base_version=1).status_code == 200

    denied = put(client, headers["bob"], anno, document("Leaf2"), base_version=2, force="true")
    assert denied.status_code == 403

    forced = put(client, headers["admin"], anno, document("Root3"), base_version=1, force="true")
    assert forced.status_code == 200 and forced.json()["version"] == 3
    with db.session_scope() as session:
        assert session.scalar(select(Annotation)).version == 3


def test_lease_conflict_on_save(client, auth_headers, ol):
    headers = bootstrap(client, auth_headers, ol)
    anno = task_id(client, headers["admin"])
    put(client, headers["admin"], anno, document("Root"))

    other = put(client, headers["bob"], anno, document("Leaf"))
    assert other.status_code == 409 and other.json()["code"] == "lease_conflict"
    assert other.json()["detail"]["claimed_by"] == "rainy"


def test_editing_a_submitted_task_needs_a_reviewer(client, auth_headers, ol):
    headers = bootstrap(client, auth_headers, ol)
    anno = task_id(client, headers["admin"])
    put(client, headers["admin"], anno, document("Root"), base_version=0)
    assert client.post(f"/api/v2/tasks/{anno}/submit", headers=headers["admin"]).status_code == 200

    locked = put(client, headers["admin"], anno, document("Root2"), base_version=1)
    assert locked.status_code == 409 and locked.json()["detail"]["state"] == "submitted"

    forced = put(client, headers["admin"], anno, document("Root2"), base_version=1, force="true")
    assert forced.status_code == 200 and forced.json()["state"] == "submitted"


def test_reworking_a_rejected_task_returns_it_to_draft(client, auth_headers, ol):
    headers = bootstrap(client, auth_headers, ol)
    anno = task_id(client, headers["admin"])
    put(client, headers["admin"], anno, document("Root"), base_version=0)
    client.post(f"/api/v2/tasks/{anno}/submit", headers=headers["admin"])
    client.post(
        f"/api/v2/tasks/{anno}/review", json={"decision": "reject", "note": "fix"}, headers=headers["admin"]
    )

    reworked = put(client, headers["admin"], anno, document("Root", "Shoot"), base_version=1)
    assert reworked.status_code == 200
    assert reworked.json()["state"] == "draft"


# endregion


# region history
def test_version_history_and_preview(client, auth_headers, ol):
    headers = bootstrap(client, auth_headers, ol)
    anno = task_id(client, headers["admin"])
    put(client, headers["admin"], anno, document("Root"), note="first")
    put(client, headers["admin"], anno, document("Root", "Shoot"), base_version=1)

    versions = client.get(
        f"/api/v2/projects/projA/annotations/{anno}/versions", headers=headers["admin"]
    ).json()
    assert [v["version"] for v in versions] == [2, 1]
    assert versions[1]["author"] == "rainy" and versions[1]["note"] == "first"
    assert versions[1]["labels"] == ["Root"]

    old = client.get(f"/api/v2/projects/projA/annotations/{anno}/versions/1", headers=headers["admin"])
    assert old.status_code == 200
    assert [r["labels"][0]["name"] for r in old.json()["results"].values()] == ["Root"]

    current = client.get(f"/api/v2/projects/projA/annotations/{anno}/versions/2", headers=headers["admin"])
    assert len(current.json()["results"]) == 2


def test_label_registry_grows_with_annotations(client, auth_headers, ol):
    headers = bootstrap(client, auth_headers, ol)
    anno = task_id(client, headers["admin"])
    put(client, headers["admin"], anno, document("Root", "Shoot"))

    labels = client.get("/api/v2/projects/projA/labels", headers=headers["admin"]).json()
    assert sorted(x["name"] for x in labels) == ["Root", "Shoot"]
    with client.app.state.db.session_scope() as session:
        assert len(session.scalars(select(Label)).all()) == 2


def test_unknown_task_and_project_mismatch_are_404(client, auth_headers, ol):
    headers = bootstrap(client, auth_headers, ol)
    anno = task_id(client, headers["admin"])
    assert put(client, headers["admin"], "deadbeef", document("Root")).status_code == 404
    assert put(client, headers["admin"], anno, document("Root")).status_code in (200, 409)
    other = client.put(
        f"/api/v2/projects/other/annotations/{anno}", json=document("Root"), headers=headers["admin"]
    )
    assert other.status_code == 404


# endregion
