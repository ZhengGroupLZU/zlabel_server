"""Project-scoped instances: the mirror from documents, CRUD and counts."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import func, select

from app.db.models import Annotation, Instance, InstanceResult, Project
from app.services.instance_service import parse_instance_annotations
from app.services.label_palette import LABEL_PALETTE


def _document(**instances: int) -> dict:
    """A tiny wire document: ``results[rN]`` with an instance id + label + status."""
    results = {}
    statuses = {}
    for index, (label, number) in enumerate(instances.items(), start=1):
        results[f"r{index}"] = {"labels": [{"name": label}], "instance_id": number}
        if number:
            statuses[str(number)] = "normal_seed" if number == 1 else "moldy_seed"
    return {"image_path": "images/D1.png", "results": results, "instances": statuses}


def _save(client, headers, anno_id: str, document: dict) -> None:
    """PUT a document, carrying the current version (optimistic locking)."""
    current = client.get(f"/api/v2/projects/projA/annotations/{anno_id}", headers=headers)
    version = int(current.headers.get("X-Anno-Version", 0)) if current.status_code == 200 else 0
    params = {"base_version": version} if version else None
    resp = client.put(
        f"/api/v2/projects/projA/annotations/{anno_id}",
        json=document,
        params=params,
        headers=headers,
    )
    assert resp.status_code == 200, resp.text


def _annotate(client, auth_headers, harness, document: dict) -> str:
    """Seed one task, scan it and save ``document`` as its annotation."""
    from tests.app.test_projects import seed

    seed(harness, "projA", files=("images/D1.png",))
    headers = auth_headers(client)
    client.post("/api/v2/projects/scan", headers=headers)
    anno_id = client.get("/api/v2/projects/projA/tasks", headers=headers).json()["items"][0]["anno_id"]
    _save(client, headers, anno_id, document)
    return anno_id


# region parsing
def test_parse_instance_annotations():
    parsed = parse_instance_annotations(
        {
            "results": {
                "r1": {"labels": [{"name": "Seed"}], "instance_id": 2},
                "r2": {"labels": [{"name": "Root"}], "instance_id": 2},
                "r3": {"labels": [{"name": "Dish"}], "instance_id": 0},
                "r4": {"labels": [{"name": "Shoot"}]},
            },
            "instances": {"2": "normal_seed", "9": "unused"},
        }
    )
    assert set(parsed) == {2}
    assert parsed[2]["status"] == "normal_seed"
    assert parsed[2]["results"] == [("r1", "Seed"), ("r2", "Root")]


def test_parse_instance_annotations_accepts_a_list():
    parsed = parse_instance_annotations(
        {"results": [{"id": "r9", "labels": [{"name": "Seed"}], "instance_id": "3"}]}
    )
    assert parsed == {3: {"status": "", "results": [("r9", "Seed")]}}


# endregion


# region mirror
def test_saving_an_annotation_mirrors_instances(client, auth_headers, harness, services):
    _annotate(client, auth_headers, harness, _document(Seed=1, Root=1, Shoot=2))

    rows = services.instances.list_instances("projA")
    assert [(row["number"], row["status"]) for row in rows] == [(1, "normal_seed"), (2, "moldy_seed")]
    assert [row["results"] for row in rows] == [2, 1]
    assert [row["tasks"] for row in rows] == [1, 1]
    assert all(row["color"] in LABEL_PALETTE for row in rows)
    assert [row["color"] for row in rows] == list(dict.fromkeys(row["color"] for row in rows))

    results = services.instances.results_of("projA", 1)
    assert [(item["result_id"], item["label"]) for item in results] == [
        ("r1", "Seed"),
        ("r2", "Root"),
    ]
    assert results[0]["rel_path"] == "images/D1.png"


def test_resaving_rewrites_the_links(client, auth_headers, harness, services):
    anno_id = _annotate(client, auth_headers, harness, _document(Seed=1, Root=1, Shoot=2))
    headers = auth_headers(client)
    # Root moves to instance 2 and Shoot disappears
    _save(client, headers, anno_id, _document(Seed=1, Root=2))

    rows = {row["number"]: row for row in services.instances.list_instances("projA")}
    assert [row["results"] for row in rows.values()] == [1, 1]
    assert [item["result_id"] for item in services.instances.results_of("projA", 2)] == ["r2"]


def test_the_mirror_keeps_an_edited_status(client, auth_headers, harness, services):
    anno_id = _annotate(client, auth_headers, harness, _document(Seed=1))
    services.instances.update_instance("projA", 1, status="dead_seed")

    headers = auth_headers(client)
    _save(client, headers, anno_id, _document(Seed=1))

    assert services.instances.list_instances("projA")[0]["status"] == "dead_seed"


def test_backfill_syncs_existing_documents(client, auth_headers, harness, services, db):
    """``sync-instances`` re-mirrors documents stored before the tables existed."""
    anno_id = _annotate(client, auth_headers, harness, _document(Seed=1, Shoot=2))
    with db.session_scope() as session:
        session.execute(InstanceResult.__table__.delete())
        session.execute(Instance.__table__.delete())
        assert session.scalar(select(func.count()).select_from(Instance)) == 0

    stats = services.instances.sync_project("projA")

    assert stats == {"documents": 1, "instances": 2, "results": 2}
    assert [row["number"] for row in services.instances.list_instances("projA")] == [1, 2]
    assert anno_id  # the document was the source


# endregion


# region crud + api
def test_instance_crud(client, auth_headers):
    headers = auth_headers(client)
    assert client.post("/api/v2/projects", json={"name": "projA"}, headers=headers).status_code == 201

    created = client.post(
        "/api/v2/projects/projA/instances",
        json={"name": "seed-7", "status": "normal_seed"},
        headers=headers,
    )
    assert created.status_code == 201, created.text
    assert created.json()["number"] == 1 and created.json()["color"] in LABEL_PALETTE

    # an explicit number is honoured, a duplicate is refused
    second = client.post("/api/v2/projects/projA/instances", json={"number": 5}, headers=headers)
    assert second.json()["number"] == 5
    assert (
        client.post("/api/v2/projects/projA/instances", json={"number": 5}, headers=headers).status_code
        == 409
    )
    # ... and the next free number follows the highest one
    assert client.post("/api/v2/projects/projA/instances", json={}, headers=headers).json()["number"] == 6

    patched = client.patch(
        "/api/v2/projects/projA/instances/1",
        json={"status": "dead_seed", "color": "#abc", "archived": True, "note": "n"},
        headers=headers,
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["status"] == "dead_seed"
    assert patched.json()["color"] == "#aabbcc" and patched.json()["archived"] is True

    listed = client.get("/api/v2/projects/projA/instances", headers=headers).json()
    assert [row["number"] for row in listed] == [5, 6]  # 1 is archived
    assert [
        row["number"]
        for row in client.get(
            "/api/v2/projects/projA/instances", params={"include_archived": "true"}, headers=headers
        ).json()
    ] == [1, 5, 6]

    assert client.delete("/api/v2/projects/projA/instances/5", headers=headers).status_code == 204
    assert [row["number"] for row in listed] == [5, 6]  # the earlier listing is a snapshot
    assert client.get("/api/v2/projects/projA/instances/5/results", headers=headers).status_code == 404

    statuses = client.get("/api/v2/projects/projA/instances/statuses", headers=headers).json()
    assert statuses[:2] == ["normal_seed", "moldy_seed"] and "dead_seed" in statuses


def test_instance_writes_need_a_reviewer(client, auth_headers, harness):
    from tests.app.test_projects import seed

    seed(harness, "projA", files=("images/D1.png",))
    admin = auth_headers(client)
    client.post("/api/v2/projects/scan", headers=admin)
    client.post(
        "/api/v2/admin/users",
        json={"name": "ann1", "password": "secret123", "role": "annotator"},
        headers=admin,
    )
    annotator = auth_headers(client, "ann1", "secret123")

    assert client.get("/api/v2/projects/projA/instances", headers=annotator).status_code == 200
    assert client.post("/api/v2/projects/projA/instances", json={}, headers=annotator).status_code == 403
    created = client.post("/api/v2/projects/projA/instances", json={}, headers=admin).json()
    assert (
        client.patch(
            f"/api/v2/projects/projA/instances/{created['number']}", json={"name": "x"}, headers=annotator
        ).status_code
        == 403
    )
    assert (
        client.delete(f"/api/v2/projects/projA/instances/{created['number']}", headers=annotator).status_code
        == 403
    )


def test_deleting_an_instance_keeps_the_documents(client, auth_headers, harness, services, db):
    anno_id = _annotate(client, auth_headers, harness, _document(Seed=1, Root=1))
    assert services.instances.list_instances("projA")[0]["results"] == 2

    services.instances.delete_instance("projA", 1)

    assert services.instances.list_instances("projA") == []
    with db.session_scope() as session:
        assert session.scalar(select(func.count()).select_from(InstanceResult)) == 0
        assert session.scalar(select(Annotation)) is not None  # the document is untouched
    # a later save that still references the number recreates the registry row
    headers = auth_headers(client)
    _save(client, headers, anno_id, _document(Seed=1, Root=1))
    assert services.instances.list_instances("projA")[0]["results"] == 2


def test_instance_paths_are_project_scoped(client, auth_headers, harness, services, db):
    from tests.app.test_projects import seed

    seed(harness, "projA", files=("images/D1.png",))
    seed(harness, "projB", files=("images/D1.png",))
    headers = auth_headers(client)
    client.post("/api/v2/projects/scan", headers=headers)
    services.instances.create_instance("projA", number=1, name="a")
    services.instances.create_instance("projB", number=1, name="b")

    with db.session_scope() as session:
        assert session.scalar(select(func.count()).select_from(Instance)) == 2
        assert session.scalar(select(func.count()).select_from(Project)) == 2
    assert services.instances.list_instances("projA")[0]["name"] == "a"
    assert services.instances.list_instances("projB")[0]["name"] == "b"


def test_unknown_projects_and_instances_are_404(client, auth_headers):
    headers = auth_headers(client)
    assert client.get("/api/v2/projects/nope/instances", headers=headers).status_code == 404
    client.post("/api/v2/projects", json={"name": "projA"}, headers=headers)
    assert client.patch("/api/v2/projects/projA/instances/9", json={}, headers=headers).status_code == 404
    assert client.delete("/api/v2/projects/projA/instances/9", headers=headers).status_code == 404


def test_stored_documents_are_the_source_of_truth(client, auth_headers, harness, services, db):
    """The JSON on disk keeps the instance data even when the registry is cleared."""
    _annotate(client, auth_headers, harness, _document(Seed=1, Root=1))
    doc_path = Path(services.settings.storage_root)
    with db.session_scope() as session:
        annotation = session.scalar(select(Annotation))
        stored = json.loads(
            (doc_path / "projA" / ".zlabel" / "annos" / f"{annotation.anno_id}.zlabel").read_text(
                encoding="utf-8"
            )
        )
    assert stored["results"]["r1"]["instance_id"] == 1
    assert stored["instances"]["1"] == "normal_seed"


# endregion
