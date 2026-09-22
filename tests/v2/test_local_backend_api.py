"""The whole API on the local disk backend (no OpenList in the storage path)."""

from __future__ import annotations

from pathlib import Path


def seed_image(root: Path, project: str, rel: str) -> Path:
    """Drop a frame straight into the storage tree (no OpenList involved)."""
    path = root / project / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png-bytes")
    return path


def test_scan_finds_directories_without_a_marker_file(local_client, local_settings, auth_headers):
    seed_image(Path(local_settings.storage_root), "projA", "images/dish01/D1.png")
    seed_image(Path(local_settings.storage_root), "projA", "images/dish01/D2.png")
    seed_image(Path(local_settings.storage_root), "projB", "x.png")
    headers = auth_headers(local_client)  # the fixture account is the admin

    stats = local_client.post("/api/v2/projects/scan", headers=headers).json()
    assert stats == {"projects": 2, "tasks": 3, "missing": 0, "deactivated": 0, "skipped": 0}

    projects = local_client.get("/api/v2/projects", headers=headers).json()
    assert [p["name"] for p in projects] == ["projA", "projB"]
    tasks = local_client.get("/api/v2/projects/projA/tasks", headers=headers).json()["items"]
    assert [t["rel_path"] for t in tasks] == ["images/dish01/D1.png", "images/dish01/D2.png"]
    assert tasks[0]["group"] == "images/dish01" and tasks[0]["day"] == 1


def test_annotation_roundtrip_writes_the_desktop_layout(local_client, local_settings, auth_headers):
    root = Path(local_settings.storage_root)
    seed_image(root, "projA", "images/dish01/D1.png")
    headers = auth_headers(local_client)
    local_client.post("/api/v2/projects/scan", headers=headers)
    anno_id = local_client.get("/api/v2/projects/projA/tasks", headers=headers).json()["items"][0]["anno_id"]

    missing = local_client.get(f"/api/v2/projects/projA/annotations/{anno_id}", headers=headers)
    assert missing.status_code == 404  # not annotated yet

    saved = local_client.put(
        f"/api/v2/projects/projA/annotations/{anno_id}",
        json={"results": {"r1": {"labels": [{"name": "Root"}]}}},
        headers=headers,
    )
    assert saved.status_code == 200 and saved.json()["version"] == 1

    # the file landed where the desktop keeps its own annotations
    anno_file = root / "projA" / ".zlabel" / "annos" / f"{anno_id}.zlabel"
    history = root / "projA" / ".zlabel" / "annos" / "_history" / anno_id / "v1.zlabel"
    assert anno_file.is_file() and history.is_file()
    assert "Root" in anno_file.read_text(encoding="utf-8")

    # and it reads back through the API
    got = local_client.get(f"/api/v2/projects/projA/annotations/{anno_id}", headers=headers)
    assert got.status_code == 200 and got.headers["X-Anno-Version"] == "1"
    versions = local_client.get(
        f"/api/v2/projects/projA/annotations/{anno_id}/versions", headers=headers
    ).json()
    assert [v["version"] for v in versions] == [1]


def test_images_are_served_from_disk(local_client, local_settings, auth_headers):
    root = Path(local_settings.storage_root)
    seed_image(root, "projA", "images/a/D1.png")
    headers = auth_headers(local_client)
    local_client.post("/api/v2/projects/scan", headers=headers)

    resp = local_client.get("/api/v2/projects/projA/images/images/a/D1.png", headers=headers)
    assert resp.status_code == 200 and resp.content == b"png-bytes"
    assert resp.headers["ETag"] and resp.headers["X-Image-Sha256"]

    cached = local_client.get(
        "/api/v2/projects/projA/images/images/a/D1.png",
        headers={**headers, "If-None-Match": resp.headers["ETag"]},
    )
    assert cached.status_code == 304
    assert local_client.get("/api/v2/projects/projA/images/nope.png", headers=headers).status_code == 404


def test_vanished_directories_are_deactivated(local_client, local_settings, auth_headers):
    import shutil

    root = Path(local_settings.storage_root)
    seed_image(root, "projA", "a.png")
    seed_image(root, "gone", "b.png")
    headers = auth_headers(local_client)
    local_client.post("/api/v2/projects/scan", headers=headers)
    assert {p["name"] for p in local_client.get("/api/v2/projects", headers=headers).json()} == {
        "projA",
        "gone",
    }

    shutil.rmtree(root / "gone")
    local_client.post("/api/v2/projects/scan", headers=headers)
    assert {p["name"] for p in local_client.get("/api/v2/projects", headers=headers).json()} == {"projA"}


def test_creating_a_project_prepares_the_directory(local_client, local_settings, auth_headers):
    headers = auth_headers(local_client)
    created = local_client.post(
        "/api/v2/projects", json={"name": "fresh", "display_name": "Fresh"}, headers=headers
    )
    assert created.status_code == 201

    project_dir = Path(local_settings.storage_root) / "fresh"
    assert project_dir.is_dir()
    # the marker file is written too, so an OpenList deployment still recognises it
    assert (project_dir / local_settings.project_marker).is_file()

    # a project without frames is still listed (no marker-driven discovery)
    local_client.post("/api/v2/projects/scan", headers=headers)
    assert "fresh" in {p["name"] for p in local_client.get("/api/v2/projects", headers=headers).json()}


def test_zero_openlist_end_to_end(local_client, local_settings, auth_headers):
    """No OpenList anywhere: local accounts + local storage, full annotate flow."""
    root = Path(local_settings.storage_root)
    seed_image(root, "projA", "images/dish01/D1.png")
    headers = auth_headers(local_client)

    me = local_client.get("/api/v2/auth/me", headers=headers).json()
    assert me["role"] == "admin"

    local_client.post("/api/v2/projects/scan", headers=headers)
    anno_id = local_client.get("/api/v2/projects/projA/tasks", headers=headers).json()["items"][0]["anno_id"]

    assert local_client.post(f"/api/v2/tasks/{anno_id}/claim", headers=headers).status_code == 200
    saved = local_client.put(
        f"/api/v2/projects/projA/annotations/{anno_id}",
        json={"results": {"r1": {"labels": [{"name": "Root"}]}}},
        headers=headers,
    )
    assert saved.status_code == 200
    assert local_client.post(f"/api/v2/tasks/{anno_id}/submit", headers=headers).status_code == 200
    reviewed = local_client.post(
        f"/api/v2/tasks/{anno_id}/review", json={"decision": "approve"}, headers=headers
    )
    assert reviewed.status_code == 200 and reviewed.json()["state"] == "approved"

    progress = local_client.get("/api/v2/projects/projA/progress", headers=headers).json()
    assert progress["approved"] == 1 and progress["finished"] == 1
    health = local_client.get("/api/v2/health").json()
    assert health["status"] == "ok" and "tasks.claim" in health["capabilities"]
