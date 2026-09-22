"""Project discovery/sync, labels and progress."""

from __future__ import annotations

from sqlalchemy import select

from v2.contracts.ids import anno_id_for
from v2.db.models import Label, Project, Task, User
from v2.services.project_service import ProjectService
from v2.vendor.openlist_api import OpenListAPIError

ROOT = "/zlabel_server/projects"
SUFFIX = "/.zlabel-server-project-root"


def seed(
    ol,
    name: str = "projA",
    *,
    marker: bool = True,
    files: tuple[str, ...] = (),
) -> str:
    ol.add_dir(f"{ROOT}/{name}")
    if marker:
        ol.add_file(f"{ROOT}/{name}{SUFFIX}", b"root")
    for rel in files:
        ol.add_file(f"{ROOT}/{name}/{rel}", b"png")
    return name


def service(services) -> ProjectService:
    return services.projects


# region scanning
def test_scan_requires_the_marker_file(services, ol):
    seed(ol, "projA", files=("images/dish01/D1.png",))
    seed(ol, "not_a_project", marker=False, files=("x.png",))

    stats = service(services).scan_and_sync(force=True)
    assert stats["projects"] == 1 and stats["tasks"] == 1

    with services.db.session_scope() as session:
        assert session.scalar(select(Project.name)) == "projA"
        task = session.scalar(select(Task))
        assert task.rel_path == "images/dish01/D1.png"
        assert task.group_name == "images/dish01" and task.day == 1
        assert task.anno_id == anno_id_for("projA", "images/dish01/D1.png")


def test_scan_marks_vanished_files_missing(services, ol):
    seed(ol, "projA", files=("a.png", "b.png"))
    service(services).scan_and_sync(force=True)
    del ol.files[f"{ROOT}/projA/b.png"]
    stats = service(services).scan_and_sync(force=True)
    assert stats["missing"] == 1

    with services.db.session_scope() as session:
        assert session.scalar(select(Task).where(Task.rel_path == "b.png")).missing is True

    ol.add_file(f"{ROOT}/projA/b.png", b"png")  # the file comes back
    service(services).scan_and_sync(force=True)
    with services.db.session_scope() as session:
        assert session.scalar(select(Task).where(Task.rel_path == "b.png")).missing is False


def test_removing_the_marker_deactivates_the_project(services, ol):
    seed(ol, "projA", files=("a.png",))
    service(services).scan_and_sync(force=True)
    del ol.files[f"{ROOT}/projA{SUFFIX}"]

    service(services).scan_and_sync(force=True)
    with services.db.session_scope() as session:
        assert session.scalar(select(Project)).active is False


def test_uninspectable_markers_do_not_deactivate(services, ol):
    """A flaky OpenList must never make us drop projects."""
    seed(ol, "projA", files=("a.png",))
    service(services).scan_and_sync(force=True)
    ol.fail_on(f"{ROOT}/projA{SUFFIX}", OpenListAPIError("storage offline", status_code=500))

    service(services).scan_and_sync(force=True)
    with services.db.session_scope() as session:
        assert session.scalar(select(Project)).active is True


def test_scan_is_throttled_unless_forced(services, ol):
    seed(ol, "projA", files=("a.png",))
    first = service(services).scan_and_sync(force=True)
    second = service(services).scan_and_sync()  # inside the throttle window
    assert first["skipped"] == 0 and second["skipped"] == 1


# endregion


# region api
def test_projects_endpoint_lists_progress(client, auth_headers, ol):
    seed(ol, "projA", files=("a.png", "b.png"))
    headers = auth_headers(client)
    assert client.post("/api/v2/projects/scan", headers=headers).status_code == 200

    projects = client.get("/api/v2/projects", headers=headers).json()
    assert [p["name"] for p in projects] == ["projA"]
    assert projects[0]["progress"]["total"] == 2
    assert projects[0]["progress"]["draft"] == 2


def test_create_project_writes_the_marker(client, auth_headers, ol):
    headers = auth_headers(client)  # first user == admin
    resp = client.post("/api/v2/projects", json={"name": "fresh", "display_name": "Fresh"}, headers=headers)
    assert resp.status_code == 201, resp.text
    assert f"{ROOT}/fresh{SUFFIX}" in ol.files
    assert resp.json()["display_name"] == "Fresh"

    assert client.post("/api/v2/projects", json={"name": "fresh"}, headers=headers).status_code == 409
    assert client.post("/api/v2/projects", json={"name": "a/b"}, headers=headers).status_code == 409


def test_project_writes_need_a_reviewer(client, auth_headers, ol):
    seed(ol, "projA", files=("a.png",))
    admin = auth_headers(client, "rainy")
    ol.users["bob"] = "pw"
    bob = auth_headers(client, "bob", "pw")
    client.post("/api/v2/projects/scan", headers=admin)

    assert client.patch("/api/v2/projects/projA", json={"description": "x"}, headers=bob).status_code == 403
    assert client.patch("/api/v2/projects/projA", json={"description": "x"}, headers=admin).status_code == 200
    assert client.get("/api/v2/projects/projA", headers=bob).json()["description"] == "x"


def test_progress_reflects_task_states(client, auth_headers, ol, db):
    seed(ol, "projA", files=("a.png", "b.png", "c.png"))
    headers = auth_headers(client)
    client.post("/api/v2/projects/scan", headers=headers)

    with db.session_scope() as session:
        tasks = list(session.scalars(select(Task).order_by(Task.rel_path)))
        tasks[0].state = "submitted"
        tasks[1].state = "approved"
        tasks[2].state = "rejected"

    progress = client.get("/api/v2/projects/projA/progress", headers=headers).json()
    assert progress["total"] == 3
    assert (progress["draft"], progress["submitted"], progress["approved"], progress["rejected"]) == (
        0,
        1,
        1,
        1,
    )
    assert progress["finished"] == 1


def test_progress_by_user(client, auth_headers, ol, db):
    seed(ol, "projA", files=("a.png",))
    headers = auth_headers(client)
    client.post("/api/v2/projects/scan", headers=headers)
    with db.session_scope() as session:
        task = session.scalar(select(Task))
        task.claimer = session.scalar(select(User))
        task.state = "approved"

    progress = client.get("/api/v2/projects/projA/progress", params={"by_user": True}, headers=headers).json()
    assert progress["by_user"]["rainy"]["approved"] == 1


def test_unknown_project_is_404(client, auth_headers):
    headers = auth_headers(client)
    assert client.get("/api/v2/projects/nope", headers=headers).status_code == 404
    assert client.get("/api/v2/projects/nope/progress", headers=headers).status_code == 404
    assert client.get("/api/v2/projects/nope/labels", headers=headers).status_code == 404


# endregion


# region labels
def test_label_crud_and_roles(client, auth_headers, ol):
    seed(ol, "projA", files=("a.png",))
    admin = auth_headers(client, "rainy")
    ol.users["bob"] = "pw"
    bob = auth_headers(client, "bob", "pw")
    client.post("/api/v2/projects/scan", headers=admin)

    created = client.post(
        "/api/v2/projects/projA/labels", json={"name": "Root", "color": "#112233"}, headers=admin
    )
    assert created.status_code == 201, created.text
    label = created.json()
    assert (label["name"], label["color"]) == ("Root", "#112233")

    # duplicate names conflict (case-insensitively); annotators may only read
    assert (
        client.post("/api/v2/projects/projA/labels", json={"name": "root"}, headers=admin).status_code == 409
    )
    assert (
        client.post("/api/v2/projects/projA/labels", json={"name": "Shoot"}, headers=bob).status_code == 403
    )
    assert [x["name"] for x in client.get("/api/v2/projects/projA/labels", headers=bob).json()] == ["Root"]

    patched = client.patch(
        f"/api/v2/projects/projA/labels/{label['id']}", json={"color": "#ff0000"}, headers=admin
    )
    assert patched.json()["color"] == "#ff0000"

    archived = client.patch(
        f"/api/v2/projects/projA/labels/{label['id']}", json={"archived": True}, headers=admin
    )
    assert archived.json()["archived"] is True
    assert client.get("/api/v2/projects/projA/labels", headers=admin).json() == []
    listed = client.get(
        "/api/v2/projects/projA/labels", params={"include_archived": True}, headers=admin
    ).json()
    assert len(listed) == 1

    assert client.delete(f"/api/v2/projects/projA/labels/{label['id']}", headers=admin).status_code == 204
    with client.app.state.db.session_scope() as session:
        assert session.scalar(select(Label)) is None


def test_ensure_labels_creates_missing_ones(services, ol):
    seed(ol, "projA", files=("a.png",))
    service(services).scan_and_sync(force=True)
    with services.db.session_scope() as session:
        project_id = session.scalar(select(Project.id))
        labels = service(services).ensure_labels(session, project_id, ["Root", "root", "Shoot", " "])
        assert [label.name for label in labels] == ["Root", "Shoot"]
    with services.db.session_scope() as session:
        assert len(session.scalars(select(Label)).all()) == 2


# endregion
