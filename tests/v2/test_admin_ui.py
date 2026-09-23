"""The web admin UI: cookie login, the page set and their write paths.

Menu: Dashboard / Users / Projects / Files / Audit log. Project-scoped data
(files, members, labels, tasks) lives on the project detail page.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from v2.app import create_app
from v2.core.config import Settings
from v2.db.base import Database
from v2.services.container import Services

CSRF_RE = re.compile(r'name="csrftoken" value="([^"]+)"')
PAGES = ("/admin/", "/admin/users", "/admin/projects", "/admin/files", "/admin/audit-log/list")


def csrf_token(client: TestClient, page: str = "/admin/login") -> str:
    match = CSRF_RE.search(client.get(page).text)
    assert match, f"no csrf token on {page}"
    return match.group(1)


def _api_login(client: TestClient, name: str, password: str) -> dict[str, str]:
    resp = client.post("/api/v2/auth/login", json={"username": name, "password": password})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['token']}"}


def _audit(client: TestClient, action: str | None = None) -> list[tuple[str, str, str]]:
    """``(action, target_type, target_id)`` of the audit rows, oldest first."""
    from sqlalchemy import select

    from v2.db.models import AuditLog

    with client.app.state.services.db.session_scope() as session:
        rows = session.scalars(select(AuditLog).order_by(AuditLog.id)).all()
        return [(r.action, r.target_type, r.target_id) for r in rows if action is None or r.action == action]


def _png(size: tuple[int, int] = (32, 32)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _seed_task(client: TestClient, project: str, rel: str = "images/D1.png") -> None:
    """Drop a real PNG into the storage tree and scan it into the task table."""
    root = Path(client.app.state.settings.storage_root) / project / rel
    root.parent.mkdir(parents=True, exist_ok=True)
    root.write_bytes(_png())
    client.app.state.services.projects.scan_and_sync(force=True)


@pytest.fixture
def admin_client(tmp_path) -> TestClient:
    """A server whose admin UI is enabled, with one admin and one annotator."""
    from v2.adapters.local_disk import LocalDiskBackend

    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        storage_root=str(tmp_path / "storage"),
        admin_enabled=True,
        inference_url="",
    )
    settings.ensure_dirs()
    database = Database(settings.database_url)
    database.create_all()
    services = Services.build(settings, database, storage=LocalDiskBackend(settings))
    services.auth.identity.create_user("boss", "bosssecret", admin=True)
    services.auth.identity.create_user("peon", "peonsecret", role="annotator")
    with TestClient(create_app(settings, database, services=services)) as client:
        yield client


def login(client: TestClient, user: str = "boss", password: str = "bosssecret", follow: bool = False):
    token = CSRF_RE.search(client.get("/admin/login").text)
    return client.post(
        "/admin/login",
        data={"username": user, "password": password, "csrftoken": token.group(1) if token else ""},
        follow_redirects=follow,
    )


# region gate
def test_anonymous_visitors_are_sent_to_the_login(admin_client):
    resp = admin_client.get("/admin/", follow_redirects=False)
    assert resp.status_code == 303 and "/admin/login" in resp.headers["location"]


def test_only_admins_may_log_in(admin_client):
    """An annotator's credentials are valid for the API but not for the admin UI."""
    refused = login(admin_client, "peon", "peonsecret")
    assert refused.status_code == 400 and "not an administrator" in refused.text
    assert "zl_admin" not in admin_client.cookies

    wrong = login(admin_client, "boss", "nope")
    assert wrong.status_code == 400 and "invalid credentials" in wrong.text

    ok = login(admin_client)
    assert ok.status_code == 303 and "zl_admin" in admin_client.cookies


# endregion


# region pages
def test_pages_render_html_not_escaped_source(admin_client):
    """Hand-built pages go through index.html, which autoescapes plain strings."""
    login(admin_client)
    for path in PAGES:
        page = admin_client.get(path)
        assert page.status_code == 200, path
        if path == "/admin/audit-log/list":
            continue  # the stock ModelView table is not one of our pages
        assert "&lt;div" not in page.text, path
        assert '<div class="row row-cards">' in page.text, path


def test_dashboard_shows_storage_and_projects(admin_client):
    login(admin_client)
    admin_client.app.state.services.projects.create_project("projA", "Project A", actor_id=1)
    dashboard = admin_client.get("/admin/")
    assert dashboard.status_code == 200
    # our own cards, not starlette-admin's default index
    assert 'text-muted">Projects' in dashboard.text
    assert 'text-muted">Tasks' in dashboard.text
    assert "Storage" in dashboard.text
    assert "Rescan storage" in dashboard.text
    assert "projA" in dashboard.text and "Project A" in dashboard.text


def test_menu_keeps_the_expected_pages(admin_client):
    login(admin_client)
    page = admin_client.get("/admin/").text
    for label in ("Dashboard", "Users", "Projects", "Files", "Audit log"):
        assert f'class="nav-link-title">{label}</span>' in page, label
    # the old per-table pages are gone from the menu
    for removed in ("Tasks", "Labels", "Members", "Storage", "Accounts"):
        assert f'class="nav-link-title">{removed}</span>' not in page, removed


# endregion


# region users
def test_users_page_creates_roles_and_revokes(admin_client):
    """Every Users-page write goes through AuthService: audit + session revoke."""
    login(admin_client)
    token = csrf_token(admin_client, "/admin/users")
    created = admin_client.post(
        "/admin/users/create",
        data={"name": "carol", "password": "secret123", "role": "reviewer", "csrftoken": token},
        follow_redirects=False,
    )
    assert created.status_code == 303
    carol = _api_login(admin_client, "carol", "secret123")
    assert admin_client.get("/api/v2/auth/me", headers=carol).json()["role"] == "reviewer"
    assert _audit(admin_client, "create_user")

    users = admin_client.get("/api/v2/auth/users", headers=_api_login(admin_client, "boss", "bosssecret"))
    carol_id = next(u["id"] for u in users.json() if u["name"] == "carol")
    demoted = admin_client.post(
        "/admin/users/update",
        data={"user_id": str(carol_id), "role": "annotator", "csrftoken": token},
        follow_redirects=False,
    )
    assert demoted.status_code == 303
    assert admin_client.get("/api/v2/auth/me", headers=carol).status_code == 401  # session revoked
    assert _audit(admin_client, "update_user")

    # disabling blocks the next login (ticking "active" re-enables it)
    assert (
        admin_client.post(
            "/admin/users/update",
            data={"user_id": str(carol_id), "role": "annotator", "csrftoken": token},
            follow_redirects=False,
        ).status_code
        == 303
    )
    assert (
        admin_client.post(
            "/api/v2/auth/login", json={"username": "carol", "password": "secret123"}
        ).status_code
        == 401
    )
    admin_client.post(
        "/admin/users/update",
        data={"user_id": str(carol_id), "role": "annotator", "active": "1", "csrftoken": token},
        follow_redirects=False,
    )
    assert (
        admin_client.post(
            "/admin/users/password",
            data={"user_id": str(carol_id), "password": "brand-new-1", "csrftoken": token},
            follow_redirects=False,
        ).status_code
        == 303
    )
    assert (
        admin_client.post(
            "/api/v2/auth/login", json={"username": "carol", "password": "secret123"}
        ).status_code
        == 401
    )
    assert (
        admin_client.post(
            "/api/v2/auth/login", json={"username": "carol", "password": "brand-new-1"}
        ).status_code
        == 200
    )
    assert _audit(admin_client, "set_password")


def test_users_page_rejects_a_short_password(admin_client):
    login(admin_client)
    token = csrf_token(admin_client, "/admin/users")
    resp = admin_client.post(
        "/admin/users/create",
        data={"name": "dave", "password": "123", "role": "annotator", "csrftoken": token},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert "at least" in resp.text  # the ValidationFailed message is flashed
    assert not _audit(admin_client, "create_user")


def test_users_page_lists_no_password_hashes_and_filters(admin_client):
    login(admin_client)
    page = admin_client.get("/admin/users")
    assert page.status_code == 200 and "<code>boss</code>" in page.text and "<code>peon</code>" in page.text
    assert "scrypt$" not in page.text

    filtered = admin_client.get("/admin/users", params={"q": "boss"})
    assert "<code>boss</code>" in filtered.text and "<code>peon</code>" not in filtered.text
    filtered = admin_client.get("/admin/users", params={"role": "annotator"})
    assert "<code>peon</code>" in filtered.text and "<code>boss</code>" not in filtered.text


# endregion


# region projects
def test_projects_page_creates_filters_and_updates(admin_client):
    login(admin_client)
    token = csrf_token(admin_client, "/admin/projects")
    created = admin_client.post(
        "/admin/projects/create",
        data={"name": "projA", "display_name": "Project A", "csrftoken": token},
        follow_redirects=False,
    )
    assert created.status_code == 303
    root = Path(admin_client.app.state.settings.storage_root)
    assert (root / "projA").is_dir()
    assert _audit(admin_client, "create_project")
    # the create form's timeline box was not submitted -> flat project
    assert admin_client.app.state.services.projects.get_project("projA", active_only=False).timeline is False

    # a duplicate name is refused and flashed, not a 500
    again = admin_client.post(
        "/admin/projects/create",
        data={"name": "projA", "csrftoken": token},
        follow_redirects=True,
    )
    assert again.status_code == 200 and "already exists" in again.text

    # filters: an inactive project only shows up when asked for
    admin_client.app.state.services.projects.update_project("projA", active=False, actor_id=1)
    assert "<code>projA</code>" not in admin_client.get("/admin/projects", params={"active": "1"}).text
    assert "<code>projA</code>" in admin_client.get("/admin/projects", params={"active": "0"}).text
    admin_client.app.state.services.projects.update_project("projA", active=True, actor_id=1)

    # rename = the display name (the directory name is the project identity)
    updated = admin_client.post(
        "/admin/projects/update",
        data={
            "project": "projA",
            "display_name": "Renamed",
            "description": "hello",
            "active": "1",
            "csrftoken": token,
        },
        follow_redirects=False,
    )
    assert updated.status_code == 303
    project = admin_client.app.state.services.projects.get_project("projA", active_only=False)
    assert project.display_name == "Renamed" and project.description == "hello"
    assert project.timeline is False
    assert _audit(admin_client, "update_project")

    # turning the timeline on parses the existing tasks right away
    rel = "sp/dish/D1.png"
    image = Path(admin_client.app.state.settings.storage_root) / "projA" / rel
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"png")
    admin_client.post("/admin/projects/scan", data={"csrftoken": token}, follow_redirects=False)
    from sqlalchemy import select

    from v2.db.models import Task

    with admin_client.app.state.services.db.session_scope() as session:
        task = session.scalar(select(Task).where(Task.rel_path == rel))
        assert (task.group_name, task.day) == ("", 0)  # timeline off: nothing parsed
    toggled = admin_client.post(
        "/admin/projects/update",
        data={
            "project": "projA",
            "display_name": "Renamed",
            "active": "1",
            "timeline": "1",
            "csrftoken": token,
        },
        follow_redirects=False,
    )
    assert toggled.status_code == 303
    with admin_client.app.state.services.db.session_scope() as session:
        task = session.scalar(select(Task).where(Task.rel_path == rel))
        assert (task.group_name, task.day) == ("sp/dish", 1)


def test_project_detail_tabs_render(admin_client):
    login(admin_client)
    admin_client.app.state.services.projects.create_project("projA", "Project A", actor_id=1)
    _seed_task(admin_client, "projA")
    for tab in ("overview", "files", "members", "labels", "tasks", "instances"):
        page = admin_client.get("/admin/projects", params={"project": "projA", "tab": tab})
        assert page.status_code == 200, tab
    assert "images" in admin_client.get("/admin/projects", params={"project": "projA", "tab": "files"}).text
    assert (
        "images/D1.png"
        in admin_client.get(
            "/admin/projects", params={"project": "projA", "tab": "files", "rel": "images"}
        ).text
    )
    assert (
        "images/D1.png"
        in admin_client.get("/admin/projects", params={"project": "projA", "tab": "tasks"}).text
    )
    # an unknown project goes back to the list with a flash, not a 500
    missing = admin_client.get("/admin/projects", params={"project": "nope"})
    assert missing.status_code == 200 and "unknown project" in missing.text


def test_project_files_upload_preview_download_delete(admin_client):
    login(admin_client)
    admin_client.app.state.services.projects.create_project("projA", actor_id=1)
    token = csrf_token(admin_client, "/admin/projects?project=projA&tab=files")
    root = Path(admin_client.app.state.settings.storage_root) / "projA"

    upload = admin_client.post(
        "/admin/projects/upload",
        data={"project": "projA", "rel": "", "csrftoken": token},
        files={"file": ("D1.png", _png())},
        follow_redirects=False,
    )
    assert upload.status_code == 303 and (root / "D1.png").is_file()

    listing = admin_client.get("/admin/projects", params={"project": "projA", "tab": "files"})
    assert "D1.png" in listing.text and "Preview" in listing.text

    preview = admin_client.get("/admin/projects/preview", params={"project": "projA", "rel": "D1.png"})
    assert preview.status_code == 200 and preview.headers["content-type"] == "image/png"
    thumb = admin_client.get(
        "/admin/projects/preview", params={"project": "projA", "rel": "D1.png", "thumb": "1"}
    )
    assert thumb.status_code == 200 and thumb.content.startswith(b"\x89PNG")

    download = admin_client.get("/admin/projects/download", params={"project": "projA", "rel": "D1.png"})
    assert download.status_code == 200 and download.content == (root / "D1.png").read_bytes()

    assert (
        admin_client.post(
            "/admin/projects/mkdir",
            data={"project": "projA", "rel": "", "name": "images", "csrftoken": token},
            follow_redirects=False,
        ).status_code
        == 303
    )
    assert (root / "images").is_dir()

    deleted = admin_client.post(
        "/admin/projects/delete-file",
        data={"project": "projA", "rel": "D1.png", "csrftoken": token},
        follow_redirects=False,
    )
    assert deleted.status_code == 303 and not (root / "D1.png").exists()


def test_project_members_and_labels(admin_client):
    """Project-scoped writes go through ProjectService (audited)."""
    login(admin_client)
    admin_client.app.state.services.projects.create_project("projA", actor_id=1)
    token = csrf_token(admin_client, "/admin/projects?project=projA&tab=members")

    added = admin_client.post(
        "/admin/projects/member/save",
        data={"project": "projA", "user_id": "2", "role": "reviewer", "csrftoken": token},
        follow_redirects=False,
    )
    assert added.status_code == 303
    assert _audit(admin_client, "add_member")
    members = admin_client.get("/admin/projects", params={"project": "projA", "tab": "members"}).text
    assert "peon" in members

    removed = admin_client.post(
        "/admin/projects/member/delete",
        data={"project": "projA", "user_id": "2", "csrftoken": token},
        follow_redirects=False,
    )
    assert removed.status_code == 303 and _audit(admin_client, "remove_member")

    created = admin_client.post(
        "/admin/projects/label/save",
        data={"project": "projA", "name": "Root", "color": "#112233", "sort": "0", "csrftoken": token},
        follow_redirects=False,
    )
    assert created.status_code == 303 and _audit(admin_client, "create_label")
    labels = admin_client.get("/admin/projects", params={"project": "projA", "tab": "labels"}).text
    assert "Root" in labels and "#112233" in labels

    label_id = admin_client.app.state.services.projects.list_labels("projA")[0].id
    updated = admin_client.post(
        "/admin/projects/label/save",
        data={
            "project": "projA",
            "label_id": str(label_id),
            "name": "Shoot",
            "color": "#445566",
            "sort": "1",
            "archived": "1",
            "csrftoken": token,
        },
        follow_redirects=False,
    )
    assert updated.status_code == 303 and _audit(admin_client, "update_label")
    assert (
        admin_client.app.state.services.projects.list_labels("projA", include_archived=True)[0].name
        == "Shoot"
    )

    assert (
        admin_client.post(
            "/admin/projects/label/delete",
            data={"project": "projA", "label_id": str(label_id), "csrftoken": token},
            follow_redirects=False,
        ).status_code
        == 303
    )
    assert _audit(admin_client, "delete_label")


def test_label_form_prefills_an_unused_palette_color(admin_client):
    """The Add-label form starts from a palette colour instead of black."""
    from v2.services.label_palette import LABEL_PALETTE

    login(admin_client)
    admin_client.app.state.services.projects.create_project("projA", actor_id=1)
    page = admin_client.get("/admin/projects", params={"project": "projA", "tab": "labels"})
    match = re.search(r'name="color"[^>]*value="(#[0-9a-fA-F]{6})"', page.text)
    assert match, "no colour input on the labels tab"
    assert match.group(1) in LABEL_PALETTE

    token = csrf_token(admin_client, "/admin/projects?project=projA&tab=labels")
    created = admin_client.post(
        "/admin/projects/label/save",
        data={"project": "projA", "name": "Root", "color": match.group(1), "csrftoken": token},
        follow_redirects=False,
    )
    assert created.status_code == 303
    assert admin_client.app.state.services.projects.list_labels("projA")[0].color == match.group(1)


def test_labels_tab_saves_everything_with_one_button(admin_client):
    """The id column is the 0-based position; one Save button persists the table."""
    login(admin_client)
    services = admin_client.app.state.services
    services.projects.create_project("projA", actor_id=1)
    for name in ("Seed", "Root", "Shoot"):
        services.projects.create_label("projA", name, actor_id=1)

    token = csrf_token(admin_client, "/admin/projects?project=projA&tab=labels")
    page = admin_client.get("/admin/projects", params={"project": "projA", "tab": "labels"})
    # one form wraps the table, one button saves the whole page
    assert 'id="label-bulk-form"' in page.text and 'id="label-save-all"' in page.text
    assert "label-order-value" in page.text and "Save all labels" in page.text
    assert 'name="name-1"' in page.text and 'name="color-1"' in page.text
    assert "archived-1" in page.text
    # per-row delete buttons submit their own (out-of-form) forms
    assert 'form="label-delete-1"' in page.text
    assert page.text.count('draggable="true"') == 3
    # only the ⠿ handle (first column) is draggable: the row itself is not
    assert "<tr draggable" not in page.text
    assert page.text.index("⠿</td>") < page.text.index('<td class="text-muted">0</td>')
    assert re.findall(r'<td class="text-muted">(\d+)</td>', page.text) == ["0", "1", "2"]
    # the drag script must be emitted after the table (a parser-time lookup of an
    # element below it returns null and the drag would silently do nothing) and it
    # has to set dropEffect, or the browser keeps the "no-drop" cursor
    assert page.text.index('id="label-table"') < page.text.index("addEventListener('drop'")
    assert "dropEffect = 'move'" in page.text and "setData(" in page.text
    assert "syncOrder" in page.text  # the drop updates the hidden order field

    ids = [label.id for label in services.projects.list_labels("projA")]
    saved = admin_client.post(
        "/admin/projects/label/save-all",
        data={
            "project": "projA",
            "order": ",".join(str(i) for i in reversed(ids)),
            f"name-{ids[0]}": "Seeds",  # Seed was renamed
            f"color-{ids[0]}": "#abc",  # shorthand hex typed by hand
            f"archived-{ids[1]}": "1",  # Root is archived
            f"name-{ids[2]}": "Shoot",
            "csrftoken": token,
        },
        follow_redirects=False,
    )
    assert saved.status_code == 303
    labels = services.projects.list_labels("projA", include_archived=True)
    assert [label.name for label in labels] == ["Shoot", "Root", "Seeds"]
    assert [label.sort for label in labels] == [0, 1, 2]
    assert [label.id for label in labels] == [ids[2], ids[1], ids[0]]  # primary keys untouched
    by_name = {label.name: label for label in labels}
    assert by_name["Seeds"].color == "#aabbcc" and by_name["Root"].archived is True
    assert _audit(admin_client, "update_label") and _audit(admin_client, "reorder_labels")

    # an incomplete order is refused with a flash, not a 500
    bad = admin_client.post(
        "/admin/projects/label/save-all",
        data={"project": "projA", "order": str(ids[0]), "csrftoken": token},
        follow_redirects=True,
    )
    assert bad.status_code == 200 and "exactly once" in bad.text

    # without an order (no JS) the fields still save, the order stays
    no_js = admin_client.post(
        "/admin/projects/label/save-all",
        data={"project": "projA", f"name-{ids[2]}": "Shoots", "csrftoken": token},
        follow_redirects=True,
    )
    assert no_js.status_code == 200
    assert [label.name for label in services.projects.list_labels("projA", include_archived=True)] == [
        "Shoots",
        "Root",
        "Seeds",
    ]


def test_refresh_buttons_post_to_real_urls(admin_client):
    """The Rescan buttons must not produce ``//`` in their action (used to 404)."""
    login(admin_client)
    services = admin_client.app.state.services
    services.projects.create_project("projA", actor_id=1)

    dashboard = admin_client.get("/admin/").text
    assert 'action="/admin/scan"' in dashboard
    assert "/admin//scan" not in dashboard

    listing = admin_client.get("/admin/projects").text
    assert 'action="/admin/projects/scan"' in listing
    assert "Refresh projects" in listing and "manual only" in listing
    assert "/admin//" not in listing

    # the refresh button really scans: a task dropped in afterwards is picked up
    root = Path(admin_client.app.state.settings.storage_root) / "projA" / "images"
    root.mkdir(parents=True, exist_ok=True)
    (root / "D2.png").write_bytes(b"png-bytes")
    token = csrf_token(admin_client, "/admin/projects")
    resp = admin_client.post("/admin/projects/scan", data={"csrftoken": token}, follow_redirects=False)
    assert resp.status_code == 303
    from sqlalchemy import select

    from v2.db.models import Task

    with services.db.session_scope() as session:
        assert session.scalar(select(Task).where(Task.rel_path == "images/D2.png")) is not None


def test_project_delete_needs_the_typed_name(admin_client):
    """Deleting is irreversible: the admin has to type the project name."""
    login(admin_client)
    services = admin_client.app.state.services
    services.projects.create_project("projA", actor_id=1)
    _seed_task(admin_client, "projA")

    page = admin_client.get("/admin/projects", params={"project": "projA", "tab": "overview"})
    assert 'id="danger-zone"' in page.text and "Delete project" in page.text
    token = csrf_token(admin_client, "/admin/projects?project=projA&tab=overview")

    wrong = admin_client.post(
        "/admin/projects/delete",
        data={"project": "projA", "confirm": "proj", "csrftoken": token},
        follow_redirects=True,
    )
    assert wrong.status_code == 200 and "exactly" in wrong.text
    services.projects.get_project("projA", active_only=False)  # still registered

    deleted = admin_client.post(
        "/admin/projects/delete",
        data={"project": "projA", "confirm": "projA", "delete_files": "1", "csrftoken": token},
        follow_redirects=True,
    )
    assert deleted.status_code == 200 and "deleted" in deleted.text
    from sqlalchemy import select

    from v2.db.models import Project

    root = Path(admin_client.app.state.settings.storage_root)
    assert not (root / "projA").exists()
    with services.db.session_scope() as session:
        assert session.scalar(select(Project)) is None
    assert _audit(admin_client, "delete_project")
    assert "<code>projA</code>" not in admin_client.get("/admin/projects").text


def test_label_color_picker_offers_swatches_and_hex(admin_client):
    """The colour control is swatches + a hex field (typed codes are normalised)."""
    from v2.services.label_palette import LABEL_PALETTE

    login(admin_client)
    services = admin_client.app.state.services
    services.projects.create_project("projA", actor_id=1)
    services.projects.create_label("projA", "Seed", actor_id=1)

    page = admin_client.get("/admin/projects", params={"project": "projA", "tab": "labels"})
    # one palette dropdown for the row editor and one for the add form; every entry
    # carries its colour dot and the hex code
    assert page.text.count("data-color-swatch data-color=") == len(LABEL_PALETTE) * 2
    assert page.text.count('class="zl-color-dot"') == len(LABEL_PALETTE) * 2 + 2  # items + toggles
    assert page.text.count("data-zl-color-menu hidden") == 2
    assert "<code>#e6194b</code>" in page.text  # hex shown next to each swatch
    assert page.text.count("zl-color-item active") == 2  # each picker marks its colour
    assert 'class="form-control form-control-sm zl-hex"' in page.text
    assert 'pattern="#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})"' in page.text
    assert "data-preview-for=" in page.text and "zl-preview" in page.text
    assert (
        "dropEffect" not in page.text.split("data-zl-color-menu")[0][-200:]
    )  # sanity: colour JS is separate
    assert "addEventListener('input'" in page.text and "data-zl-color-toggle" in page.text

    token = csrf_token(admin_client, "/admin/projects?project=projA&tab=labels")
    typed = admin_client.post(
        "/admin/projects/label/save",
        data={"project": "projA", "name": "Root", "color": "#abc", "csrftoken": token},
        follow_redirects=False,
    )
    assert typed.status_code == 303
    colors = {label.name: label.color for label in services.projects.list_labels("projA")}
    assert colors["Root"] == "#aabbcc"  # shorthand expanded by the server too


def test_instance_tab_lists_saves_creates_and_deletes(admin_client):
    """A project-scoped registry like Labels: the number is read-only metadata id."""
    login(admin_client)
    services = admin_client.app.state.services
    services.projects.create_project("projA", actor_id=1)
    services.instances.create_instance("projA", number=1, name="seed-1", status="normal_seed")
    services.instances.create_instance("projA", number=2)

    page = admin_client.get("/admin/projects", params={"project": "projA", "tab": "instances"})
    assert page.status_code == 200
    assert 'id="instance-bulk-form"' in page.text and 'id="instance-save-all"' in page.text
    assert 'name="name-1"' in page.text and 'name="status-2"' in page.text
    assert 'name="note-1"' in page.text and "instance-statuses" in page.text  # datalist
    assert 'form="instance-delete-1"' in page.text
    assert 'name="number-1"' not in page.text  # the id is not editable
    assert "<code>results</code>" in page.text or "results" in page.text

    token = csrf_token(admin_client, "/admin/projects?project=projA&tab=instances")
    saved = admin_client.post(
        "/admin/projects/instance/save-all",
        data={
            "project": "projA",
            "name-1": "seed-one",
            "status-1": "dead_seed",
            "color-1": "#abc",
            "archived-1": "1",
            "note-1": "hello",
            "name-2": "",
            "status-2": "moldy_seed",
            "csrftoken": token,
        },
        follow_redirects=False,
    )
    assert saved.status_code == 303
    rows = {row["number"]: row for row in services.instances.list_instances("projA", include_archived=True)}
    assert rows[1]["name"] == "seed-one" and rows[1]["status"] == "dead_seed"
    assert rows[1]["color"] == "#aabbcc" and rows[1]["note"] == "hello"
    assert rows[1]["archived"] is True and rows[2]["status"] == "moldy_seed"
    assert _audit(admin_client, "update_instance")

    created = admin_client.post(
        "/admin/projects/instance/create",
        data={"project": "projA", "name": "seed-3", "status": "normal_seedling", "csrftoken": token},
        follow_redirects=False,
    )
    assert created.status_code == 303
    assert [row["number"] for row in services.instances.list_instances("projA", include_archived=True)] == [
        1,
        2,
        3,
    ]

    deleted = admin_client.post(
        "/admin/projects/instance/delete",
        data={"project": "projA", "number": "3", "csrftoken": token},
        follow_redirects=False,
    )
    assert deleted.status_code == 303
    assert [row["number"] for row in services.instances.list_instances("projA", include_archived=True)] == [
        1,
        2,
    ]
    assert _audit(admin_client, "create_instance") and _audit(admin_client, "delete_instance")

    # a duplicate number is flashed, not a 500
    duplicate = admin_client.post(
        "/admin/projects/instance/create",
        data={"project": "projA", "number": "1", "csrftoken": token},
        follow_redirects=True,
    )
    assert duplicate.status_code == 200 and "already exists" in duplicate.text


def test_project_tasks_filter(admin_client):
    login(admin_client)
    admin_client.app.state.services.projects.create_project("projA", actor_id=1)
    _seed_task(admin_client, "projA")

    tasks = admin_client.get("/admin/projects", params={"project": "projA", "tab": "tasks"})
    assert "images/D1.png" in tasks.text and "draft" in tasks.text
    approved = admin_client.get(
        "/admin/projects", params={"project": "projA", "tab": "tasks", "state": "approved"}
    )
    assert "no tasks match" in approved.text


# endregion


# region files (global)
def test_files_page_uploads_downloads_and_deletes(admin_client):
    login(admin_client)
    token = csrf_token(admin_client, "/admin/files")
    assert (
        admin_client.post(
            "/admin/files/mkdir",
            data={"path": "/", "name": "projA", "csrftoken": token},
            follow_redirects=False,
        ).status_code
        == 303
    )
    uploaded = admin_client.post(
        "/admin/files/upload",
        data={"path": "/projA", "csrftoken": token},
        files={"file": ("D1.png", _png())},
        follow_redirects=False,
    )
    assert uploaded.status_code == 303
    listing = admin_client.get("/admin/files", params={"path": "/projA"})
    assert "D1.png" in listing.text and "Preview" in listing.text

    preview = admin_client.get("/admin/files/preview", params={"path": "/projA/D1.png"})
    assert preview.status_code == 200 and preview.headers["content-type"] == "image/png"
    download = admin_client.get("/admin/files/download", params={"path": "/projA/D1.png"})
    assert download.status_code == 200 and download.content.startswith(b"\x89PNG")

    deleted = admin_client.post(
        "/admin/files/delete",
        data={"path": "/projA/D1.png", "csrftoken": token},
        follow_redirects=False,
    )
    assert deleted.status_code == 303
    gone = admin_client.get("/admin/files/download", params={"path": "/projA/D1.png"}, follow_redirects=False)
    assert gone.status_code == 303  # not found: flashed and sent back to the listing


def test_file_paths_cannot_escape_the_storage_root(admin_client):
    login(admin_client)
    escaped = admin_client.get("/admin/files", params={"path": "/../etc"}, follow_redirects=False)
    assert escaped.status_code == 303  # refused and flashed, not served
    assert (
        admin_client.get(
            "/admin/files/preview", params={"path": "/../../etc/passwd"}, follow_redirects=False
        ).status_code
        == 404
    )


# endregion


# region misc
def test_logout_clears_the_cookie(admin_client):
    login(admin_client)
    token = csrf_token(admin_client, "/admin/")
    out = admin_client.post("/admin/logout", data={"csrftoken": token}, follow_redirects=False)
    assert out.status_code == 303
    assert "zl_admin" not in admin_client.cookies
    assert admin_client.get("/admin/", follow_redirects=False).status_code == 303


def test_admin_ui_can_be_disabled(tmp_path):
    from v2.adapters.local_disk import LocalDiskBackend

    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        storage_root=str(tmp_path / "storage"),
        admin_enabled=False,
    )
    settings.ensure_dirs()
    database = Database(settings.database_url)
    database.create_all()
    services = Services.build(settings, database, storage=LocalDiskBackend(settings))
    with TestClient(create_app(settings, database, services=services)) as client:
        assert client.get("/admin/").status_code == 404
        # the API is unaffected
        assert client.get("/api/v2/health").status_code == 200


# endregion
