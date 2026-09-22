"""The web admin UI: cookie login, role gate, pages and their guards."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from v2.app import create_app
from v2.core.config import Settings
from v2.db.base import Database
from v2.services.container import Services

CSRF_RE = re.compile(r'name="csrftoken" value="([^"]+)"')
LIST_PAGES = (
    "/admin/user/list",
    "/admin/project/list",
    "/admin/project-member/list",
    "/admin/label/list",
    "/admin/task/list",
    "/admin/audit-log/list",
)


def csrf_token(client: TestClient, page: str = "/admin/login") -> str:
    match = CSRF_RE.search(client.get(page).text)
    assert match, f"no csrf token on {page}"
    return match.group(1)


@pytest.fixture
def admin_client(tmp_path) -> TestClient:
    """A server whose admin UI is enabled, with one admin and one annotator."""
    from v2.adapters.local_disk import LocalDiskBackend

    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        storage_backend="local",
        storage_root=str(tmp_path / "storage"),
        identity="local",
        admin_enabled=True,
        oplist_host="",
        inference_url="",
    )
    settings.ensure_dirs()
    database = Database(settings.database_url)
    database.create_all()
    services = Services.build(settings, database, openlist=LocalDiskBackend(settings), identity="local")
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


def test_dashboard_and_model_pages(admin_client):
    login(admin_client)
    dashboard = admin_client.get("/admin/")
    assert dashboard.status_code == 200
    # our own cards, not starlette-admin's default index
    assert 'text-muted">Projects' in dashboard.text
    assert 'text-muted">Frames' in dashboard.text
    assert "Storage" in dashboard.text

    for path in LIST_PAGES:
        assert admin_client.get(path).status_code == 200, path


def test_accounts_page_never_exposes_password_hashes(admin_client):
    login(admin_client)
    page = admin_client.get("/admin/user/list")
    assert "boss" in page.text and "peon" in page.text
    assert "scrypt$" not in page.text
    # creating accounts here would skip hashing: the view must not offer it
    assert "<form" not in page.text.split("</nav>")[-1] or "password_hash" not in page.text


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
        storage_backend="local",
        storage_root=str(tmp_path / "storage"),
        identity="local",
        admin_enabled=False,
    )
    settings.ensure_dirs()
    database = Database(settings.database_url)
    database.create_all()
    services = Services.build(settings, database, openlist=LocalDiskBackend(settings), identity="local")
    with TestClient(create_app(settings, database, services=services)) as client:
        assert client.get("/admin/").status_code == 404
        # the API is unaffected
        assert client.get("/api/v2/health").status_code == 200
