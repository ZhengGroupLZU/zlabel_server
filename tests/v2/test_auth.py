"""Login, sessions, role gating."""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select

from v2.db.models import ROLE_ADMIN, ROLE_ANNOTATOR, ROLE_REVIEWER, Session, User, utcnow
from v2.services.auth_service import hash_token

LOGIN = "/api/v2/auth/login"
ME = "/api/v2/auth/me"


def test_first_user_becomes_admin(client, auth_headers, db):
    headers = auth_headers(client)
    me = client.get(ME, headers=headers).json()
    assert me["name"] == "rainy" and me["role"] == ROLE_ADMIN

    with db.session_scope() as session:
        user = session.scalar(select(User))
        assert user.name == "rainy" and user.role == ROLE_ADMIN
        assert session.scalar(select(Session)) is not None  # a session row exists


def test_login_rejects_bad_credentials(client):
    resp = client.post(LOGIN, json={"username": "rainy", "password": "nope"})
    assert resp.status_code == 401
    assert resp.json()["code"] == "unauthorized"


def test_me_requires_a_token(client):
    resp = client.get(ME)
    assert resp.status_code == 401
    assert resp.json()["code"] == "unauthorized"


def test_login_twice_issues_independent_sessions(client, auth_headers, db):
    first = auth_headers(client)
    second = auth_headers(client)
    assert first != second
    for headers in (first, second):
        assert client.get(ME, headers=headers).status_code == 200
    with db.session_scope() as session:
        assert len(session.scalars(select(Session)).all()) == 2


def test_logout_revokes_the_session_immediately(client, auth_headers):
    headers = auth_headers(client)
    assert client.post("/api/v2/auth/logout", headers=headers).status_code == 204
    assert client.get(ME, headers=headers).status_code == 401


def test_expired_session_is_rejected(client, auth_headers, db, app):
    headers = auth_headers(client)
    with db.session_scope() as session:
        row = session.scalar(select(Session))
        row.expires_at = utcnow() - timedelta(seconds=1)
    app.state.services.auth.cache.clear()
    assert client.get(ME, headers=headers).status_code == 401


def test_disabled_user_is_rejected(client, auth_headers, db, app):
    headers = auth_headers(client)
    with db.session_scope() as session:
        session.scalar(select(User)).active = False
    app.state.services.auth.cache.clear()
    assert client.get(ME, headers=headers).status_code == 401


def test_second_user_gets_the_annotator_role(client, auth_headers, harness):
    auth_headers(client, "rainy")  # first login creates the admin
    harness.users["bob"] = "pw"
    headers = auth_headers(client, "bob", "pw")
    assert client.get(ME, headers=headers).json()["role"] == ROLE_ANNOTATOR


def test_user_admin_endpoints_are_admin_only(client, auth_headers, harness):
    admin = auth_headers(client, "rainy")
    harness.users["bob"] = "pw"
    bob = auth_headers(client, "bob", "pw")

    assert client.get("/api/v2/auth/users", headers=bob).status_code == 403
    users = client.get("/api/v2/auth/users", headers=admin).json()
    assert {u["name"] for u in users} == {"rainy", "bob"}

    bob_id = next(u["id"] for u in users if u["name"] == "bob")
    promoted = client.put(f"/api/v2/auth/users/{bob_id}/role", params={"role": ROLE_REVIEWER}, headers=admin)
    assert promoted.status_code == 200 and promoted.json()["role"] == ROLE_REVIEWER
    # the role change revokes the old session
    assert client.get(ME, headers=bob).status_code == 401


def test_login_adopts_a_legacy_identity_id(client, auth_headers, db):
    """Rows from the pre-P6 database keep their role and stats.

    Their ``identity_id`` is re-pointed at ``local:<id>`` on the next login (the
    lookup falls back to the name), so an upgrade does not create a second account.
    """
    headers = auth_headers(client)
    assert client.get(ME, headers=headers).status_code == 200
    with db.session_scope() as session:
        user = session.scalar(select(User))
        user.identity_id = "3"  # a legacy, upstream-style value
    login = client.post(LOGIN, json={"username": "rainy", "password": "secret"})
    assert login.status_code == 200
    with db.session_scope() as session:
        user = session.scalar(select(User))
        assert user.identity_id == f"local:{user.id}" and user.name == "rainy"


def test_unknown_role_is_rejected(client, auth_headers):
    admin = auth_headers(client)
    me = client.get(ME, headers=admin).json()
    resp = client.put(f"/api/v2/auth/users/{me['id']}/role", params={"role": "wizard"}, headers=admin)
    assert resp.status_code == 403


def test_resolve_uses_the_cache(client, auth_headers, app):
    headers = auth_headers(client)
    auth = app.state.services.auth
    token = headers["Authorization"].split()[1]
    assert client.get(ME, headers=headers).status_code == 200
    cached = auth.resolve(token)
    assert auth.resolve(token) == cached
    assert hash_token(token) in auth.cache._entries
