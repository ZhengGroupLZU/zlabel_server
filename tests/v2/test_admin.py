"""P5: the administration surface (users, storage, files) and its role gate."""

from __future__ import annotations

from pathlib import Path

ADMIN = "/api/v2/admin"


def _add_account(
    client, admin: dict[str, str], name: str = "bob", role: str = "annotator", password: str = "bobsecret"
) -> dict[str, str]:
    """Create a local account through the admin API and log it in."""
    created = client.post(
        f"{ADMIN}/users", json={"name": name, "password": password, "role": role}, headers=admin
    )
    assert created.status_code == 201, created.text
    login = client.post("/api/v2/auth/login", json={"username": name, "password": password})
    assert login.status_code == 200, login.text
    return {"Authorization": f"Bearer {login.json()['token']}"}


def test_only_admins_may_administer(local_client, auth_headers):
    admin = auth_headers(local_client)
    bob = _add_account(local_client, admin)

    assert local_client.get(f"{ADMIN}/storage", headers=bob).status_code == 403
    assert local_client.get(f"{ADMIN}/files", headers=bob).status_code == 403
    assert (
        local_client.post(
            f"{ADMIN}/users", json={"name": "x", "password": "secret123"}, headers=bob
        ).status_code
        == 403
    )
    assert local_client.get(f"{ADMIN}/storage", headers=admin).status_code == 200


def test_admin_creates_and_manages_accounts(local_client, auth_headers):
    admin = auth_headers(local_client)
    created = local_client.post(
        f"{ADMIN}/users",
        json={"name": "carol", "password": "secret123", "role": "reviewer"},
        headers=admin,
    )
    assert created.status_code == 201, created.text
    carol_id = created.json()["id"]
    assert created.json()["role"] == "reviewer"

    # the new account can log in right away
    carol = local_client.post("/api/v2/auth/login", json={"username": "carol", "password": "secret123"})
    assert carol.status_code == 200 and carol.json()["user"]["role"] == "reviewer"

    # a short password is refused (validation from the identity layer)
    assert (
        local_client.post(
            f"{ADMIN}/users", json={"name": "dave", "password": "123"}, headers=admin
        ).status_code
        == 422
    )

    # demote + disable
    updated = local_client.patch(f"{ADMIN}/users/{carol_id}", json={"role": "annotator"}, headers=admin)
    assert updated.status_code == 200 and updated.json()["role"] == "annotator"
    disabled = local_client.patch(f"{ADMIN}/users/{carol_id}", json={"active": False}, headers=admin)
    assert disabled.status_code == 200 and disabled.json()["id"] == carol_id
    assert (
        local_client.post(
            "/api/v2/auth/login", json={"username": "carol", "password": "secret123"}
        ).status_code
        == 401
    )

    # a password reset revokes the old sessions and lets the new one in
    local_client.patch(f"{ADMIN}/users/{carol_id}", json={"active": True}, headers=admin)
    reset = local_client.post(
        f"{ADMIN}/users/{carol_id}/password", json={"password": "brand-new-1"}, headers=admin
    )
    assert reset.status_code == 204
    assert (
        local_client.post(
            "/api/v2/auth/login", json={"username": "carol", "password": "secret123"}
        ).status_code
        == 401
    )
    assert (
        local_client.post(
            "/api/v2/auth/login", json={"username": "carol", "password": "brand-new-1"}
        ).status_code
        == 200
    )


def test_storage_usage_and_file_browsing(local_client, local_settings, auth_headers):
    admin = auth_headers(local_client)
    root = Path(local_settings.storage_root)

    info = local_client.get(f"{ADMIN}/storage", headers=admin).json()
    assert info["backend"] == "local" and info["files"] == 0

    upload = local_client.put(
        f"{ADMIN}/files",
        params={"path": "/projA/images/D1.png"},
        files={"file": ("D1.png", b"png-bytes")},
        headers=admin,
    )
    assert upload.status_code == 200 and upload.json()["size"] == len(b"png-bytes")
    assert (root / "projA" / "images" / "D1.png").is_file()

    listing = local_client.get(f"{ADMIN}/files", params={"path": "/projA/images"}, headers=admin).json()
    assert [e["name"] for e in listing["entries"]] == ["D1.png"]
    directories = local_client.get(f"{ADMIN}/files", params={"path": "/projA"}, headers=admin).json()
    assert [e["name"] for e in directories["entries"]] == ["images"] and directories["entries"][0][
        "type"
    ] == "dir"

    downloaded = local_client.get(
        f"{ADMIN}/files/download", params={"path": "/projA/images/D1.png"}, headers=admin
    )
    assert downloaded.content == b"png-bytes"

    assert (
        local_client.post(f"{ADMIN}/files/mkdir", params={"path": "/projA/notes"}, headers=admin).status_code
        == 204
    )
    moved = local_client.post(
        f"{ADMIN}/files/move",
        data={"source": "/projA/images/D1.png", "target": "/projA/notes/D1.png"},
        headers=admin,
    )
    assert moved.status_code == 204 and (root / "projA" / "notes" / "D1.png").is_file()

    assert (
        local_client.delete(
            f"{ADMIN}/files", params={"path": "/projA/notes/D1.png"}, headers=admin
        ).status_code
        == 204
    )
    usage = local_client.get(f"{ADMIN}/storage", headers=admin).json()
    assert usage["files"] == 0 and usage["bytes"] == 0


def test_file_admin_refuses_escape_and_foreign_backend(local_client, auth_headers, monkeypatch):
    admin = auth_headers(local_client)
    assert local_client.get(f"{ADMIN}/files", params={"path": "/../etc"}, headers=admin).status_code == 422
    assert (
        local_client.get(
            f"{ADMIN}/files/download", params={"path": "/../../etc/passwd"}, headers=admin
        ).status_code
        == 422
    )

    # a backend that manages its own files answers with a clear message
    from v2.core.errors import ValidationFailed

    class _External:
        kind = "openlist"
        root = "/zlabel_server/projects"

    monkeypatch.setattr(local_client.app.state.services, "openlist", _External())
    assert local_client.get(f"{ADMIN}/files", headers=admin).status_code == 422
    assert (
        "owns its own file management" in local_client.get(f"{ADMIN}/files", headers=admin).json()["message"]
    )
    assert local_client.get(f"{ADMIN}/storage", headers=admin).json()["hint"]
    assert ValidationFailed is not None


def test_upload_size_limit(local_client, auth_headers):
    admin = auth_headers(local_client)
    local_client.app.state.services.settings.max_upload_bytes = 4
    resp = local_client.put(
        f"{ADMIN}/files",
        params={"path": "/big.bin"},
        files={"file": ("big.bin", b"too large")},
        headers=admin,
    )
    assert resp.status_code == 413 and resp.json()["code"] == "payload_too_large"
