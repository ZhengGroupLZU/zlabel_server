"""Scanning is manual only: creating the app must not walk the storage tree.

The API and the admin pages scan when the admin asks for it (``POST
/projects/scan`` / the Refresh buttons); there is no startup or periodic scan.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.adapters.local_disk import LocalDiskBackend
from app.app import create_app
from app.core.config import Settings
from app.db.base import Database
from app.services.container import Services


def _seed_task(root: Path, project: str = "projA") -> None:
    image = root / project / "images" / "D1.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"png-bytes")


def _app(tmp_path):
    settings = Settings(
        database_url=f"sqlite+pysqlite:///{(tmp_path / 'db.sqlite').as_posix()}",
        storage_root=str(tmp_path / "storage"),
        upload_dir=str(tmp_path / "uploads"),
        inference_url="",
    )
    settings.ensure_dirs()
    database = Database(settings.database_url)
    database.create_all()
    services = Services.build(settings, database, storage=LocalDiskBackend(settings))
    return create_app(settings, database, services=services)


def test_startup_does_not_scan(tmp_path):
    """A dataset on disk stays invisible until somebody scans explicitly."""
    _seed_task(Path(tmp_path / "storage"))
    app = _app(tmp_path)

    with TestClient(app) as client:
        assert client.get("/api/v2/health").status_code == 200
        projects = client.get("/api/v2/projects", headers=_admin_headers(client)).json()
    assert projects == []


def test_scan_endpoint_is_the_manual_trigger(tmp_path):
    _seed_task(Path(tmp_path / "storage"))
    app = _app(tmp_path)

    with TestClient(app) as client:
        headers = _admin_headers(client)
        stats = client.post("/api/v2/projects/scan", headers=headers).json()
        assert stats == {"projects": 1, "tasks": 1, "missing": 0, "deactivated": 0, "skipped": 0}
        assert [p["name"] for p in client.get("/api/v2/projects", headers=headers).json()] == ["projA"]


def _admin_headers(client: TestClient) -> dict[str, str]:
    """Bootstrap an admin (the fixture services are built without accounts)."""
    from app.adapters.identity import LocalIdentity

    services = client.app.state.services
    identity: LocalIdentity = services.auth.identity
    if services.auth.find_user("boss") is None:
        identity.create_user("boss", "bosssecret", admin=True)
    resp = client.post("/api/v2/auth/login", json={"username": "boss", "password": "bosssecret"})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['token']}"}
