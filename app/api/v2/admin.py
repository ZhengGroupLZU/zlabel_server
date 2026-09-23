"""``/api/v2/admin`` — deployment administration (P5).

Everything here needs the global ``admin`` role. The Web UI (or the CLI, or curl)
uses these endpoints to manage accounts, look at storage and browse/move files.

Account operations are delegated to :class:`~app.services.auth_service.AuthService`
so the REST path, the CLI and the admin UI share the same audit rows and the same
session revocation rules.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile

from app.api.deps import get_services, require_roles
from app.core.errors import PayloadTooLarge, ValidationFailed
from app.core.logging import get_logger
from app.schemas.auth import UserOut
from app.services.auth_service import AuthContext
from app.services.container import Services

logger = get_logger("zlabel.app.admin")

router = APIRouter(prefix="/admin", tags=["admin"])

AdminAuth = Depends(require_roles("admin"))


# region users
@router.post("/users", response_model=UserOut, status_code=201)
def create_user(
    payload: dict[str, Any],
    auth: AuthContext = AdminAuth,
    services: Services = Depends(get_services),
) -> UserOut:
    """Create an account."""
    user = services.auth.create_user(
        str(payload.get("name") or "").strip(),
        str(payload.get("password") or ""),
        role=str(payload.get("role") or "annotator"),
        email=str(payload.get("email") or ""),
        actor_id=auth.user_id,
    )
    return UserOut.of(user)


@router.patch("/users/{user_id}", response_model=UserOut)
def update_user(
    user_id: int,
    payload: dict[str, Any],
    auth: AuthContext = AdminAuth,
    services: Services = Depends(get_services),
) -> UserOut:
    """Change a role and/or enable/disable an account (both revoke its sessions)."""
    role = payload.get("role")
    active = payload.get("active")
    if role is None and active is None:
        raise ValidationFailed("nothing to update: send 'role' and/or 'active'")
    user = services.auth.update_user(
        user_id,
        role=str(role) if role is not None else None,
        active=bool(active) if active is not None else None,
        actor_id=auth.user_id,
    )
    return UserOut.of(user)


@router.post("/users/{user_id}/password", status_code=204)
def set_password(
    user_id: int,
    payload: dict[str, Any],
    auth: AuthContext = AdminAuth,
    services: Services = Depends(get_services),
) -> Response:
    """Set a password (revokes the account's sessions, forcing a fresh login)."""
    services.auth.set_password(user_id, str(payload.get("password") or ""), actor_id=auth.user_id)
    return Response(status_code=204)


# endregion


# region storage
@router.get("/storage")
def storage_info(
    _auth: AuthContext = AdminAuth, services: Services = Depends(get_services)
) -> dict[str, Any]:
    """Where the data lives and how much of it there is."""
    storage = services.storage
    info: dict[str, Any] = {"backend": storage.kind, "root": storage.root}
    info.update(storage.usage())
    info["path"] = str(storage.root_dir)
    return info


@router.get("/files")
def list_files(
    path: str = Query("/"),
    _auth: AuthContext = AdminAuth,
    services: Services = Depends(get_services),
) -> dict[str, Any]:
    """One directory: sub-directories and files."""
    storage = services.storage
    entries = []
    for name in storage.list_dirs(path):
        entries.append({"name": name, "type": "dir"})
    for full in storage.glob_files(path):
        parent, _, name = full.rpartition("/")
        if parent == path.rstrip("/"):
            info = storage.file_info(full)
            entries.append({"name": name, "type": "file", "size": info.size, "modified": info.modified})
    return {"path": path, "entries": sorted(entries, key=lambda e: (e["type"] != "dir", e["name"]))}


@router.put("/files")
async def upload_file(
    path: str = Query(...),
    file: UploadFile = File(...),
    _auth: AuthContext = AdminAuth,
    services: Services = Depends(get_services),
) -> dict[str, Any]:
    """Store a file at ``path`` (the Web UI's upload)."""
    content = await file.read()
    if len(content) > services.settings.max_upload_bytes:
        raise PayloadTooLarge(f"file exceeds {services.settings.max_upload_bytes} bytes")
    services.storage.put_bytes(path, content)
    return {"path": path, "size": len(content)}


@router.get("/files/download")
def download_file(
    path: str = Query(...),
    _auth: AuthContext = AdminAuth,
    services: Services = Depends(get_services),
) -> Response:
    content = services.storage.get_bytes(path)
    return Response(content=content, media_type="application/octet-stream")


@router.delete("/files", status_code=204)
def delete_file(
    path: str = Query(...),
    _auth: AuthContext = AdminAuth,
    services: Services = Depends(get_services),
) -> Response:
    services.storage.delete(path)
    return Response(status_code=204)


@router.post("/files/mkdir", status_code=204)
def make_directory(
    path: str = Query(...),
    _auth: AuthContext = AdminAuth,
    services: Services = Depends(get_services),
) -> Response:
    services.storage.ensure_dir(path)
    return Response(status_code=204)


@router.post("/files/move", status_code=204)
def move_file(
    source: str = Form(...),
    target: str = Form(...),
    _auth: AuthContext = AdminAuth,
    services: Services = Depends(get_services),
) -> Response:
    """Move/rename inside the storage root (same volume, atomic)."""
    storage = services.storage
    src = storage.disk_path(source)
    if not src.exists():
        raise ValidationFailed(f"not found: {source}")
    destination = storage.disk_path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    src.replace(destination)
    logger.info(f"admin moved {source!r} -> {target!r}")
    return Response(status_code=204)


# endregion
