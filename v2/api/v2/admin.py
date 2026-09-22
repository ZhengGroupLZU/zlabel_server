"""``/api/v2/admin`` — deployment administration (P5).

Everything here needs the global ``admin`` role. The Web UI (or the CLI, or curl)
uses these endpoints to manage accounts, look at storage and — when the server owns
the tree — browse/move files.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile

from v2.api.deps import get_services, require_roles
from v2.core.errors import PayloadTooLarge, ValidationFailed
from v2.core.logging import get_logger
from v2.schemas.auth import UserOut
from v2.services.auth_service import AuthContext
from v2.services.container import Services

logger = get_logger("zlabel.v2.admin")

router = APIRouter(prefix="/admin", tags=["admin"])

AdminAuth = Depends(require_roles("admin"))


def _local_storage(services: Services):
    """The local backend, or a clear error for a backend that manages itself."""
    storage = services.openlist
    if getattr(storage, "kind", "") != "local":
        raise ValidationFailed(
            f"the {getattr(storage, 'kind', 'unknown')!r} backend owns its own file management"
        )
    return storage


# region users
@router.post("/users", response_model=UserOut, status_code=201)
def create_user(
    payload: dict[str, Any],
    _auth: AuthContext = AdminAuth,
    services: Services = Depends(get_services),
) -> UserOut:
    """Create an account (only with the local identity provider)."""
    identity = services.auth.identity
    if getattr(identity, "kind", "") != "local":
        raise ValidationFailed("accounts live in the external identity provider; create them there")
    name = str(payload.get("name") or "").strip()
    password = str(payload.get("password") or "")
    role = str(payload.get("role") or "annotator")
    created = identity.create_user(
        name, password, role=role, email=str(payload.get("email") or ""), admin=role == "admin"
    )
    with services.db.session_scope() as session:
        from v2.db.models import User

        return UserOut.of(session.get(User, created["id"]))


@router.patch("/users/{user_id}", response_model=UserOut)
def update_user(
    user_id: int,
    payload: dict[str, Any],
    auth: AuthContext = AdminAuth,
    services: Services = Depends(get_services),
) -> UserOut:
    """Change a role (sessions are revoked) and/or enable/disable an account."""
    from sqlalchemy import select

    from v2.db.models import User

    role = payload.get("role")
    if role is not None:
        services.auth.set_role(auth, user_id, str(role))
    if "active" in payload:
        with services.db.session_scope() as session:
            user = session.get(User, user_id)
            if user is None:
                raise ValidationFailed(f"unknown user: {user_id}")
            user.active = bool(payload["active"])
            if not user.active:
                services.auth.revoke_all(user_id)
    with services.db.session_scope() as session:
        user = session.scalar(select(User).where(User.id == user_id))
        if user is None:
            raise ValidationFailed(f"unknown user: {user_id}")
        return UserOut.of(user)


@router.post("/users/{user_id}/password", status_code=204)
def set_password(
    user_id: int,
    payload: dict[str, Any],
    _auth: AuthContext = AdminAuth,
    services: Services = Depends(get_services),
) -> Response:
    """Set a password (local identity only)."""
    identity = services.auth.identity
    if getattr(identity, "kind", "") != "local":
        raise ValidationFailed("accounts live in the external identity provider; change it there")
    from v2.db.models import User

    with services.db.session_scope() as session:
        user = session.get(User, user_id)
        if user is None:
            raise ValidationFailed(f"unknown user: {user_id}")
        name = user.name
    if not identity.set_password(name, str(payload.get("password") or "")):
        raise ValidationFailed(f"cannot set the password of {name!r}")
    services.auth.revoke_all(user_id)  # force a fresh login
    return Response(status_code=204)


# endregion


# region storage
@router.get("/storage")
def storage_info(
    _auth: AuthContext = AdminAuth, services: Services = Depends(get_services)
) -> dict[str, Any]:
    """Which backend stores the data, and how much is there."""
    storage = services.openlist
    info: dict[str, Any] = {"backend": getattr(storage, "kind", "unknown"), "root": storage.root}
    if hasattr(storage, "usage"):
        info.update(storage.usage())
        info["path"] = str(storage.root_dir)
    else:
        info["hint"] = "the external backend reports its own usage"
    return info


@router.get("/files")
def list_files(
    path: str = Query("/"),
    _auth: AuthContext = AdminAuth,
    services: Services = Depends(get_services),
) -> dict[str, Any]:
    """One directory: sub-directories and files (local backend only)."""
    storage = _local_storage(services)
    entries = []
    for name in storage.list_dirs(path, ""):
        entries.append({"name": name, "type": "dir"})
    for full in storage.glob_files(path, ""):
        parent, _, name = full.rpartition("/")
        if parent == path.rstrip("/"):
            info = storage.file_info(full, "")
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
    storage = _local_storage(services)
    content = await file.read()
    if len(content) > services.settings.max_upload_bytes:
        raise PayloadTooLarge(f"file exceeds {services.settings.max_upload_bytes} bytes")
    storage.put_bytes(path, content, "")
    return {"path": path, "size": len(content)}


@router.get("/files/download")
def download_file(
    path: str = Query(...),
    _auth: AuthContext = AdminAuth,
    services: Services = Depends(get_services),
) -> Response:
    storage = _local_storage(services)
    content = storage.get_bytes(path, "")
    return Response(content=content, media_type="application/octet-stream")


@router.delete("/files", status_code=204)
def delete_file(
    path: str = Query(...),
    _auth: AuthContext = AdminAuth,
    services: Services = Depends(get_services),
) -> Response:
    _local_storage(services).delete(path)
    return Response(status_code=204)


@router.post("/files/mkdir", status_code=204)
def make_directory(
    path: str = Query(...),
    _auth: AuthContext = AdminAuth,
    services: Services = Depends(get_services),
) -> Response:
    _local_storage(services).ensure_dir(path, "")
    return Response(status_code=204)


@router.post("/files/move", status_code=204)
def move_file(
    source: str = Form(...),
    target: str = Form(...),
    _auth: AuthContext = AdminAuth,
    services: Services = Depends(get_services),
) -> Response:
    """Move/rename inside the storage root (same volume, atomic)."""
    storage = _local_storage(services)
    src = storage.disk_path(source)
    if not src.exists():
        raise ValidationFailed(f"not found: {source}")
    destination = storage.disk_path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    src.replace(destination)
    logger.info(f"admin moved {source!r} -> {target!r}")
    return Response(status_code=204)


# endregion
