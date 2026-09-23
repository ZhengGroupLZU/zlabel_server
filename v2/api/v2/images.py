"""``/api/v2/projects/{project}/images/{rel_path}`` — task images, with HTTP caching.

Reading a task image is side-effect free (v1 used it to warm the model, which is how
predictions ended up running on the wrong embedding).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Header, Response, UploadFile

from v2.api.deps import get_auth, get_services
from v2.core.errors import PayloadTooLarge
from v2.services.auth_service import AuthContext
from v2.services.container import Services
from v2.services.image_store import sha256_bytes

router = APIRouter(tags=["images"])

MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def _media_type(rel_path: str) -> str:
    lowered = rel_path.lower()
    for suffix, media in MEDIA_TYPES.items():
        if lowered.endswith(suffix):
            return media
    return "application/octet-stream"


@router.get("/projects/{project}/images/{rel_path:path}")
def get_image(
    project: str,
    rel_path: str,
    if_none_match: str | None = Header(None),
    _auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> Response:
    """``ETag`` + ``If-None-Match`` friendly; 404 when the task image is gone."""
    services.projects.get_project(project)
    path = services.storage.image_path(project, rel_path)
    info = services.storage.file_info(path)
    headers = {"ETag": info.etag, "Cache-Control": "private, max-age=0, must-revalidate"}
    if if_none_match and if_none_match.strip() == info.etag:
        return Response(status_code=304, headers=headers)
    content = services.storage.get_bytes(path)
    headers["X-Image-Sha256"] = sha256_bytes(content)
    return Response(content=content, media_type=_media_type(rel_path), headers=headers)


@router.put("/projects/{project}/images/{rel_path:path}")
async def upload_image(
    project: str,
    rel_path: str,
    file: UploadFile = File(...),
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> dict:
    """Store a task image the client owns (local dataset + remote inference).

    Returns its ``sha256``: later ``predict`` calls may reference the image by
    that digest instead of re-uploading the bytes.
    """
    services.projects.get_project(project)
    content = await file.read()
    if len(content) > services.settings.max_upload_bytes:
        raise PayloadTooLarge(
            f"image exceeds the {services.settings.max_upload_bytes} byte limit",
            detail={"rel_path": rel_path},
        )
    digest, size = services.images.put(content)
    return {"sha256": digest, "size": size, "rel_path": rel_path, "uploaded_by": auth.name}


@router.get("/images/{digest}")
def get_uploaded_image(
    digest: str,
    _auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> Response:
    """Serve a previously uploaded task image (the inference worker pulls them here)."""
    content = services.images.get(digest)
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={"ETag": f'"{digest[:16]}"', "Cache-Control": "private, max-age=3600"},
    )
