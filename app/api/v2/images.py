"""``/api/v2/projects/{project}/images/{rel_path}`` — task images, with HTTP caching.

Reading a task image is side-effect free (v1 used it to warm the model, which is how
predictions ended up running on the wrong embedding).

``?max_side=`` asks for a downscaled copy (the desktop draws the task on screen and
has no use for a 20 MB original); ``max_side=0`` or no parameter at all serves the
stored file. Only the *display* path is ever scaled: ``POST /predict`` reads the
stored file itself, so inference keeps working on full-resolution pixels.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Header, Query, Response, UploadFile

from app.api.deps import get_auth, get_services
from app.core.errors import PayloadTooLarge
from app.services.auth_service import AuthContext
from app.services.container import Services
from app.services.image_store import sha256_bytes

router = APIRouter(tags=["images"])

MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def _media_type(rel_path: str) -> str:
    lowered = rel_path.lower()
    for suffix, media in MEDIA_TYPES.items():
        if lowered.endswith(suffix):
            return media
    return "application/octet-stream"


def _variant_etag(info_etag: str, max_side: int) -> str:
    """The ETag of the *requested variant*, not of the bytes we happen to send.

    Deriving it without decoding keeps ``If-None-Match`` cheap (a 304 must not run
    the scaler). An image already within ``max_side`` is served byte-for-byte but
    still answers with this ETag, so the client's per-resolution cache stays correct
    when the requested size changes.
    """
    sized = info_etag.rstrip('"')
    return f'"{sized}-m{max_side}"'


@router.get("/projects/{project}/images/{rel_path:path}")
def get_image(
    project: str,
    rel_path: str,
    max_side: int = Query(0, ge=0, description="long-edge limit for a downscaled copy; 0 = stored file"),
    if_none_match: str | None = Header(None),
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> Response:
    """``ETag`` + ``If-None-Match`` friendly; 404 when the task image is gone.

    With ``max_side`` the response carries the stored file's size in
    ``X-Image-Original-Size`` (``WxH``): the client maps its full-resolution
    annotation coordinates onto the smaller image with it. ``X-Image-Sha256`` is
    always the digest of the **stored** file, so it keeps identifying the original.
    """
    services.projects.get_project(project, auth=auth)
    path = services.storage.image_path(project, rel_path)
    info = services.storage.file_info(path)
    wanted = services.scaler.clamp(max_side)
    etag = _variant_etag(info.etag, wanted) if wanted else info.etag
    headers = {"ETag": etag, "Cache-Control": "private, max-age=0, must-revalidate"}
    if if_none_match and if_none_match.strip() == etag:
        return Response(status_code=304, headers=headers)
    content = services.storage.get_bytes(path)
    headers["X-Image-Sha256"] = sha256_bytes(content)
    if wanted:
        variant = services.scaler.variant(content, max_side=wanted, etag=info.etag)
        if variant is not None:
            headers["X-Image-Original-Size"] = f"{variant.original_size[0]}x{variant.original_size[1]}"
            return Response(content=variant.content, media_type=variant.media_type, headers=headers)
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
    services.projects.get_project(project, auth=auth)
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
