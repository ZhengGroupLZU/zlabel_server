"""``POST /api/v2/projects/{project}/predict`` — stateless segmentation.

Same wire shape as v1 (``data`` form field with the annotation JSON, optional
image upload) so the desktop's existing response parsing keeps working, but the
model lives in its own process and every request carries the task image it applies to.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import get_auth, get_services
from app.core.errors import ValidationFailed
from app.core.logging import get_logger
from app.services.auth_service import AuthContext
from app.services.container import Services
from app.services.image_store import sha256_bytes

router = APIRouter(tags=["predict"])
logger = get_logger("zlabel.app.predict")


def _parse_job(data: str) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except Exception as e:  # noqa: BLE001
        raise ValidationFailed(f"data is not valid JSON: {e}") from e
    if not isinstance(payload, dict):
        raise ValidationFailed("data must be a JSON object")
    if not payload.get("anno_id"):
        raise ValidationFailed("data.anno_id is required")
    return payload


def _resolve_image_fast(
    payload: dict, upload: UploadFile | None, services: Services, *, need_bytes: bool
) -> tuple[bytes | None, str]:
    """Which image a predict applies to, and its digest (without a redundant read).

    For a task image referenced by ``rel_path`` the digest is derived from the bytes, so
    the whole stored file used to be read (and, over OpenList, fetched) on every request
    even when the worker already had it decoded. A cheap ``file_info`` etag now memoises
    that read: the digest is stable for a given (size, mtime), and a replaced file
    invalidates it.
    """
    if upload is not None:
        data = upload.file.read()
        return data, sha256_bytes(data)
    rel_path = str(payload.get("rel_path") or "").strip()
    if rel_path:
        path = services.storage.image_path(str(payload["project"]), rel_path)
        info = services.storage.file_info(path)
        digest = services.digests.get(path, info.etag)
        if digest is not None and not need_bytes:
            return None, digest  # cached: no bytes needed, the worker has/refetches them
        data = services.storage.get_bytes(path)
        digest = sha256_bytes(data)
        services.digests.put(path, info.etag, digest)
        return data, digest
    digest = str(payload.get("image_sha256") or "").strip()
    if digest:
        return (services.images.get(digest) if need_bytes else None), digest
    raise ValidationFailed("no image: send a file, rel_path or image_sha256")


@router.post("/projects/{project}/predict")
async def predict(
    project: str,
    data: str = Form(...),
    image: UploadFile | None = File(None),
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> dict[str, Any]:
    services.projects.get_project(project, auth=auth)
    payload = _parse_job(data)
    task = services.tasks.get_task(payload["anno_id"])
    if task.project != project:
        raise ValidationFailed(f"task {payload['anno_id']} does not belong to project {project}")

    inline = services.settings.inference_inline_images
    content, digest = _resolve_image_fast({**payload, "project": project}, image, services, need_bytes=inline)
    image_url = None
    if not inline:
        # the worker pulls it back from us on an embedding miss; ``put`` is idempotent,
        # and a memoised request may arrive without the bytes at all - restore them if
        # the store lost them, otherwise the worker's pull would 404
        if content is not None:
            services.images.put(content)
        elif not services.images.path_for(digest).is_file():
            rel_path = str(payload.get("rel_path") or "").strip()
            if rel_path:  # the bytes vanished from the store: restore them once
                services.images.put(services.storage.get_bytes(services.storage.image_path(project, rel_path)))
        image_url = f"/api/v2/internal/images/{digest}"
    job = {
        "job_id": f"{project}:{payload['anno_id']}:{digest[:12]}",
        "anno_id": payload["anno_id"],
        "image_sha256": digest,
        "image_b64": base64.b64encode(content).decode("ascii") if inline else None,
        "image_url": image_url,
        "model": payload.get("model") or None,
        "prompts": {
            "points": payload.get("points"),
            "labels": payload.get("labels"),
            "rects": payload.get("rects"),
            "texts": payload.get("texts"),
        },
        "threshold": int(payload.get("threshold", 100)),
        "mode": int(payload.get("mode", 1)),
        "return_type": int(payload.get("return_type", 1)),
        "crop_box": payload.get("crop_box"),
        # interactive refinement: the previous prediction's logits come back in, this
        # prediction's own logits go out (both crop-space, base64 float16)
        "mask_input": payload.get("mask_input"),
        "want_low_res": bool(payload.get("want_low_res")),
    }
    logger.info(f"predict anno_id={payload['anno_id']} sha={digest[:12]} mode={job['mode']}")
    return services.inference.infer(job)
