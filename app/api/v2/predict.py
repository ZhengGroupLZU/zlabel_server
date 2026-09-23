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


def _resolve_image(payload: dict, upload: UploadFile | None, services: Services) -> bytes:
    """Uploaded bytes win, then the stored task image, then the upload cache.

    The cache branch is the desktop's bandwidth contract: it uploads a local task image
    once via ``PUT /projects/{p}/images/{rel_path}`` (content addressed) and every later
    predict references it by ``image_sha256`` instead of sending the bytes again. An
    unknown digest is deliberately a 404 - that is how a client learns to upload again.
    """
    if upload is not None:
        return upload.file.read()
    rel_path = str(payload.get("rel_path") or "").strip()
    if rel_path:
        project = str(payload["project"])
        path = services.storage.image_path(project, rel_path)
        return services.storage.get_bytes(path)
    digest = str(payload.get("image_sha256") or "").strip()
    if digest:
        return services.images.get(digest)
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

    content = _resolve_image({**payload, "project": project}, image, services)
    digest = sha256_bytes(content)
    inline = services.settings.inference_inline_images
    image_url = None
    if not inline:
        # the worker pulls it back from us on an embedding miss
        services.images.put(content)
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
    }
    logger.info(f"predict anno_id={payload['anno_id']} sha={digest[:12]} mode={job['mode']}")
    return services.inference.infer(job)
