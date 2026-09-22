"""Machine-to-machine endpoints (the inference worker is the only client).

Guarded by the same shared secret as ``/infer`` (``ZLSERVER_INFERENCE_TOKEN``) instead
of a user session, because the worker has no user identity.
"""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, Header, Request, Response

from v2.api.deps import bearer_token, get_services
from v2.core.errors import Forbidden, Unauthorized
from v2.services.container import Services

router = APIRouter(prefix="/internal", tags=["internal"])


def require_internal_token(request: Request, authorization: str | None = Header(None)) -> None:
    expected = getattr(request.app.state.settings, "inference_token", "")
    if not expected:
        raise Forbidden("ZLSERVER_INFERENCE_TOKEN is not configured")
    if not secrets.compare_digest(bearer_token(authorization), expected):
        raise Unauthorized("bad internal token")


@router.get("/images/{digest}")
def get_uploaded_image(
    digest: str,
    _token: None = Depends(require_internal_token),
    services: Services = Depends(get_services),
) -> Response:
    """Serve an uploaded frame so the worker can pull it on an embedding miss."""
    content = services.images.get(digest)
    return Response(content=content, media_type="application/octet-stream")
