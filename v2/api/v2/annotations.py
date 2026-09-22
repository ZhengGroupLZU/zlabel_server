"""``/api/v2/projects/{project}/annotations`` — read, save, version history.

The request body of ``PUT`` is the *annotation document itself* (no envelope), so
the file stored in OpenList stays byte-compatible with the desktop mirror.
``base_version``, ``force`` and ``note`` travel as query parameters.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, Query, Response

from v2.api.deps import get_auth, get_services
from v2.schemas.annotations import SaveResponse, VersionOut
from v2.services.auth_service import AuthContext
from v2.services.container import Services

router = APIRouter(prefix="/projects/{project}/annotations", tags=["annotations"])


@router.get("/{anno_id}")
def get_annotation(
    project: str,
    anno_id: str,
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> Response:
    """The stored document; ``404 not_found`` means "not annotated yet"."""
    content, version = services.annotations.get(project, anno_id, auth.oplist_token)
    return Response(
        content=content,
        media_type="application/json",
        headers={
            "ETag": f'"v{version}"',
            "X-Anno-Version": str(version),
            "Cache-Control": "no-store",
        },
    )


@router.put("/{anno_id}", response_model=SaveResponse)
def save_annotation(
    project: str,
    anno_id: str,
    document: dict = Body(...),
    base_version: int | None = Query(None, description="version the client started from"),
    force: bool = Query(False, description="reviewer+: overwrite a newer server copy"),
    note: str = Query("", description="optional note stored with this version"),
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> SaveResponse:
    result = services.annotations.save(
        auth, project, anno_id, document, base_version=base_version, force=force, note=note
    )
    return SaveResponse.of(anno_id, result)


@router.get("/{anno_id}/versions", response_model=list[VersionOut])
def list_versions(
    project: str,
    anno_id: str,
    _auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> list[VersionOut]:
    return [VersionOut.of(row) for row in services.annotations.versions(project, anno_id)]


@router.get("/{anno_id}/versions/{version}")
def get_version(
    project: str,
    anno_id: str,
    version: int,
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> Response:
    """Content of a stored version (the current one included)."""
    content = services.annotations.get_version(project, anno_id, version, auth.oplist_token)
    return Response(
        content=content,
        media_type="application/json",
        headers={"X-Anno-Version": str(version), "Cache-Control": "no-store"},
    )
