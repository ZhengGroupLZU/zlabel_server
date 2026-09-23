"""``/api/v2/projects/{project}/labels`` — the server-side label registry."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from v2.api.deps import get_auth, get_services
from v2.schemas.projects import LabelCreate, LabelOrder, LabelOut, LabelPatch
from v2.services.auth_service import AuthContext
from v2.services.container import Services

router = APIRouter(prefix="/projects/{project}/labels", tags=["labels"])


@router.get("", response_model=list[LabelOut])
def list_labels(
    project: str,
    include_archived: bool = False,
    _auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> list[LabelOut]:
    return [
        LabelOut.of(label)
        for label in services.projects.list_labels(project, include_archived=include_archived)
    ]


@router.post("", response_model=LabelOut, status_code=201)
def create_label(
    project: str,
    payload: LabelCreate,
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> LabelOut:
    services.projects.require_project_reviewer(auth, project)
    return LabelOut.of(
        services.projects.create_label(
            project, payload.name, color=payload.color, sort=payload.sort, actor_id=auth.user_id
        )
    )


@router.patch("/{label_id}", response_model=LabelOut)
def update_label(
    project: str,
    label_id: int,
    payload: LabelPatch,
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> LabelOut:
    services.projects.require_project_reviewer(auth, project)
    return LabelOut.of(
        services.projects.update_label(
            project,
            label_id,
            name=payload.name,
            color=payload.color,
            sort=payload.sort,
            archived=payload.archived,
            actor_id=auth.user_id,
        )
    )


@router.put("/order", response_model=list[LabelOut])
def reorder_labels(
    project: str,
    payload: LabelOrder,
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> list[LabelOut]:
    """Set the label order (the ordinal "id" shown in the admin UI)."""
    services.projects.require_project_reviewer(auth, project)
    return [
        LabelOut.of(label)
        for label in services.projects.reorder_labels(project, payload.order, actor_id=auth.user_id)
    ]


@router.delete("/{label_id}", status_code=204)
def delete_label(
    project: str,
    label_id: int,
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> Response:
    services.projects.require_project_reviewer(auth, project)
    services.projects.delete_label(project, label_id, actor_id=auth.user_id)
    return Response(status_code=204)
