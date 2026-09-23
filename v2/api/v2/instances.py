"""``/api/v2/projects/{project}/instances`` — the mirrored instance registry.

Instances are created automatically when an annotation document introduces an
``instance_id``; the endpoints here let clients and the admin UI inspect and edit
the metadata (status/name/note/colour/archived). The number itself is the id the
documents reference and is never renumbered.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from v2.api.deps import get_auth, get_services
from v2.schemas.projects import InstanceCreate, InstanceOut, InstancePatch, InstanceResultOut
from v2.services.auth_service import AuthContext
from v2.services.container import Services

router = APIRouter(prefix="/projects/{project}/instances", tags=["instances"])


@router.get("", response_model=list[InstanceOut])
def list_instances(
    project: str,
    include_archived: bool = False,
    _auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> list[InstanceOut]:
    """The project's instances, ordered by number, with result/frame counts."""
    services.projects.require_access(_auth, project)
    return [
        InstanceOut(**item)
        for item in services.instances.list_instances(project, include_archived=include_archived)
    ]


@router.post("", response_model=InstanceOut, status_code=201)
def create_instance(
    project: str,
    payload: InstanceCreate,
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> InstanceOut:
    services.projects.require_project_reviewer(auth, project)
    return InstanceOut(
        **services.instances.create_instance(
            project,
            number=payload.number,
            name=payload.name,
            note=payload.note,
            status=payload.status,
            color=payload.color,
            actor_id=auth.user_id,
        )
    )


@router.get("/{number}/results", response_model=list[InstanceResultOut])
def instance_results(
    project: str,
    number: int,
    _auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> list[InstanceResultOut]:
    """The annotations that belong to one instance (frame + result + label)."""
    services.projects.require_access(_auth, project)
    return [InstanceResultOut(**item) for item in services.instances.results_of(project, number)]


@router.patch("/{number}", response_model=InstanceOut)
def update_instance(
    project: str,
    number: int,
    payload: InstancePatch,
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> InstanceOut:
    services.projects.require_project_reviewer(auth, project)
    return InstanceOut(
        **services.instances.update_instance(
            project,
            number,
            name=payload.name,
            note=payload.note,
            status=payload.status,
            color=payload.color,
            archived=payload.archived,
            actor_id=auth.user_id,
        )
    )


@router.delete("/{number}", status_code=204)
def delete_instance(
    project: str,
    number: int,
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> Response:
    """Drop the registry row (and its links); the documents keep their numbers.

    A later annotation save that still references the number re-creates it.
    """
    services.projects.require_project_reviewer(auth, project)
    services.instances.delete_instance(project, number, actor_id=auth.user_id)
    return Response(status_code=204)


@router.get("/statuses", response_model=list[str])
def list_statuses(
    project: str,
    _auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> list[str]:
    """The status vocabulary the UI offers (presets + what the project uses)."""
    services.projects.require_access(_auth, project)
    return services.instances.status_presets(project)
