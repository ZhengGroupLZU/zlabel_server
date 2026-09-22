"""``/api/v2/projects/{project}/members`` — who may work on a project (P4).

Membership is the source of truth in strict access mode; a project's reviewers and
admins manage it. Everyone *in* the project may read the list.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from v2.api.deps import get_auth, get_services
from v2.schemas.projects import MemberCreate, MemberOut, MemberPatch
from v2.services.auth_service import AuthContext
from v2.services.container import Services

router = APIRouter(prefix="/projects/{project}/members", tags=["members"])


@router.get("", response_model=list[MemberOut])
def list_members(
    project: str,
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> list[MemberOut]:
    services.projects.require_access(auth, project)
    return [MemberOut(**member) for member in services.projects.list_members(project)]


@router.post("", response_model=MemberOut, status_code=201)
def add_member(
    project: str,
    payload: MemberCreate,
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> MemberOut:
    services.projects.require_project_reviewer(auth, project)
    member = services.projects.add_member(project, payload.user_id, payload.role, actor_id=auth.user_id)
    return MemberOut(**member)


@router.patch("/{user_id}", response_model=MemberOut)
def set_member_role(
    project: str,
    user_id: int,
    payload: MemberPatch,
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> MemberOut:
    services.projects.require_project_reviewer(auth, project)
    member = services.projects.add_member(project, user_id, payload.role, actor_id=auth.user_id)
    return MemberOut(**member)


@router.delete("/{user_id}", status_code=204)
def remove_member(
    project: str,
    user_id: int,
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> Response:
    services.projects.require_project_reviewer(auth, project)
    services.projects.remove_member(project, user_id, actor_id=auth.user_id)
    return Response(status_code=204)
