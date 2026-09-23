"""``/api/v2/projects`` — listing, creation, metadata, scanning, progress."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response

from app.api.deps import get_auth, get_services, require_roles
from app.core.errors import ApiError
from app.schemas.projects import ProgressOut, ProjectCreate, ProjectOut, ProjectPatch, ScanStats
from app.services.auth_service import AuthContext
from app.services.container import Services

router = APIRouter(prefix="/projects", tags=["projects"])


def _project_exists(services: Services, project: str) -> bool:
    """Unknown project = a brand-new directory that a scan may discover."""
    try:
        services.projects.get_project(project)
        return True
    except ApiError:
        return False


@router.get("", response_model=list[ProjectOut])
def list_projects(
    _auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> list[ProjectOut]:
    return [
        ProjectOut.of(project, services.projects.progress(project.name))
        for project in services.projects.list_projects(auth=_auth)
    ]


@router.post("", response_model=ProjectOut, status_code=201)
def create_project(
    payload: ProjectCreate,
    auth: AuthContext = Depends(require_roles("reviewer", "admin")),  # no project yet to scope to
    services: Services = Depends(get_services),
) -> ProjectOut:
    project = services.projects.create_project(
        payload.name, payload.display_name, timeline=payload.timeline, actor_id=auth.user_id
    )
    return ProjectOut.of(project)


@router.get("/{project}", response_model=ProjectOut)
def get_project(
    project: str,
    _auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> ProjectOut:
    found = services.projects.get_project(project, auth=_auth)
    return ProjectOut.of(found, services.projects.progress(project))


@router.patch("/{project}", response_model=ProjectOut)
def update_project(
    project: str,
    payload: ProjectPatch,
    auth: AuthContext = Depends(require_roles("reviewer", "admin")),
    services: Services = Depends(get_services),
) -> ProjectOut:
    services.projects.require_project_reviewer(auth, project)
    updated = services.projects.update_project(
        project,
        display_name=payload.display_name,
        description=payload.description,
        active=payload.active,
        timeline=payload.timeline,
        actor_id=auth.user_id,
    )
    return ProjectOut.of(updated, services.projects.progress(project))


@router.delete("/{project}", status_code=204)
def delete_project(
    project: str,
    delete_files: bool = Query(True, description="also delete the project directory on disk"),
    auth: AuthContext = Depends(require_roles("admin")),
    services: Services = Depends(get_services),
) -> Response:
    """Delete a project (admin only, irreversible): rows + files by default."""
    services.projects.delete_project(project, delete_files=delete_files, actor_id=auth.user_id)
    return Response(status_code=204)


@router.post("/scan", response_model=ScanStats)
def scan_all(
    force: bool = Query(True),
    _auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> ScanStats:
    """Re-walk the storage tree into the task table (the client's "Scan"/Fetch button)."""
    return ScanStats(**services.projects.scan_and_sync(force=force))


@router.post("/{project}/scan", response_model=ScanStats)
def scan_project(
    project: str,  # noqa: ARG001 - the storage walk is global per root
    force: bool = Query(True),
    _auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> ScanStats:
    """Same as ``POST /projects/scan``: the storage walk is global per root.

    Needs a reviewer for this project - or, when the project is not known yet
    (discovering brand-new storage directories), a global reviewer.
    """
    if not _project_exists(services, project):
        _auth.require_reviewer()
    else:
        services.projects.require_project_reviewer(_auth, project)
    return ScanStats(**services.projects.scan_and_sync(force=force))


@router.get("/{project}/progress", response_model=ProgressOut)
def project_progress(
    project: str,
    by_user: bool = Query(False),
    _auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> ProgressOut:
    services.projects.get_project(project, auth=_auth)
    return ProgressOut.of(services.projects.progress(project, by_user=by_user))
