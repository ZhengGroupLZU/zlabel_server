"""``/api/v2`` task endpoints: listing, groups, claim/lease, submit/review."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from v2.api.deps import get_auth, get_services
from v2.schemas.tasks import GroupOut, ReopenRequest, ReviewRequest, TaskListOut, TaskOut
from v2.services.auth_service import AuthContext
from v2.services.container import Services

router = APIRouter(tags=["tasks"])

MAX_LIMIT = 500


@router.get("/projects/{project}/tasks", response_model=TaskListOut)
def list_tasks(
    project: str,
    state: str | None = Query(None),
    claim: str | None = Query(None, description="free | mine | others"),
    mine: bool = Query(False),
    group: str | None = Query(None),
    limit: int = Query(50, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
    order: str = Query("sequence", description="sequence | id | recent | random"),
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> TaskListOut:
    services.projects.get_project(project)
    rows, total = services.tasks.list_tasks(
        project,
        state=state,
        claim="mine" if mine else claim,
        user_id=auth.user_id,
        group=group,
        limit=limit,
        offset=offset,
        order=order,
    )
    return TaskListOut(items=[TaskOut.of(row) for row in rows], total=total)


@router.get("/projects/{project}/groups", response_model=list[GroupOut])
def list_groups(
    project: str,
    state: str | None = Query(None),
    mine: bool = Query(False),
    claim: str | None = Query(None),
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> list[GroupOut]:
    """Sequence groups with their frames (the client's timeline source)."""
    services.projects.get_project(project)
    groups = services.tasks.groups(
        project, state=state, user_id=auth.user_id, claim="mine" if mine else claim
    )
    return [
        GroupOut(group=g["group"], count=g["count"], frames=[TaskOut.of(row) for row in g["frames"]])
        for g in groups
    ]


@router.get("/projects/{project}/my-stats")
def my_stats(
    project: str,
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> dict[str, int]:
    services.projects.get_project(project)
    return services.tasks.mine(project, auth.user_id)


@router.get("/tasks/{anno_id}", response_model=TaskOut)
def get_task(
    anno_id: str,
    _auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> TaskOut:
    return TaskOut.of(services.tasks.get_task(anno_id))


@router.post("/tasks/{anno_id}/claim", response_model=TaskOut)
def claim_task(
    anno_id: str,
    force: bool = Query(False),
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> TaskOut:
    """Returns 409 ``lease_conflict`` with the current holder when taken."""
    return TaskOut.of(services.tasks.claim(auth, anno_id, force=force))


@router.post("/tasks/{anno_id}/release", response_model=TaskOut)
def release_task(
    anno_id: str,
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> TaskOut:
    return TaskOut.of(services.tasks.release(auth, anno_id))


@router.post("/tasks/{anno_id}/heartbeat", response_model=TaskOut)
def heartbeat_task(
    anno_id: str,
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> TaskOut:
    return TaskOut.of(services.tasks.heartbeat(auth, anno_id))


@router.post("/tasks/{anno_id}/submit", response_model=TaskOut)
def submit_task(
    anno_id: str,
    force: bool = Query(False),
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> TaskOut:
    return TaskOut.of(services.tasks.submit(auth, anno_id, force=force))


@router.post("/tasks/{anno_id}/review", response_model=TaskOut)
def review_task(
    anno_id: str,
    payload: ReviewRequest,
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> TaskOut:
    """Reviewer/admin only (403 otherwise); a rejection requires a note."""
    return TaskOut.of(services.tasks.review(auth, anno_id, payload.decision, payload.note))


@router.post("/tasks/{anno_id}/reopen", response_model=TaskOut)
def reopen_task(
    anno_id: str,
    payload: ReopenRequest,
    auth: AuthContext = Depends(get_auth),
    services: Services = Depends(get_services),
) -> TaskOut:
    return TaskOut.of(services.tasks.reopen(auth, anno_id, payload.note))
