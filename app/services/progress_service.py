"""Per-user workload of a project (the status bar's Annotators tab).

The panel used to show the *current* state distribution of whatever each user
holds right now, which answers "who is working" but not "who did how much".
This module aggregates the history instead - ``audit_log`` for the actions
(claim / submit / review) and ``annotation_versions`` for the authored versions.

Almost every column is a **task count**, so a row reads like a small workflow
snapshot and the columns add up (except ``total``, whose per-user unions overlap):

==============  ===========================================================
``role``        ``users.role`` (annotator / reviewer / admin)
``claimed``     distinct tasks the user ever claimed (release + re-claim = 1)
``created``     distinct tasks whose first version the user authored
``modified``    distinct tasks the user authored a later version of
``submitted``   distinct tasks the user handed in (submitting twice = 1 task)
``approved``    of those, the ones currently approved
``rejected``    of those, the ones currently rejected
``open``        tasks the user still holds in ``draft`` (in hand, not handed in)
``total``       distinct tasks in the union of the four task-level columns above
``submits``     submit *actions* (a rejected and resubmitted task counts twice)
``rejections``  rejection *events* attributed to the annotator of record
==============  ===========================================================

The last two are deliberately narrow "times" columns: they keep the process
signal without making the unit of the table ambiguous. A review is attributed to
the annotator of record (``annotator_id``, then the historic ``annotator`` name,
then the task's current holder) and never to the reviewer.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session as OrmSession

from app.db.models import (
    STATE_APPROVED,
    STATE_DRAFT,
    STATE_REJECTED,
    AnnotationVersion,
    AuditLog,
    Project,
    ProjectMember,
    Task,
    User,
)

#: audit actions this module reads
CLAIM_ACTION = "claim"
SUBMIT_ACTION = "submit"
REVIEW_REJECT = "review_reject"

#: numeric columns of one row (everything except ``role``)
#: task-count columns (a row of these can be read and added up)
TASK_FIELDS: tuple[str, ...] = (
    "claimed",
    "created",
    "modified",
    "submitted",
    "approved",
    "rejected",
    "open",
    "total",
)
#: event-count columns ("how many times": a task can count more than once)
EVENT_FIELDS: tuple[str, ...] = ("submits", "rejections")
WORKLOAD_FIELDS: tuple[str, ...] = TASK_FIELDS + EVENT_FIELDS


def empty_row(role: str = "") -> dict[str, Any]:
    """A zeroed workload row (also the shape the API returns)."""
    return {"role": role, **dict.fromkeys(WORKLOAD_FIELDS, 0)}


def by_user_workload(session: OrmSession, project_id: int | None = None) -> dict[str, dict[str, Any]]:
    """``{user name: row}`` for every user with history in the project's tasks.

    ``project_id`` restricts the aggregation to one project; ``None`` covers every
    active project. Users with no record at all are not invented here - the caller
    adds the project's members so an idle collaborator still shows up with zeros.
    """
    rows: dict[str, dict[str, Any]] = {}

    for name, count in _claimed_tasks(session, project_id).items():
        rows.setdefault(name, empty_row())["claimed"] = count
    for name, created, modified in _version_authors(session, project_id):
        entry = rows.setdefault(name, empty_row())
        entry["created"] = created
        entry["modified"] = modified
    for name, count in _submitted_tasks(session, project_id).items():
        rows.setdefault(name, empty_row())["submitted"] = count
    for state in (STATE_APPROVED, STATE_REJECTED):
        for name, count in _tasks_in_state(session, project_id, state).items():
            rows.setdefault(name, empty_row())[state] = count
    for name, count in _open_tasks(session, project_id).items():
        rows.setdefault(name, empty_row())["open"] = count
    for name, count in _submit_actions(session, project_id).items():
        rows.setdefault(name, empty_row())["submits"] = count
    for name, count in _rejection_events(session, project_id).items():
        rows.setdefault(name, empty_row())["rejections"] = count

    touched = _touched_tasks(session, project_id)
    for name in rows:
        rows[name]["total"] = len(touched.get(name, ()))
    roles = _roles(session, list(rows))
    for name, entry in rows.items():
        entry["role"] = roles.get(name, "")
    return rows


def project_members(session: OrmSession, project_id: int) -> list[tuple[str, str]]:
    """``[(name, role)]`` of the project's members (the project role wins)."""
    rows = session.execute(
        select(User.name, ProjectMember.role, User.role)
        .join(ProjectMember, ProjectMember.user_id == User.id)
        .where(ProjectMember.project_id == project_id)
    ).all()
    return [(str(name), str(member_role or global_role or "")) for name, member_role, global_role in rows]


# region queries
def _task_scope(project_id: int | None) -> list[Any]:
    """The ``where`` clauses that keep a query inside the wanted active project tasks."""
    clauses: list[Any] = [Project.active.is_(True), Task.missing.is_(False)]
    if project_id is not None:
        clauses.append(Task.project_id == project_id)
    return clauses


def _claimed_tasks(session: OrmSession, project_id: int | None) -> dict[str, int]:
    """Distinct tasks each user ever claimed (a release + re-claim stays one)."""
    query = (
        select(User.name, func.count(func.distinct(AuditLog.target_id)))
        .select_from(AuditLog)
        .join(User, User.id == AuditLog.user_id)
        .join(Task, Task.anno_id == AuditLog.target_id)
        .join(Project, Project.id == Task.project_id)
        .where(AuditLog.action == CLAIM_ACTION, AuditLog.target_type == "task", *_task_scope(project_id))
    )
    return {str(name): int(count) for name, count in session.execute(query.group_by(User.name)).all()}


def _version_authors(session: OrmSession, project_id: int | None) -> list[tuple[str, int, int]]:
    """``(name, created, modified)``: version 1 = created, later versions = modified."""
    query = (
        select(
            User.name,
            func.count(func.distinct(case((AnnotationVersion.version == 1, AnnotationVersion.task_id)))),
            func.count(func.distinct(case((AnnotationVersion.version > 1, AnnotationVersion.task_id)))),
        )
        .select_from(AnnotationVersion)
        .join(User, User.id == AnnotationVersion.author_id)
        .join(Task, Task.id == AnnotationVersion.task_id)
        .join(Project, Project.id == Task.project_id)
        .where(*_task_scope(project_id))
    )
    return [
        (str(name), int(created or 0), int(modified or 0))
        for name, created, modified in session.execute(query.group_by(User.name)).all()
    ]


def _submitted_tasks(session: OrmSession, project_id: int | None) -> dict[str, int]:
    """Distinct tasks each user handed in (submitting the same task twice = 1)."""
    query = (
        select(User.name, func.count(func.distinct(AuditLog.target_id)))
        .select_from(AuditLog)
        .join(User, User.id == AuditLog.user_id)
        .join(Task, Task.anno_id == AuditLog.target_id)
        .join(Project, Project.id == Task.project_id)
        .where(AuditLog.action == SUBMIT_ACTION, AuditLog.target_type == "task", *_task_scope(project_id))
    )
    return {str(name): int(count) for name, count in session.execute(query.group_by(User.name)).all()}


def _submit_actions(session: OrmSession, project_id: int | None) -> dict[str, int]:
    """Every ``submit`` action counts (the ``submits`` column)."""
    query = (
        select(User.name, func.count())
        .select_from(AuditLog)
        .join(User, User.id == AuditLog.user_id)
        .join(Task, Task.anno_id == AuditLog.target_id)
        .join(Project, Project.id == Task.project_id)
        .where(AuditLog.action == SUBMIT_ACTION, AuditLog.target_type == "task", *_task_scope(project_id))
    )
    return {str(name): int(count) for name, count in session.execute(query.group_by(User.name)).all()}


def _tasks_in_state(session: OrmSession, project_id: int | None, state: str) -> dict[str, int]:
    """Tasks currently in ``state``, attributed to whoever submitted them last.

    This is the snapshot half of the table, so it lines up with the States tab:
    ``submitted ~= approved + rejected + open + still under review``. The submitter
    comes from the last ``submit`` audit row; a task saved by an older build (no
    audit trail) falls back to its current holder.
    """
    last_submit = (
        select(func.max(AuditLog.id))
        .where(
            AuditLog.target_id == Task.anno_id,
            AuditLog.action == SUBMIT_ACTION,
            AuditLog.target_type == "task",
        )
        .correlate(Task)
        .scalar_subquery()
    )
    annotator_id = func.coalesce(AuditLog.user_id, Task.claimed_by)
    query = (
        select(User.name, func.count(func.distinct(Task.anno_id)))
        .select_from(Task)
        .join(AuditLog, AuditLog.id == last_submit, isouter=True)
        .join(User, User.id == annotator_id)
        .join(Project, Project.id == Task.project_id)
        .where(Task.state == state, *_task_scope(project_id))
    )
    return {str(name): int(count) for name, count in session.execute(query.group_by(User.name)).all()}


def _open_tasks(session: OrmSession, project_id: int | None) -> dict[str, int]:
    """Tasks currently in hand and not handed in (``claimed_by`` + ``draft``)."""
    query = (
        select(User.name, func.count(func.distinct(Task.anno_id)))
        .select_from(Task)
        .join(User, User.id == Task.claimed_by)
        .join(Project, Project.id == Task.project_id)
        .where(Task.state == STATE_DRAFT, *_task_scope(project_id))
    )
    return {str(name): int(count) for name, count in session.execute(query.group_by(User.name)).all()}


def _rejection_events(session: OrmSession, project_id: int | None) -> dict[str, int]:
    """Rejection *events* attributed to the annotator of record (the ``rejections`` column)."""
    query = (
        select(AuditLog.detail_json, Task.claimed_by)
        .select_from(AuditLog)
        .join(Task, Task.anno_id == AuditLog.target_id)
        .join(Project, Project.id == Task.project_id)
        .where(AuditLog.action == REVIEW_REJECT, *_task_scope(project_id))
    )
    rows = session.execute(query).all()
    if not rows:
        return {}
    names = {
        int(user_id): str(name)
        for user_id, name in session.execute(select(User.id, User.name)).all()
        if user_id is not None
    }
    totals: dict[str, int] = defaultdict(int)
    for detail_json, holder_id in rows:
        detail = _detail(detail_json)
        annotator = names.get(_as_int(detail.get("annotator_id")), "")
        annotator = annotator or str(detail.get("annotator") or "") or names.get(holder_id, "")
        if annotator:
            totals[annotator] += 1
    return dict(totals)


def _touched_tasks(session: OrmSession, project_id: int | None) -> dict[str, set[str]]:
    """Tasks per user over claims, authored versions and submits - the Total union."""
    claimed = session.execute(
        select(User.name, AuditLog.target_id)
        .select_from(AuditLog)
        .join(User, User.id == AuditLog.user_id)
        .join(Task, Task.anno_id == AuditLog.target_id)
        .join(Project, Project.id == Task.project_id)
        .where(AuditLog.action == CLAIM_ACTION, AuditLog.target_type == "task", *_task_scope(project_id))
        .distinct()
    ).all()
    submitted = session.execute(
        select(User.name, AuditLog.target_id)
        .select_from(AuditLog)
        .join(User, User.id == AuditLog.user_id)
        .join(Task, Task.anno_id == AuditLog.target_id)
        .join(Project, Project.id == Task.project_id)
        .where(AuditLog.action == SUBMIT_ACTION, AuditLog.target_type == "task", *_task_scope(project_id))
        .distinct()
    ).all()
    authored = session.execute(
        select(User.name, Task.anno_id)
        .select_from(AnnotationVersion)
        .join(User, User.id == AnnotationVersion.author_id)
        .join(Task, Task.id == AnnotationVersion.task_id)
        .join(Project, Project.id == Task.project_id)
        .where(*_task_scope(project_id))
        .distinct()
    ).all()

    touched: dict[str, set[str]] = defaultdict(set)
    for name, anno_id in [*claimed, *submitted, *authored]:
        if name and anno_id:
            touched[str(name)].add(str(anno_id))
    return touched


def _roles(session: OrmSession, names: list[str]) -> dict[str, str]:
    if not names:
        return {}
    rows = session.execute(select(User.name, User.role).where(User.name.in_(names))).all()
    return {str(name): str(role or "") for name, role in rows}


def _detail(detail_json: str | None) -> dict[str, Any]:
    try:
        value = json.loads(detail_json or "{}")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# endregion
