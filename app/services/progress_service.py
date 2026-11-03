"""Per-user workload of a project (the status bar's Annotators tab).

The panel used to show the *current* state distribution of whatever each user
holds right now, which answers "who is working" but not "who did how much".
This module aggregates the history instead - ``audit_log`` for the actions
(claim / submit / review) and ``annotation_versions`` for the authored versions:

=============  ============================================================
``role``       ``users.role`` (annotator / reviewer / admin)
``claimed``    distinct tasks the user ever claimed (release + re-claim = 1)
``created``    distinct tasks whose first version the user authored
``modified``   distinct tasks the user authored a later version of
``submitted``  ``submit`` actions; a rejected task resubmitted counts twice
``approved``   ``review_approve`` events attributed to the task's annotator
``rejected``   ``review_reject`` events, same attribution
``total``      distinct tasks in the union of the task-level columns above
=============  ============================================================

The task-level columns are task counts while ``submitted``/``approved``/
``rejected`` are event counts, so a row does not add up - that is why the client
shows ``—`` in the Total column of its summary row. A review is attributed to the
annotator of record (``annotator_id``, then the historic ``annotator`` name, then
the task's current holder) and never to the reviewer.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session as OrmSession

from app.db.models import AnnotationVersion, AuditLog, Project, ProjectMember, Task, User

#: audit actions this module reads
CLAIM_ACTION = "claim"
SUBMIT_ACTION = "submit"
REVIEW_ACTIONS = ("review_approve", "review_reject")

#: numeric columns of one row (everything except ``role``)
WORKLOAD_FIELDS: tuple[str, ...] = (
    "claimed",
    "created",
    "modified",
    "submitted",
    "approved",
    "rejected",
    "total",
)


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
    for name, count in _submit_actions(session, project_id).items():
        rows.setdefault(name, empty_row())["submitted"] = count
    for name, approved, rejected in _review_events(session, project_id):
        entry = rows.setdefault(name, empty_row())
        entry["approved"] = approved
        entry["rejected"] = rejected

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


def _submit_actions(session: OrmSession, project_id: int | None) -> dict[str, int]:
    """Every ``submit`` action counts (a rejected task resubmitted counts twice)."""
    query = (
        select(User.name, func.count())
        .select_from(AuditLog)
        .join(User, User.id == AuditLog.user_id)
        .join(Task, Task.anno_id == AuditLog.target_id)
        .join(Project, Project.id == Task.project_id)
        .where(AuditLog.action == SUBMIT_ACTION, AuditLog.target_type == "task", *_task_scope(project_id))
    )
    return {str(name): int(count) for name, count in session.execute(query.group_by(User.name)).all()}


def _review_events(session: OrmSession, project_id: int | None) -> list[tuple[str, int, int]]:
    """``(annotator name, approved, rejected)`` - the annotator's record, not the reviewer's."""
    query = (
        select(AuditLog.action, AuditLog.detail_json, Task.claimed_by)
        .select_from(AuditLog)
        .join(Task, Task.anno_id == AuditLog.target_id)
        .join(Project, Project.id == Task.project_id)
        .where(AuditLog.action.in_(REVIEW_ACTIONS), *_task_scope(project_id))
    )
    rows = session.execute(query).all()
    if not rows:
        return []
    names = {
        int(user_id): str(name)
        for user_id, name in session.execute(select(User.id, User.name)).all()
        if user_id is not None
    }
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for action, detail_json, holder_id in rows:
        detail = _detail(detail_json)
        annotator = names.get(_as_int(detail.get("annotator_id")), "")
        annotator = annotator or str(detail.get("annotator") or "") or names.get(holder_id, "")
        if not annotator:
            continue  # anonymous row (user deleted): nothing to attribute it to
        totals[annotator][0 if action == "review_approve" else 1] += 1
    return [(name, values[0], values[1]) for name, values in totals.items()]


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
