"""Task listing, claim/lease and the submit/review workflow.

Claiming is what keeps two annotators off the same frame: a claim carries a lease
that expires, so a crashed client cannot lock a task forever. Reviewers may force
their way in; everyone else gets a ``lease_conflict`` carrying the current holder.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session as OrmSession

from v2.core.config import Settings
from v2.core.errors import Conflict, LeaseConflict, NotFound, ValidationFailed
from v2.core.logging import get_logger
from v2.db.base import Database
from v2.db.models import (
    STATE_APPROVED,
    STATE_DRAFT,
    STATE_REJECTED,
    STATE_SUBMITTED,
    STATES,
    Annotation,
    Project,
    Task,
    User,
    utcnow,
)
from v2.services import audit
from v2.services.auth_service import AuthContext

logger = get_logger("zlabel.v2.tasks")

CLAIM_FILTERS = ("free", "mine", "others")
ORDERINGS = ("sequence", "id", "recent")


@dataclass(frozen=True)
class TaskRow:
    """A task plus the metadata the API needs (project name, annotation version,
    resolved holder/reviewer names — reading them through the ORM relationship
    would go stale right after a claim mutates ``claimed_by``)."""

    task: Task
    project: str
    version: int
    has_annotation: bool
    holder: str = ""
    reviewer: str = ""


class TaskService:
    def __init__(self, db: Database, settings: Settings) -> None:
        self.db = db
        self.settings = settings

    # region queries
    def list_tasks(
        self,
        project: str | None,
        *,
        state: str | None = None,
        claim: str | None = None,
        user_id: int | None = None,
        group: str | None = None,
        limit: int = 50,
        offset: int = 0,
        order: str = "sequence",
    ) -> tuple[list[TaskRow], int]:
        if state is not None and state not in STATES:
            raise ValidationFailed(f"unknown state: {state}")
        if claim is not None and claim not in CLAIM_FILTERS:
            raise ValidationFailed(f"unknown claim filter: {claim}")
        if order not in ORDERINGS:
            raise ValidationFailed(f"unknown order: {order}")

        with self.db.session_scope() as session:
            query = self._base_query(session, project)
            if state is not None:
                query = query.where(Task.state == state)
            if group is not None:
                query = query.where(Task.group_name == group)
            query = self._apply_claim_filter(query, claim, user_id)
            total = int(session.scalar(select(func.count()).select_from(query.subquery())) or 0)
            rows = session.scalars(self._order(query, order).limit(limit).offset(offset)).all()
            return self._rows(session, rows), total

    def get_task(self, anno_id: str) -> TaskRow:
        with self.db.session_scope() as session:
            task = session.scalar(select(Task).where(Task.anno_id == anno_id))
            if task is None:
                raise NotFound(f"unknown task: {anno_id}")
            return self._rows(session, [task])[0]

    def groups(
        self,
        project: str,
        *,
        state: str | None = None,
        user_id: int | None = None,
        claim: str | None = None,
        limit_groups: int = 200,
    ) -> list[dict]:
        """Sequence groups with their frames (for the client's timeline)."""
        with self.db.session_scope() as session:
            query = self._base_query(session, project)
            if state is not None:
                query = query.where(Task.state == state)
            query = self._apply_claim_filter(query, claim, user_id)
            rows = session.scalars(query.order_by(Task.group_name, Task.day, Task.rel_path)).all()
            grouped: dict[str, list[Task]] = {}
            for task in rows:
                grouped.setdefault(task.group_name, []).append(task)
            result = []
            for name in sorted(grouped)[:limit_groups]:
                frames = grouped[name]
                result.append({"group": name, "count": len(frames), "frames": self._rows(session, frames)})
            return result

    def _base_query(self, session: OrmSession, project: str | None) -> Select:
        query = (
            select(Task)
            .join(Project, Project.id == Task.project_id)
            .where(Project.active.is_(True), Task.missing.is_(False))
        )
        if project:
            query = query.where(Project.name == project)
        return query

    @staticmethod
    def _apply_claim_filter(query: Select, claim: str | None, user_id: int | None) -> Select:
        now = utcnow()
        if claim == "free":
            return query.where(
                (Task.claimed_by.is_(None))
                | (Task.lease_expires_at.is_(None))
                | (Task.lease_expires_at <= now)
            )
        if claim == "mine":
            return query.where(Task.claimed_by == user_id)
        if claim == "others":
            return query.where(Task.claimed_by.is_not(None), Task.claimed_by != user_id)
        return query

    @staticmethod
    def _order(query: Select, order: str) -> Select:
        if order == "id":
            return query.order_by(Task.id)
        if order == "recent":
            return query.order_by(Task.updated_at.desc())
        # sequence: grouped frames (by group, day) first, ungrouped singles last
        return query.order_by((Task.group_name == "").asc(), Task.group_name, Task.day, Task.rel_path)

    def _rows(self, session: OrmSession, tasks: list[Task]) -> list[TaskRow]:
        """Attach project name + annotation version (one extra query, no N+1)."""
        if not tasks:
            return []
        project_ids = {t.project_id for t in tasks}
        names = dict(
            session.execute(select(Project.id, Project.name).where(Project.id.in_(project_ids))).all()
        )
        anno_ids = [t.anno_id for t in tasks]
        versions = dict(
            session.execute(
                select(Annotation.anno_id, Annotation.version).where(Annotation.anno_id.in_(anno_ids))
            ).all()
        )
        user_ids = {t.claimed_by for t in tasks if t.claimed_by} | {
            t.reviewed_by for t in tasks if t.reviewed_by
        }
        users = (
            dict(session.execute(select(User.id, User.name).where(User.id.in_(user_ids))).all())
            if user_ids
            else {}
        )
        return [
            TaskRow(
                task=task,
                project=names.get(task.project_id, ""),
                version=int(versions.get(task.anno_id, 0) or 0),
                has_annotation=task.anno_id in versions,
                holder=users.get(task.claimed_by, ""),
                reviewer=users.get(task.reviewed_by, ""),
            )
            for task in tasks
        ]

    # endregion

    # region claim / lease
    def claim(self, auth: AuthContext, anno_id: str, *, force: bool = False) -> TaskRow:
        if force:
            auth.require_reviewer()
        with self.db.session_scope() as session:
            task = self._must_get(session, anno_id)
            holder = self._holder_state(task, auth.user_id)
            if holder == "other" and not force:
                raise self._lease_conflict(session, task)
            if task.state in (STATE_SUBMITTED, STATE_APPROVED) and not (force or auth.is_reviewer):
                raise Conflict(
                    f"task is {task.state} and cannot be claimed",
                    detail={"state": task.state, "reviewed_by": self._name(session, task.reviewed_by)},
                )
            stolen_from = self._name(session, task.claimed_by) if holder == "other" else ""
            task.claimed_by = auth.user_id
            task.claimed_at = utcnow()
            task.lease_expires_at = utcnow() + self._lease()
            task.updated_at = utcnow()
            audit.record(
                session,
                action="claim",
                user_id=auth.user_id,
                target_type="task",
                target_id=anno_id,
                detail={"force": force, "stolen_from": stolen_from, "state": task.state},
            )
            return self._rows(session, [task])[0]

    def release(self, auth: AuthContext, anno_id: str) -> TaskRow:
        with self.db.session_scope() as session:
            task = self._must_get(session, anno_id)
            if task.claimed_by != auth.user_id:
                auth.require_reviewer()  # only the holder or a reviewer may release
            released = self._name(session, task.claimed_by)
            task.claimed_by = None
            task.claimed_at = None
            task.lease_expires_at = None
            task.updated_at = utcnow()
            audit.record(
                session,
                action="release",
                user_id=auth.user_id,
                target_type="task",
                target_id=anno_id,
                detail={"released_user": released},
            )
            return self._rows(session, [task])[0]

    def heartbeat(self, auth: AuthContext, anno_id: str) -> TaskRow:
        """Extend the lease; only the current holder may call it."""
        with self.db.session_scope() as session:
            task = self._must_get(session, anno_id)
            if task.claimed_by != auth.user_id:
                raise self._lease_conflict(session, task)
            task.lease_expires_at = utcnow() + self._lease()
            return self._rows(session, [task])[0]

    def renew_lease(self, session: OrmSession, task: Task, user_id: int) -> None:
        """Called by the annotation service after a successful save."""
        task.claimed_by = user_id
        task.claimed_at = task.claimed_at or utcnow()
        task.lease_expires_at = utcnow() + self._lease()

    def ensure_claimable(
        self, session: OrmSession, task: Task, auth: AuthContext, *, force: bool = False
    ) -> None:
        """A save/claim guard shared with the annotation service."""
        if force:
            auth.require_reviewer()
            return
        if self._holder_state(task, auth.user_id) == "other":
            raise self._lease_conflict(session, task)

    def _holder_state(self, task: Task, user_id: int) -> str:
        """``free`` (no claim / expired lease), ``mine`` or ``other``."""
        if task.claimed_by is None:
            return "free"
        if task.claimed_by == user_id:
            return "mine"
        if task.lease_expires_at is None or task.lease_expires_at <= utcnow():
            return "free"  # the lease expired: the task is up for grabs again
        return "other"

    def _lease_conflict(self, session: OrmSession, task: Task) -> LeaseConflict:
        return LeaseConflict(
            f"task is claimed by {self._name(session, task.claimed_by) or 'someone else'}",
            detail={
                "anno_id": task.anno_id,
                "claimed_by": self._name(session, task.claimed_by),
                "claimed_at": task.claimed_at.isoformat() if task.claimed_at else None,
                "lease_expires_at": task.lease_expires_at.isoformat() if task.lease_expires_at else None,
                "state": task.state,
            },
        )

    def _lease(self) -> timedelta:
        return timedelta(minutes=self.settings.lease_minutes)

    def _must_get(self, session: OrmSession, anno_id: str) -> Task:
        task = session.scalar(select(Task).where(Task.anno_id == anno_id))
        if task is None:
            raise NotFound(f"unknown task: {anno_id}")
        return task

    @staticmethod
    def _name(session: OrmSession, user_id: int | None) -> str:
        if user_id is None:
            return ""
        user = session.get(User, user_id)
        return user.name if user is not None else ""

    # endregion

    # region workflow
    def submit(self, auth: AuthContext, anno_id: str, *, force: bool = False) -> TaskRow:
        """Annotator hands the frame to review (needs a stored annotation)."""
        with self.db.session_scope() as session:
            task = self._must_get(session, anno_id)
            if not force and task.claimed_by != auth.user_id:
                raise self._lease_conflict(session, task)
            if force:
                auth.require_reviewer()
            if task.state == STATE_SUBMITTED:
                raise Conflict("task is already submitted", detail={"state": task.state})
            if task.state == STATE_APPROVED and not force:
                raise Conflict("task is already approved", detail={"state": task.state})
            if not self._has_annotation(session, task):
                raise ValidationFailed("nothing to submit: this task has no annotation yet")
            task.state = STATE_SUBMITTED
            task.submitted_at = utcnow()
            task.lease_expires_at = None  # the reviewer takes over; no live lease
            task.updated_at = utcnow()
            audit.record(
                session,
                action="submit",
                user_id=auth.user_id,
                target_type="task",
                target_id=anno_id,
            )
            return self._rows(session, [task])[0]

    def review(self, auth: AuthContext, anno_id: str, decision: str, note: str = "") -> TaskRow:
        """Reviewer approves or rejects a submitted frame."""
        auth.require_reviewer()
        if decision not in ("approve", "reject"):
            raise ValidationFailed(f"unknown decision: {decision}")
        if decision == "reject" and not note.strip():
            raise ValidationFailed("a rejection needs a note explaining what to fix")
        with self.db.session_scope() as session:
            task = self._must_get(session, anno_id)
            if task.state != STATE_SUBMITTED:
                raise Conflict(
                    f"task is {task.state}: only submitted frames can be reviewed",
                    detail={"state": task.state},
                )
            task.reviewed_by = auth.user_id
            task.reviewed_at = utcnow()
            task.review_note = note.strip()
            task.updated_at = utcnow()
            if decision == "approve":
                task.state = STATE_APPROVED
            else:
                task.state = STATE_REJECTED
                task.lease_expires_at = None  # free for the annotator to pick up again
            audit.record(
                session,
                action=f"review_{decision}",
                user_id=auth.user_id,
                target_type="task",
                target_id=anno_id,
                detail={"note": task.review_note, "annotator": self._name(session, task.claimed_by)},
            )
            return self._rows(session, [task])[0]

    def reopen(self, auth: AuthContext, anno_id: str, note: str = "") -> TaskRow:
        """Reviewer pulls an approved frame back into the work queue."""
        auth.require_reviewer()
        with self.db.session_scope() as session:
            task = self._must_get(session, anno_id)
            if task.state not in (STATE_APPROVED, STATE_SUBMITTED):
                raise Conflict(f"task is {task.state}: nothing to reopen", detail={"state": task.state})
            task.state = STATE_DRAFT
            task.reviewed_by = auth.user_id
            task.reviewed_at = utcnow()
            task.review_note = note.strip()
            task.submitted_at = None
            task.lease_expires_at = None
            task.updated_at = utcnow()
            audit.record(
                session,
                action="reopen",
                user_id=auth.user_id,
                target_type="task",
                target_id=anno_id,
                detail={"note": task.review_note},
            )
            return self._rows(session, [task])[0]

    @staticmethod
    def _has_annotation(session: OrmSession, task: Task) -> bool:
        return session.scalar(select(Annotation.id).where(Annotation.task_id == task.id)) is not None

    # endregion

    # region stats
    def mine(self, project: str | None, user_id: int) -> dict[str, int]:
        with self.db.session_scope() as session:
            rows = session.execute(
                self._base_query(session, project)
                .where(Task.claimed_by == user_id)
                .with_only_columns(Task.state, func.count())
                .group_by(Task.state)
            ).all()
            counts = dict.fromkeys(STATES, 0)
            for state, count in rows:
                counts[state] = int(count)
            return counts

    # endregion
