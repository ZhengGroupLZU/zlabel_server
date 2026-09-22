"""Reading and writing annotation documents.

The document itself stays a plain JSON file in OpenList (``<project>/zlabel/
<anno_id>.zlabel``) so mirrors and other tooling keep working; the DB holds the
metadata that makes concurrent editing safe:

- ``version`` + ``base_version`` gives optimistic locking (a save based on an old
  version is refused with 409 and the current author/version),
- every accepted save is also copied to ``zlabel/_history/<anno_id>/v<n>.zlabel``
  so history can be previewed and rolled back,
- the server owns ``updated_at`` (no more client clock skew deciding conflicts).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from v2.adapters.openlist import OpenListAdapter
from v2.core.config import Settings
from v2.core.errors import Conflict, NotFound
from v2.core.logging import get_logger
from v2.db.base import Database
from v2.db.models import (
    STATE_APPROVED,
    STATE_DRAFT,
    STATE_REJECTED,
    STATE_SUBMITTED,
    Annotation,
    AnnotationVersion,
    Task,
    User,
    utcnow,
)
from v2.services import audit
from v2.services.auth_service import AuthContext
from v2.services.project_service import ProjectService
from v2.services.task_service import TaskService

logger = get_logger("zlabel.v2.annotations")


@dataclass(frozen=True)
class SaveResult:
    version: int
    state: str
    updated_at: datetime
    content_hash: str
    labels: list[str]


@dataclass(frozen=True)
class VersionRow:
    version: int
    author: str
    created_at: datetime | None
    note: str
    content_hash: str
    labels: list[str]


def extract_label_names(anno: dict[str, Any]) -> list[str]:
    """Label names used by a client annotation document (``results[*].labels``)."""
    names: list[str] = []
    results = anno.get("results") or {}
    items = results.values() if isinstance(results, dict) else results
    for result in items:
        for label in (result or {}).get("labels") or []:
            name = str((label or {}).get("name", "")).strip()
            if name and name not in names:
                names.append(name)
    return names


class AnnotationService:
    def __init__(
        self,
        db: Database,
        openlist: OpenListAdapter,
        projects: ProjectService,
        tasks: TaskService,
        settings: Settings,
    ) -> None:
        self.db = db
        self.openlist = openlist
        self.projects = projects
        self.tasks = tasks
        self.settings = settings

    # region reads
    def get(self, project: str, anno_id: str, token: str) -> tuple[bytes, int]:
        """Current document + version; 404 when this frame is not annotated yet.

        ``token`` is the *session user's* OpenList token: reads honour the user's
        own ACLs (a user must not see annotations of directories they cannot read).
        Writes, by contrast, go through the service account (``_write_token``).
        """
        content = self.openlist.get_bytes(self.openlist.anno_path(project, anno_id), token)
        with self.db.session_scope() as session:
            row = session.scalar(select(Annotation).where(Annotation.anno_id == anno_id))
        return content, int(row.version) if row is not None else 0

    def versions(self, project: str, anno_id: str) -> list[VersionRow]:
        with self.db.session_scope() as session:
            task = session.scalar(select(Task).where(Task.anno_id == anno_id))
            if task is None:
                raise NotFound(f"unknown task: {anno_id}")
            rows = session.scalars(
                select(AnnotationVersion)
                .where(AnnotationVersion.task_id == task.id)
                .order_by(AnnotationVersion.version.desc())
            ).all()
            names = self._user_names(session, [row.author_id for row in rows])
            return [
                VersionRow(
                    version=row.version,
                    author=names.get(row.author_id, ""),
                    created_at=row.created_at,
                    note=row.note or "",
                    content_hash=row.content_hash or "",
                    labels=json.loads(row.labels_json or "[]"),
                )
                for row in rows
            ]

    def get_version(self, project: str, anno_id: str, version: int, token: str) -> bytes:
        """Historical document, or the current one when the versions match."""
        current_bytes, current_version = self.get(project, anno_id, token)
        if version <= 0 or version == current_version:
            return current_bytes
        path = self.openlist.history_path(project, anno_id, version)
        return self.openlist.get_bytes(path, token)

    def meta(self, anno_id: str) -> tuple[int, str]:
        """``(version, state)`` for a task; ``(0, state)`` when never saved."""
        with self.db.session_scope() as session:
            task = session.scalar(select(Task).where(Task.anno_id == anno_id))
            if task is None:
                raise NotFound(f"unknown task: {anno_id}")
            row = session.scalar(select(Annotation).where(Annotation.task_id == task.id))
            return (int(row.version) if row is not None else 0), task.state

    # endregion

    # region writes
    def save(
        self,
        auth: AuthContext,
        project: str,
        anno_id: str,
        document: dict[str, Any],
        *,
        base_version: int | None = None,
        force: bool = False,
        note: str = "",
    ) -> SaveResult:
        """Store a document under the task's anno_id (claim + version checked)."""
        if force:
            auth.require_reviewer()
        payload = json.dumps(document, ensure_ascii=False).encode("utf-8")
        content_hash = hashlib.sha256(payload).hexdigest()
        labels = extract_label_names(document)
        # writes use the server's storage identity (ZLSERVER_OPLIST_TOKEN or the
        # service credentials): annotators usually have read-only OpenList access
        token = self.openlist.service_token()

        with self.db.session_scope() as session:
            task = session.scalar(select(Task).where(Task.anno_id == anno_id))
            if task is None:
                raise NotFound(f"unknown task: {anno_id}")
            if task.project.name != project:
                raise NotFound(f"task {anno_id} does not belong to project {project}")

            if task.state in (STATE_SUBMITTED, STATE_APPROVED) and not force:
                raise Conflict(
                    f"task is {task.state}: reopen it before editing",
                    detail={"state": task.state},
                )
            self.tasks.ensure_claimable(session, task, auth, force=force)

            current = session.scalar(select(Annotation).where(Annotation.task_id == task.id))
            current_version = int(current.version) if current is not None else 0
            # "no base version" is only acceptable while nothing is stored: an
            # edit based on an unknown version would silently clobber someone.
            effective_base = 0 if base_version is None else base_version
            if effective_base < current_version and not force:
                raise Conflict(
                    "the server copy is newer",
                    detail=self._conflict_detail(session, current),
                )

            new_version = current_version + 1
            self._write_files(project, anno_id, payload, new_version, token)
            task.labels = self.projects.ensure_labels(session, task.project_id, labels)
            if task.state == STATE_REJECTED and not force:
                task.state = STATE_DRAFT  # the annotator is reworking a rejected frame
            self.tasks.renew_lease(session, task, auth.user_id)
            task.updated_at = utcnow()

            if current is None:
                current = Annotation(task_id=task.id, anno_id=anno_id)
                session.add(current)
            current.version = new_version
            current.author_id = auth.user_id
            current.path = self.openlist.anno_path(project, anno_id)
            current.labels_json = json.dumps(labels, ensure_ascii=False)
            current.content_hash = content_hash
            current.updated_at = utcnow()
            session.add(
                AnnotationVersion(
                    task_id=task.id,
                    version=new_version,
                    author_id=auth.user_id,
                    path=self.openlist.history_path(project, anno_id, new_version),
                    labels_json=json.dumps(labels, ensure_ascii=False),
                    content_hash=content_hash,
                    note=note,
                )
            )
            audit.record(
                session,
                action="save_annotation",
                user_id=auth.user_id,
                target_type="task",
                target_id=anno_id,
                detail={"version": new_version, "labels": labels, "force": force, "hash": content_hash[:12]},
            )
            state = task.state
            updated_at = current.updated_at
        logger.info(f"annotation saved anno_id={anno_id} version={new_version} labels={labels}")
        return SaveResult(
            version=new_version, state=state, updated_at=updated_at, content_hash=content_hash, labels=labels
        )

    def _write_files(self, project: str, anno_id: str, payload: bytes, version: int, token: str) -> None:
        """History copy first, then the live document (a failure keeps the DB clean)."""
        history = self.openlist.history_path(project, anno_id, version)
        self.openlist.ensure_dir(history.rsplit("/", 1)[0], token)
        self.openlist.put_bytes(history, payload, token)
        self.openlist.put_bytes(self.openlist.anno_path(project, anno_id), payload, token)

    @staticmethod
    def _conflict_detail(session: OrmSession, row: Annotation | None) -> dict[str, Any]:
        if row is None:
            return {"server_version": 0}
        author = session.get(User, row.author_id) if row.author_id else None
        return {
            "server_version": int(row.version),
            "updated_by": author.name if author is not None else "",
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            "hash": row.content_hash or "",
        }

    @staticmethod
    def _user_names(session: OrmSession, user_ids: list[int | None]) -> dict[int, str]:
        wanted = {uid for uid in user_ids if uid}
        if not wanted:
            return {}
        return dict(session.execute(select(User.id, User.name).where(User.id.in_(wanted))).all())

    # endregion
