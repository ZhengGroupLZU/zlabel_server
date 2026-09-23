"""Reading and writing annotation documents.

The document itself stays a plain JSON file in the storage tree
(``<project>/.zlabel/annos/<anno_id>.zlabel``) so mirrors and other tooling keep
working; the DB holds the metadata that makes concurrent editing safe:

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

from app.adapters.storage import StorageBackend
from app.core.config import Settings
from app.core.errors import Conflict, NotFound, ValidationFailed
from app.core.logging import get_logger
from app.db.base import Database
from app.db.models import (
    STATE_APPROVED,
    STATE_DRAFT,
    STATE_REJECTED,
    STATE_SUBMITTED,
    STATES,
    Annotation,
    AnnotationVersion,
    Project,
    Task,
    User,
    utcnow,
)
from app.services import audit
from app.services.auth_service import AuthContext
from app.services.instance_service import InstanceService
from app.services.project_service import ProjectService
from app.services.task_service import TaskService

logger = get_logger("zlabel.app.annotations")


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
        storage: StorageBackend,
        projects: ProjectService,
        tasks: TaskService,
        instances: InstanceService,
        settings: Settings,
    ) -> None:
        self.db = db
        self.storage = storage
        self.projects = projects
        self.tasks = tasks
        self.instances = instances
        self.settings = settings

    # region reads
    def get(self, project: str, anno_id: str) -> tuple[bytes, int]:
        """Current document + version; 404 when this task is not annotated yet."""
        content = self.storage.get_bytes(self.storage.anno_path(project, anno_id))
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

    def get_version(self, project: str, anno_id: str, version: int) -> bytes:
        """Historical document, or the current one when the versions match."""
        current_bytes, current_version = self.get(project, anno_id)
        if version <= 0 or version == current_version:
            return current_bytes
        path = self.storage.history_path(project, anno_id, version)
        return self.storage.get_bytes(path)

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

        with self.db.session_scope() as session:
            task = session.scalar(select(Task).where(Task.anno_id == anno_id))
            if task is None:
                raise NotFound(f"unknown task: {anno_id}")
            if task.project.name != project:
                raise NotFound(f"task {anno_id} does not belong to project {project}")

            if task.state == STATE_APPROVED and not force:
                # approved is final: only a reviewer's reopen unlocks it
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
            self._write_files(project, anno_id, payload, new_version)
            task.labels = self.projects.ensure_labels(session, task.project_id, labels)
            # keep the instance registry / links in step with the document
            self.instances.sync_document(session, project_id=task.project_id, task=task, document=document)
            if task.state in (STATE_REJECTED, STATE_SUBMITTED) and not force:
                # reworking a rejected - or already submitted - task drops it back to
                # draft, so the reviewer never approves a version nobody has seen
                task.state = STATE_DRAFT
            self.tasks.renew_lease(session, task, auth.user_id)
            task.updated_at = utcnow()

            if current is None:
                current = Annotation(task_id=task.id, anno_id=anno_id)
                session.add(current)
            current.version = new_version
            current.author_id = auth.user_id
            current.path = self.storage.anno_path(project, anno_id)
            current.labels_json = json.dumps(labels, ensure_ascii=False)
            current.content_hash = content_hash
            current.updated_at = utcnow()
            session.add(
                AnnotationVersion(
                    task_id=task.id,
                    version=new_version,
                    author_id=auth.user_id,
                    path=self.storage.history_path(project, anno_id, new_version),
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

    def import_document(
        self,
        project: str,
        rel_path: str,
        content: bytes,
        *,
        document: dict[str, Any],
        state: str | None = None,
        actor_id: int | None = None,
        note: str = "imported",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Import a document into a task: normalise it, write it, register it.

        The file *name* is the anno id, so the document's embedded ``"id"`` must
        match it: a legacy document is copied with its old ``id`` replaced by the
        task's (everything else is kept verbatim, including unknown fields). The
        current document and its history copy are written, then the DB is brought
        in step: annotation row (a fresh version when the content is new, no-op
        when it was imported already), label registry and instance mirror.
        ``state`` can mark the task as already reviewed (``approved``) for legacy
        datasets that were finished before the server existed.
        """
        with self.db.session_scope() as session:
            task = session.scalar(
                select(Task).join(Project).where(Project.name == project, Task.rel_path == rel_path)
            )
            if task is None:
                raise NotFound(f"no task for {rel_path!r} in {project!r} - scan the project first")
            normalized = document
            if str(document.get("id") or "") != task.anno_id:
                # the id is the identity of the file: rewrite it, keep everything else
                normalized = dict(document)
                normalized["id"] = task.anno_id
                content = json.dumps(normalized, ensure_ascii=False, indent=4).encode("utf-8")
            labels = extract_label_names(normalized)
            content_hash = hashlib.sha256(content).hexdigest()
            current = session.scalar(select(Annotation).where(Annotation.task_id == task.id))
            if current is not None and (current.content_hash or "") == content_hash:
                return {
                    "anno_id": task.anno_id,
                    "version": int(current.version),
                    "labels": json.loads(current.labels_json or "[]"),
                    "imported": False,
                }
            if current is not None and self._same_document_except_id(
                self.storage.anno_path(project, task.anno_id), normalized
            ):
                # the import ran before the id rewrite existed (or files were copied
                # by hand): the annotation itself is identical, so repair the files
                # and hashes in place instead of creating a new version
                path = self.storage.anno_path(project, task.anno_id)
                history = self.storage.history_path(project, task.anno_id, int(current.version))
                if not dry_run:
                    self.storage.put_bytes(path, content)
                    self.storage.put_bytes(history, content)
                    current.content_hash = content_hash
                    current.updated_at = utcnow()
                    version_row = session.scalar(
                        select(AnnotationVersion).where(
                            AnnotationVersion.task_id == task.id,
                            AnnotationVersion.version == int(current.version),
                        )
                    )
                    if version_row is not None:
                        version_row.content_hash = content_hash
                    audit.record(
                        session,
                        action="repair_annotation_id",
                        user_id=actor_id,
                        target_type="task",
                        target_id=task.anno_id,
                        detail={"version": int(current.version), "rel_path": rel_path},
                    )
                logger.info(f"repaired the embedded id of {rel_path!r} (version {current.version})")
                return {
                    "anno_id": task.anno_id,
                    "version": int(current.version),
                    "labels": labels,
                    "imported": False,
                    "repaired": True,
                }

            version = int(current.version) + 1 if current is not None else 1
            if dry_run:
                return {
                    "anno_id": task.anno_id,
                    "version": version,
                    "labels": labels,
                    "imported": True,
                }
            path = self.storage.anno_path(project, task.anno_id)
            history = self.storage.history_path(project, task.anno_id, version)
            history_dir = history.rsplit("/", 1)[0]
            self.storage.ensure_dir(history_dir)
            self.storage.put_bytes(path, content)
            self.storage.put_bytes(history, content)

            task.labels = self.projects.ensure_labels(session, task.project_id, labels)
            self.instances.sync_document(session, project_id=task.project_id, task=task, document=normalized)
            if current is None:
                current = Annotation(task_id=task.id, anno_id=task.anno_id)
                session.add(current)
            current.version = version
            current.author_id = actor_id
            current.path = path
            current.labels_json = json.dumps(labels, ensure_ascii=False)
            current.content_hash = content_hash
            current.updated_at = utcnow()
            session.add(
                AnnotationVersion(
                    task_id=task.id,
                    version=version,
                    author_id=actor_id,
                    path=history,
                    labels_json=json.dumps(labels, ensure_ascii=False),
                    content_hash=content_hash,
                    note=note,
                )
            )
            if state:
                if state not in STATES:
                    raise ValidationFailed(f"unknown state: {state}")
                task.state = state
                if state == STATE_APPROVED:
                    task.reviewed_at = utcnow()
            task.updated_at = utcnow()
            audit.record(
                session,
                action="import_annotation",
                user_id=actor_id,
                target_type="task",
                target_id=task.anno_id,
                detail={"version": version, "labels": labels, "rel_path": rel_path},
            )
        logger.info(f"imported annotation {rel_path!r} (version {version}, labels={labels})")
        return {"anno_id": task.anno_id, "version": version, "labels": labels, "imported": True}

    def _same_document_except_id(self, path: str, document: dict[str, Any]) -> bool:
        """True when the stored document matches ``document`` but for the ``id``."""
        try:
            stored = json.loads(self.storage.get_bytes(path).decode("utf-8"))
        except Exception:  # noqa: BLE001 - missing/unreadable: not the same document
            return False
        if not isinstance(stored, dict):
            return False
        return {k: v for k, v in stored.items() if k != "id"} == {
            k: v for k, v in document.items() if k != "id"
        }

    def _write_files(self, project: str, anno_id: str, payload: bytes, version: int) -> None:
        """History copy first, then the live document (a failure keeps the DB clean)."""
        history = self.storage.history_path(project, anno_id, version)
        self.storage.ensure_dir(history.rsplit("/", 1)[0])
        self.storage.put_bytes(history, payload)
        self.storage.put_bytes(self.storage.anno_path(project, anno_id), payload)

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
