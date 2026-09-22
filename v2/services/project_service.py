"""Projects, their task table, labels and progress.

Discovery mirrors v1 (a top-level dir is a project only when it carries the
marker file) but the DB work is a plain upsert: tasks are keyed by the shared
``anno_id`` and rows that vanished from OpenList are flagged ``missing`` (kept for
history, hidden from the listings).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session as OrmSession

from v2.adapters.openlist import OpenListAdapter
from v2.contracts.ids import anno_id_for
from v2.core.config import Settings
from v2.core.errors import ApiError, Conflict, NotFound
from v2.core.logging import get_logger
from v2.db.base import Database
from v2.db.models import (
    STATE_APPROVED,
    STATES,
    Label,
    Project,
    Task,
    User,
    utcnow,
)
from v2.services import audit
from v2.services.grouping import parse_group

logger = get_logger("zlabel.v2.projects")

SCAN_THROTTLE_SECONDS = 2.0


@dataclass
class ScanOutcome:
    projects: list[dict[str, Any]] = field(default_factory=list)
    present_dirs: set[str] = field(default_factory=set)
    confirmed_missing: set[str] = field(default_factory=set)


class ProjectService:
    def __init__(self, db: Database, openlist: OpenListAdapter, settings: Settings) -> None:
        self.db = db
        self.openlist = openlist
        self.settings = settings
        self._lock = threading.Lock()
        self._last_scan = 0.0

    # region scanning
    def scan(self) -> ScanOutcome:
        """Walk OpenList: every top-level dir, marker probe, images inside."""
        token = self.openlist.service_token()
        present = set(self.openlist.list_dirs(self.openlist.root, token))
        outcome = ScanOutcome(present_dirs=present)
        for name in sorted(present):
            status = self._marker_status(name, token)
            if status is True:
                files = self.openlist.glob_images(self.openlist.project_dir(name), token)
                outcome.projects.append({"name": name, "files": files})
            elif status is False:
                outcome.confirmed_missing.add(name)
            # status is None (could not inspect) -> do not deactivate anything
        return outcome

    def scan_and_sync(self, *, force: bool = False) -> dict[str, int]:
        """Scan + persist, throttled so client-side polling cannot hammer OpenList."""
        with self._lock:
            if not force and time.monotonic() - self._last_scan < SCAN_THROTTLE_SECONDS:
                return {"projects": 0, "tasks": 0, "missing": 0, "deactivated": 0, "skipped": 1}
            outcome = self.scan()
            self._last_scan = time.monotonic()
        stats = self.sync(outcome)
        stats["skipped"] = 0
        logger.info(f"scan done: {stats}")
        return stats

    def sync(self, outcome: ScanOutcome) -> dict[str, int]:
        stats = {"projects": 0, "tasks": 0, "missing": 0, "deactivated": 0}
        with self.db.session_scope() as session:
            for entry in outcome.projects:
                name = entry["name"]
                project = self._upsert_project(session, name)
                stats["projects"] += 1
                pairs = [
                    (rel, path) for path, rel in ((p, self._rel_path(name, p)) for p in entry["files"]) if rel
                ]
                stats["tasks"] += self._upsert_tasks(session, project, pairs)
                stats["missing"] += self._mark_missing(session, project, {rel for rel, _ in pairs})
            stats["deactivated"] = self._deactivate(session, outcome)
        return stats

    def _marker_status(self, name: str, token: str) -> bool | None:
        """True = project, False = confirmed no marker, None = could not inspect."""
        try:
            self.openlist.file_info(self.openlist.marker_path(name), token)
            return True
        except NotFound:
            return False
        except ApiError as e:
            logger.warning(f"marker probe failed for {name!r}: {e}")
            return None

    def _rel_path(self, project: str, abs_path: str) -> str:
        prefix = self.openlist.project_dir(project).rstrip("/") + "/"
        if not abs_path.startswith(prefix):
            return ""
        return abs_path[len(prefix) :]

    def _upsert_project(self, session: OrmSession, name: str) -> Project:
        project = session.scalar(select(Project).where(Project.name == name))
        if project is None:
            project = Project(name=name, display_name=name)
            session.add(project)
            session.flush()
        else:
            project.active = True
        return project

    def _upsert_tasks(self, session: OrmSession, project: Project, pairs: list[tuple[str, str]]) -> int:
        existing = {t.rel_path: t for t in session.scalars(select(Task).where(Task.project_id == project.id))}
        for rel, abs_path in pairs:
            group, day = parse_group(rel)
            task = existing.get(rel)
            if task is None:
                session.add(
                    Task(
                        project_id=project.id,
                        anno_id=anno_id_for(project.name, rel),
                        path=abs_path,
                        rel_path=rel,
                        group_name=group,
                        day=day,
                    )
                )
            else:
                task.path = abs_path
                task.group_name = group
                task.day = day
                task.missing = False
        return len(pairs)

    def _mark_missing(self, session: OrmSession, project: Project, seen: set[str]) -> int:
        count = 0
        for task in session.scalars(
            select(Task).where(Task.project_id == project.id, Task.missing.is_(False))
        ):
            if task.rel_path not in seen:
                task.missing = True
                count += 1
        return count

    def _deactivate(self, session: OrmSession, outcome: ScanOutcome) -> int:
        count = 0
        for project in session.scalars(select(Project).where(Project.active.is_(True))):
            if project.name in outcome.confirmed_missing or project.name not in outcome.present_dirs:
                project.active = False
                count += 1
        return count

    # endregion

    # region projects
    def list_projects(self, *, active_only: bool = True) -> list[Project]:
        with self.db.session_scope() as session:
            query = select(Project).order_by(Project.name)
            if active_only:
                query = query.where(Project.active.is_(True))
            return list(session.scalars(query).all())

    def get_project(self, name: str, *, active_only: bool = True) -> Project:
        with self.db.session_scope() as session:
            query = select(Project).where(Project.name == name)
            if active_only:
                query = query.where(Project.active.is_(True))
            project = session.scalar(query)
            if project is None:
                raise NotFound(f"unknown project: {name}")
            return project

    def create_project(self, name: str, display_name: str = "", *, actor_id: int | None = None) -> Project:
        """Create the OpenList directory (with the marker file) and the DB row."""
        clean = name.strip().strip("/")
        if not clean or "/" in clean:
            raise Conflict("a project name must be a single path segment")
        token = self.openlist.service_token()
        directory = self.openlist.project_dir(clean)
        if self.openlist.exists(directory, token):
            raise Conflict(f"project already exists: {clean}")
        self.openlist.ensure_dir(directory, token)
        self.openlist.put_bytes(self.openlist.marker_path(clean), b"zlabel project root\n", token)
        with self.db.session_scope() as session:
            project = Project(name=clean, display_name=display_name or clean)
            session.add(project)
            session.flush()
            audit.record(
                session,
                action="create_project",
                user_id=actor_id,
                target_type="project",
                target_id=clean,
                detail={"display_name": project.display_name},
            )
            return project

    def update_project(
        self,
        name: str,
        *,
        display_name: str | None = None,
        description: str | None = None,
        active: bool | None = None,
        actor_id: int | None = None,
    ) -> Project:
        with self.db.session_scope() as session:
            project = session.scalar(select(Project).where(Project.name == name))
            if project is None:
                raise NotFound(f"unknown project: {name}")
            if display_name is not None:
                project.display_name = display_name
            if description is not None:
                project.description = description
            if active is not None:
                project.active = active
            project.updated_at = utcnow()
            audit.record(
                session,
                action="update_project",
                user_id=actor_id,
                target_type="project",
                target_id=name,
                detail={"display_name": display_name, "description": description, "active": active},
            )
            return project

    # endregion

    # region progress
    def progress(self, project: str | None = None, *, by_user: bool = False) -> dict[str, Any]:
        """Task counts per state; ``by_user`` adds a per-annotator breakdown."""
        with self.db.session_scope() as session:
            base = (
                select(Task.state, func.count())
                .join(Project, Project.id == Task.project_id)
                .where(Project.active.is_(True), Task.missing.is_(False))
                .group_by(Task.state)
            )
            if project:
                base = base.where(Project.name == project)
            counts = dict.fromkeys(STATES, 0)
            for state, count in session.execute(base).all():
                counts[state] = int(count)
            result: dict[str, Any] = {
                "total": sum(counts.values()),
                **counts,
                "finished": counts[STATE_APPROVED],  # convenience for the status bar
            }
            if by_user:
                rows = session.execute(
                    select(User.name, Task.state, func.count())
                    .join(Task, Task.claimed_by == User.id)
                    .join(Project, Project.id == Task.project_id)
                    .where(Project.active.is_(True), Task.missing.is_(False))
                    .group_by(User.name, Task.state)
                ).all()
                per_user: dict[str, dict[str, int]] = {}
                for user_name, state, count in rows:
                    per_user.setdefault(user_name, dict.fromkeys(STATES, 0))[state] = int(count)
                result["by_user"] = per_user
            return result

    # endregion

    # region labels
    def list_labels(self, project: str, *, include_archived: bool = False) -> list[Label]:
        with self.db.session_scope() as session:
            return self._labels(session, self._project_id(session, project), include_archived)

    def create_label(
        self,
        project: str,
        name: str,
        *,
        color: str = "#000000",
        sort: int = 0,
        actor_id: int | None = None,
    ) -> Label:
        clean = name.strip()
        if not clean:
            raise Conflict("a label needs a name")
        with self.db.session_scope() as session:
            project_id = self._project_id(session, project)
            existing = session.scalar(
                select(Label).where(Label.project_id == project_id, func.lower(Label.name) == clean.lower())
            )
            if existing is not None:
                if existing.archived:
                    existing.archived = False
                    existing.color = color or existing.color
                    return existing
                raise Conflict(f"label already exists: {clean}")
            label = Label(project_id=project_id, name=clean, color=color, sort=sort)
            session.add(label)
            session.flush()
            audit.record(
                session,
                action="create_label",
                user_id=actor_id,
                target_type="label",
                target_id=label.id,
                detail={"project": project, "name": clean},
            )
            return label

    def update_label(
        self,
        project: str,
        label_id: int,
        *,
        name: str | None = None,
        color: str | None = None,
        sort: int | None = None,
        archived: bool | None = None,
        actor_id: int | None = None,
    ) -> Label:
        with self.db.session_scope() as session:
            project_id = self._project_id(session, project)
            label = session.scalar(select(Label).where(Label.id == label_id, Label.project_id == project_id))
            if label is None:
                raise NotFound(f"unknown label: {label_id}")
            if name is not None:
                label.name = name.strip()
            if color is not None:
                label.color = color
            if sort is not None:
                label.sort = sort
            if archived is not None:
                label.archived = archived
            audit.record(
                session,
                action="update_label",
                user_id=actor_id,
                target_type="label",
                target_id=label.id,
                detail={"project": project, "name": label.name, "archived": label.archived},
            )
            return label

    def delete_label(self, project: str, label_id: int, *, actor_id: int | None = None) -> None:
        with self.db.session_scope() as session:
            project_id = self._project_id(session, project)
            label = session.scalar(select(Label).where(Label.id == label_id, Label.project_id == project_id))
            if label is None:
                raise NotFound(f"unknown label: {label_id}")
            audit.record(
                session,
                action="delete_label",
                user_id=actor_id,
                target_type="label",
                target_id=label.id,
                detail={"project": project, "name": label.name},
            )
            session.delete(label)

    def ensure_labels(self, session: OrmSession, project_id: int, names: list[str]) -> list[Label]:
        """Get-or-create labels by name (annotation saves grow the registry)."""
        found: list[Label] = []
        for raw in names:
            name = (raw or "").strip()
            if not name:
                continue
            label = session.scalar(
                select(Label).where(Label.project_id == project_id, func.lower(Label.name) == name.lower())
            )
            if label is None:
                label = Label(project_id=project_id, name=name, color=self.settings.default_label_color)
                session.add(label)
                session.flush()
            if label not in found:
                found.append(label)
        return found

    def _labels(self, session: OrmSession, project_id: int, include_archived: bool = False) -> list[Label]:
        query = select(Label).where(Label.project_id == project_id).order_by(Label.sort, Label.name)
        if not include_archived:
            query = query.where(Label.archived.is_(False))
        return list(session.scalars(query).all())

    def _project_id(self, session: OrmSession, name: str) -> int:
        project_id = session.scalar(select(Project.id).where(Project.name == name))
        if project_id is None:
            raise NotFound(f"unknown project: {name}")
        return int(project_id)

    # endregion
