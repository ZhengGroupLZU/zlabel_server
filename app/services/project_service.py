"""Projects, their task table, labels and progress.

The server owns the storage tree, so **every top-level directory is a project**;
the DB work is a plain upsert: tasks are keyed by the shared ``anno_id`` and rows
that vanished from the tree are flagged ``missing`` (kept for history, hidden from
the listings).
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session as OrmSession

from app.adapters.storage import StorageBackend
from app.contracts.ids import anno_id_for
from app.core.config import Settings
from app.core.errors import Conflict, Forbidden, NotFound, ValidationFailed
from app.core.logging import get_logger
from app.db.base import Database
from app.db.models import (
    ROLE_ADMIN,
    ROLE_REVIEWER,
    ROLES,
    STATE_APPROVED,
    STATES,
    Annotation,
    AnnotationVersion,
    Label,
    Project,
    ProjectMember,
    Task,
    User,
    link_task_label,
    link_task_user,
    utcnow,
)
from app.services import audit
from app.services.auth_service import AuthContext
from app.services.grouping import parse_group
from app.services.label_palette import normalize_color, pick_color

logger = get_logger("zlabel.app.projects")

SCAN_THROTTLE_SECONDS = 2.0

#: metadata file every dataset/project carries; ``id`` is the stable project key
PROJECT_JSON = ".zlabel/project.json"


def new_project_key() -> str:
    """A short, unique, filesystem-safe project id (same shape as the client's)."""
    return uuid4().hex[:12]


@dataclass
class ScanOutcome:
    projects: list[dict[str, Any]] = field(default_factory=list)
    present_dirs: set[str] = field(default_factory=set)
    confirmed_missing: set[str] = field(default_factory=set)


class ProjectService:
    def __init__(self, db: Database, storage: StorageBackend, settings: Settings) -> None:
        self.db = db
        self.storage = storage
        self.settings = settings
        self._lock = threading.Lock()
        self._last_scan = 0.0

    # region scanning
    def scan(self) -> ScanOutcome:
        """Walk the storage root: top-level dirs, then the images inside them."""
        present = set(self.storage.list_dirs(self.storage.root))
        outcome = ScanOutcome(present_dirs=present)
        with self.db.session_scope() as session:
            known = dict(session.execute(select(Project.name, Project.key)).all())
        for name in sorted(present):
            files = self.storage.glob_images(self.storage.project_dir(name))
            key = self.ensure_project_key(name, fallback=known.get(name, ""))
            outcome.projects.append({"name": name, "files": files, "key": key})
        return outcome

    def scan_and_sync(self, *, force: bool = False) -> dict[str, int]:
        """Scan + persist, throttled so client-side polling cannot hammer the disk."""
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
                key = str(entry.get("key") or "") or self.ensure_project_key(name)
                if (
                    key
                    and session.scalar(select(Project.id).where(Project.key == key, Project.name != name))
                    is not None
                ):
                    # a copied dataset carries the original id: give this copy its own
                    logger.warning(f"project {name!r} shares a key with another project: re-keying")
                    key = new_project_key()
                    self.write_project_key(name, key)
                project = self._upsert_project(session, name, key)
                stats["projects"] += 1
                pairs = [
                    (rel, path) for path, rel in ((p, self._rel_path(name, p)) for p in entry["files"]) if rel
                ]
                stats["tasks"] += self._upsert_tasks(session, project, pairs)
                stats["missing"] += self._mark_missing(session, project, {rel for rel, _ in pairs})
            stats["deactivated"] = self._deactivate(session, outcome)
        return stats

    def _rel_path(self, project: str, abs_path: str) -> str:
        prefix = self.storage.project_dir(project).rstrip("/") + "/"
        if not abs_path.startswith(prefix):
            return ""
        return abs_path[len(prefix) :]

    # region project key (the anno_id namespace, shared with the desktop)
    def project_json_path(self, project: str) -> str:
        return f"{self.storage.project_dir(project)}/{PROJECT_JSON}"

    def read_project_key(self, project: str) -> str:
        """The ``"id"`` of ``<project>/.zlabel/project.json`` ("" when absent)."""
        try:
            payload = json.loads(self.storage.get_bytes(self.project_json_path(project)).decode("utf-8"))
        except NotFound:
            return ""
        except Exception as e:  # noqa: BLE001 - a broken metadata file must not stop a scan
            logger.warning(f"unreadable project.json for {project!r}: {e}")
            return ""
        if not isinstance(payload, dict):
            return ""
        return str(payload.get("id") or "").strip()

    def write_project_key(self, project: str, key: str) -> None:
        """Merge ``"id": key`` into project.json, keeping every other field."""
        path = self.project_json_path(project)
        payload: dict[str, Any] = {"name": project}
        try:
            loaded = json.loads(self.storage.get_bytes(path).decode("utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except NotFound:
            pass
        except Exception as e:  # noqa: BLE001 - start from a fresh document
            logger.warning(f"rewriting unreadable project.json for {project!r}: {e}")
        payload["id"] = key
        self.storage.put_bytes(path, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))

    def ensure_project_key(self, project: str, fallback: str = "") -> str:
        """Adopt the dataset's project id, restore the database's, or mint one.

        ``fallback`` is the key already stored in the database: if the metadata
        file was deleted we restore *that* key instead of re-keying the project
        (which would orphan every annotation file).
        """
        key = self.read_project_key(project)
        if key:
            return key
        key = fallback or new_project_key()
        self.write_project_key(project, key)
        return key

    # endregion

    # region anno-id migration (legacy md5 -> sha256(project key))
    def _move_file(self, source: str, target: str, *, dry_run: bool, annotation_id: str = "") -> bool:
        """Copy + delete one annotation file; never clobber an existing target.

        ``annotation_id`` rewrites the document's embedded ``"id"``: the file name
        is the anno id, so a re-keyed file whose JSON still carries the old one
        would break the client (``Project.add_annotation`` looks the id up in the
        task table). Everything else in the document is kept verbatim.
        """
        if not self.storage.exists(source):
            return False
        if self.storage.exists(target):
            logger.warning(f"keeping both: {target!r} already exists (from {source!r})")
            return False
        if not dry_run:
            data = self.storage.get_bytes(source)
            if annotation_id:
                data = self._with_annotation_id(data, annotation_id)
            self.storage.put_bytes(target, data)
            self.storage.delete(source)
        return True

    @staticmethod
    def _with_annotation_id(data: bytes, anno_id: str) -> bytes:
        """Point a document's embedded ``"id"`` at its new file name."""
        try:
            payload = json.loads(data.decode("utf-8"))
        except Exception:  # noqa: BLE001 - unreadable/foreign file: keep the bytes
            return data
        if not isinstance(payload, dict) or str(payload.get("id") or "") == anno_id:
            return data
        payload["id"] = anno_id
        return json.dumps(payload, ensure_ascii=False, indent=4).encode("utf-8")

    def migrate_anno_ids(self, project: str, *, dry_run: bool = False) -> dict[str, int]:
        """Re-key one project from the legacy ``md5(name/rel)`` ids to the sha256 ids.

        Every file is located through the *task path* it stores in
        ``image_path`` (or the task row's ``rel_path``), so directories that were
        renamed before the switch are migrated too. Existing targets are never
        overwritten; the DB rows (tasks, annotations, versions) are updated to
        match. ``dry_run`` only reports what would move.
        """
        stats = {"renamed": 0, "tasks": 0, "versions": 0, "skipped": 0}
        zlabel_dir = self.storage.zlabel_dir(project)
        with self.db.session_scope() as session:
            row = session.scalar(select(Project).where(Project.name == project))
            if row is None:
                raise NotFound(f"unknown project: {project}")
            key = self.ensure_project_key(project, fallback=row.key)
            row.key = key

            for task in session.scalars(select(Task).where(Task.project_id == row.id)).all():
                new_id = anno_id_for(key, task.rel_path)
                old_id = task.anno_id
                if old_id == new_id:
                    continue
                if self._move_file(
                    f"{zlabel_dir}/{old_id}.zlabel",
                    f"{zlabel_dir}/{new_id}.zlabel",
                    dry_run=dry_run,
                    annotation_id=new_id,
                ):
                    stats["renamed"] += 1
                for annotation in session.scalars(
                    select(Annotation).where(Annotation.anno_id == old_id)
                ).all():
                    if not dry_run:
                        annotation.anno_id = new_id
                        annotation.path = f"{zlabel_dir}/{new_id}.zlabel"
                for version in session.scalars(
                    select(AnnotationVersion).where(AnnotationVersion.task_id == task.id)
                ).all():
                    old_path = self.storage.history_path(project, old_id, version.version)
                    new_path = self.storage.history_path(project, new_id, version.version)
                    if self._move_file(old_path, new_path, dry_run=dry_run, annotation_id=new_id):
                        stats["renamed"] += 1
                    if not dry_run:
                        version.path = new_path
                    stats["versions"] += 1
                if not dry_run:
                    task.anno_id = new_id
                stats["tasks"] += 1

        # annotations whose task row is gone (or files dropped in by hand): the
        # document names its task, so they can be re-keyed all the same
        for full in self.storage.glob_files(zlabel_dir):
            folder, _, name = full.rpartition("/")
            if folder != zlabel_dir or not name.endswith(".zlabel"):
                continue
            stem = name[: -len(".zlabel")]
            if len(stem) == 64:  # already a sha256 id
                continue
            try:
                payload = json.loads(self.storage.get_bytes(full).decode("utf-8"))
                image_path = str(payload.get("image_path") or "").strip()
            except Exception as e:  # noqa: BLE001 - unreadable file: leave it alone
                logger.warning(f"cannot read {full!r}: {e}")
                stats["skipped"] += 1
                continue
            if not image_path:
                stats["skipped"] += 1
                continue
            new_id = anno_id_for(key, image_path)
            if new_id == stem:
                continue
            if self._move_file(full, f"{zlabel_dir}/{new_id}.zlabel", dry_run=dry_run, annotation_id=new_id):
                stats["renamed"] += 1
        return stats

    # endregion

    def _upsert_project(self, session: OrmSession, name: str, key: str = "") -> Project:
        project = session.scalar(select(Project).where(Project.name == name))
        if project is None:
            project = Project(name=name, display_name=name, key=key)
            session.add(project)
            session.flush()
        else:
            project.active = True
            if key:
                project.key = key
        return project

    def _upsert_tasks(self, session: OrmSession, project: Project, pairs: list[tuple[str, str]]) -> int:
        existing = {t.rel_path: t for t in session.scalars(select(Task).where(Task.project_id == project.id))}
        for rel, abs_path in pairs:
            # only projects that declare a timeline get the sequence parse; other
            # projects keep group_name/day empty (the naming is not a sequence)
            group, day = parse_group(rel) if project.timeline else ("", 0)
            task = existing.get(rel)
            if task is None:
                session.add(
                    Task(
                        project_id=project.id,
                        anno_id=anno_id_for(project.key, rel),
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
    def list_projects(self, *, active_only: bool = True, auth: AuthContext | None = None) -> list[Project]:
        """Projects of the deployment, narrowed to the caller's memberships."""
        with self.db.session_scope() as session:
            query = select(Project).order_by(Project.name)
            if active_only:
                query = query.where(Project.active.is_(True))
            if auth is not None and self.strict_access and not auth.is_admin:
                query = query.join(ProjectMember, ProjectMember.project_id == Project.id).where(
                    ProjectMember.user_id == auth.user_id
                )
            return list(session.scalars(query).all())

    def get_project(self, name: str, *, active_only: bool = True, auth: AuthContext | None = None) -> Project:
        """One project; with ``auth`` the caller's access is enforced."""
        with self.db.session_scope() as session:
            query = select(Project).where(Project.name == name)
            if active_only:
                query = query.where(Project.active.is_(True))
            project = session.scalar(query)
            if project is None:
                raise NotFound(f"unknown project: {name}")
        if auth is not None:
            self.require_access(auth, name)
        return project

    def create_project(
        self,
        name: str,
        display_name: str = "",
        *,
        timeline: bool = True,
        actor_id: int | None = None,
    ) -> Project:
        """Create the project directory (a plain top-level folder) and the DB row.

        The new project gets a fresh key, written to ``.zlabel/project.json`` so the
        desktop adopts the same id: the annotation files both sides read and write
        are addressed by ``sha256("<key>/<rel>")``.
        """
        clean = name.strip().strip("/")
        if not clean or "/" in clean:
            raise Conflict("a project name must be a single path segment")
        directory = self.storage.project_dir(clean)
        if self.storage.is_dir(directory):
            raise Conflict(f"a directory named {clean!r} already exists in the storage root")
        with self.db.session_scope() as session:
            if session.scalar(select(Project.id).where(Project.name == clean)) is not None:
                raise Conflict(f"project already exists: {clean}")
        key = new_project_key()
        self.storage.ensure_dir(directory)
        self.write_project_key(clean, key)
        with self.db.session_scope() as session:
            project = Project(
                name=clean, display_name=display_name or clean, key=key, timeline=bool(timeline)
            )
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

    def project_stats(self) -> dict[str, dict[str, int]]:
        """``{project name: {"tasks": n, "members": m}}`` for the admin pages.

        ``tasks`` counts the non-missing tasks (what the client would list).
        """
        with self.db.session_scope() as session:
            names = {int(pid): name for pid, name in session.execute(select(Project.id, Project.name)).all()}
            tasks = dict(
                session.execute(
                    select(Task.project_id, func.count())
                    .where(Task.missing.is_(False))
                    .group_by(Task.project_id)
                ).all()
            )
            members = dict(
                session.execute(
                    select(ProjectMember.project_id, func.count()).group_by(ProjectMember.project_id)
                ).all()
            )
        return {
            name: {"tasks": int(tasks.get(pid, 0)), "members": int(members.get(pid, 0))}
            for pid, name in names.items()
        }

    def update_project(
        self,
        name: str,
        *,
        display_name: str | None = None,
        description: str | None = None,
        active: bool | None = None,
        timeline: bool | None = None,
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
            recomputed = 0
            if timeline is not None and bool(timeline) != bool(project.timeline):
                project.timeline = bool(timeline)
                # apply the new rule to the existing task table right away, so the
                # admin does not have to trigger a rescan
                recomputed = self._recompute_sequence(session, project)
            project.updated_at = utcnow()
            audit.record(
                session,
                action="update_project",
                user_id=actor_id,
                target_type="project",
                target_id=name,
                detail={
                    "display_name": display_name,
                    "description": description,
                    "active": active,
                    "timeline": timeline,
                    "tasks_recomputed": recomputed,
                },
            )
            return project

    @staticmethod
    def _recompute_sequence(session: OrmSession, project: Project) -> int:
        """Re-parse ``group_name``/``day`` for every task (timeline flag changed)."""
        changed = 0
        for task in session.scalars(select(Task).where(Task.project_id == project.id)).all():
            group, day = parse_group(task.rel_path) if project.timeline else ("", 0)
            if (task.group_name, task.day) != (group, day):
                task.group_name = group
                task.day = day
                changed += 1
        return changed

    # endregion

    # region project deletion
    def delete_project(
        self, name: str, *, delete_files: bool = True, actor_id: int | None = None
    ) -> dict[str, int | bool]:
        """Remove a project and (unless ``delete_files=False``) its directory.

        Deletes the registry rows first-class (tasks, annotations, history, labels,
        members, link rows): SQLite does not enforce ``ON DELETE CASCADE`` by
        default, so the children are removed explicitly, leaf first. The files are
        deleted inside the transaction, so a permission error leaves everything
        untouched. Returns ``{"tasks": n, "files_deleted": bool}``.
        """
        clean = name.strip().strip("/")
        if not clean or "/" in clean or clean in (".", ".."):
            raise ValidationFailed(f"invalid project name: {name!r}")
        directory = self.storage.project_dir(clean)
        files_deleted = False
        with self.db.session_scope() as session:
            project = session.scalar(select(Project).where(Project.name == clean))
            if project is None:
                raise NotFound(f"unknown project: {clean}")
            task_count = int(
                session.scalar(select(func.count()).select_from(Task).where(Task.project_id == project.id))
                or 0
            )
            if delete_files and self.storage.is_dir(directory):
                self.storage.delete(directory)  # raises -> the transaction rolls back
                files_deleted = True
            task_ids = select(Task.id).where(Task.project_id == project.id).scalar_subquery()
            session.execute(delete(link_task_label).where(link_task_label.c.task_id.in_(task_ids)))
            session.execute(delete(link_task_user).where(link_task_user.c.task_id.in_(task_ids)))
            session.execute(delete(AnnotationVersion).where(AnnotationVersion.task_id.in_(task_ids)))
            session.execute(delete(Annotation).where(Annotation.task_id.in_(task_ids)))
            session.execute(delete(Task).where(Task.project_id == project.id))
            session.execute(delete(Label).where(Label.project_id == project.id))
            session.execute(delete(ProjectMember).where(ProjectMember.project_id == project.id))
            audit.record(
                session,
                action="delete_project",
                user_id=actor_id,
                target_type="project",
                target_id=clean,
                detail={"tasks": task_count, "files_deleted": files_deleted},
            )
            session.delete(project)
        logger.info(f"deleted project {clean!r}: {task_count} task(s), files={files_deleted}")
        return {"tasks": task_count, "files_deleted": files_deleted}

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
        color: str = "",
        sort: int | None = None,
        actor_id: int | None = None,
    ) -> Label:
        clean = name.strip()
        if not clean:
            raise Conflict("a label needs a name")
        with self.db.session_scope() as session:
            project_id = self._project_id(session, project)
            wanted = normalize_color(color) or self.settings.default_label_color
            existing = session.scalar(
                select(Label).where(Label.project_id == project_id, func.lower(Label.name) == clean.lower())
            )
            if existing is not None:
                if existing.archived:
                    existing.archived = False
                    existing.color = wanted or existing.color
                    return existing
                raise Conflict(f"label already exists: {clean}")
            if sort is None:
                # append: the displayed ordinal id is the position, so a new label
                # must not share sort=0 with the first one
                highest = session.scalar(select(func.max(Label.sort)).where(Label.project_id == project_id))
                sort = int(highest) + 1 if highest is not None else 0
            label = Label(
                project_id=project_id,
                name=clean,
                color=wanted or pick_color(self._used_label_colors(session, project_id)),
                sort=sort,
            )
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
            wanted = normalize_color(color)
            if wanted:
                label.color = wanted
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
        """Get-or-create labels by name (annotation saves grow the registry).

        New labels get an unused palette colour (or ``default_label_color`` when the
        deployment forces one), so a task that introduces labels produces a
        readable, distinct legend instead of a wall of black.
        """
        found: list[Label] = []
        used = self._used_label_colors(session, project_id)
        for raw in names:
            name = (raw or "").strip()
            if not name:
                continue
            label = session.scalar(
                select(Label).where(Label.project_id == project_id, func.lower(Label.name) == name.lower())
            )
            if label is None:
                color = self.settings.default_label_color or pick_color(used)
                label = Label(project_id=project_id, name=name, color=color)
                session.add(label)
                session.flush()
                used.add(color)
            if label not in found:
                found.append(label)
        return found

    def save_labels(
        self,
        project: str,
        *,
        edits: dict[int, dict[str, Any]],
        order: list[int] | None = None,
        actor_id: int | None = None,
    ) -> dict[str, Any]:
        """Apply the whole Labels page in one transaction: field edits + the order.

        ``edits`` maps a label id to the submitted ``name``/``color``/``archived``;
        ``order`` is the wanted id sequence and must list every label exactly once
        (``None``/empty keeps the current order, so a no-JS save still works). Only
        real changes are written; each one gets its own ``update_label`` audit row,
        plus a single ``reorder_labels`` row when the order moved.
        """
        with self.db.session_scope() as session:
            project_id = self._project_id(session, project)
            labels = list(session.scalars(select(Label).where(Label.project_id == project_id)).all())
            by_id = {label.id: label for label in labels}
            if order:
                if len(order) != len(labels) or set(order) != set(by_id):
                    raise ValidationFailed("the order must list every label of the project exactly once")
            else:
                order = [label.id for label in sorted(labels, key=lambda item: (item.sort, item.id))]

            changed = 0
            for label_id, fields in edits.items():
                label = by_id.get(label_id)
                if label is None:
                    raise ValidationFailed(f"unknown label: {label_id}")
                detail: dict[str, Any] = {}
                name = str(fields.get("name") or "").strip()
                if name and name != label.name:
                    label.name = name
                    detail["name"] = name
                color = normalize_color(fields.get("color"))
                if color and color != label.color:
                    label.color = color
                    detail["color"] = color
                archived = bool(fields.get("archived"))
                if archived != bool(label.archived):
                    label.archived = archived
                    detail["archived"] = archived
                if detail:
                    changed += 1
                    audit.record(
                        session,
                        action="update_label",
                        user_id=actor_id,
                        target_type="label",
                        target_id=label.id,
                        detail={"project": project, **detail},
                    )

            reordered = [by_id[label_id].sort for label_id in order] != list(range(len(order)))
            for index, label_id in enumerate(order):
                by_id[label_id].sort = index
            if reordered:
                audit.record(
                    session,
                    action="reorder_labels",
                    user_id=actor_id,
                    target_type="project",
                    target_id=project,
                    detail={"order": [by_id[label_id].name for label_id in order]},
                )
        return {"changed": changed, "reordered": reordered, "total": len(order)}

    @staticmethod
    def _used_label_colors(session: OrmSession, project_id: int) -> set[str]:
        return {
            str(color).lower()
            for (color,) in session.execute(select(Label.color).where(Label.project_id == project_id)).all()
            if color
        }

    def _labels(self, session: OrmSession, project_id: int, include_archived: bool = False) -> list[Label]:
        query = select(Label).where(Label.project_id == project_id).order_by(Label.sort, Label.id)
        if not include_archived:
            query = query.where(Label.archived.is_(False))
        return list(session.scalars(query).all())

    def reorder_labels(self, project: str, order: list[int], *, actor_id: int | None = None) -> list[Label]:
        """Rewrite the label order: ``sort`` becomes 0..N-1 in the given id order.

        The displayed ordinal id is the position after this rewrite; the database
        primary keys stay untouched (task↔label links and API clients keep working).
        """
        with self.db.session_scope() as session:
            project_id = self._project_id(session, project)
            labels = list(session.scalars(select(Label).where(Label.project_id == project_id)).all())
            by_id = {label.id: label for label in labels}
            if len(order) != len(labels) or set(order) != set(by_id):
                raise ValidationFailed("the order must list every label of the project exactly once")
            for index, label_id in enumerate(order):
                by_id[label_id].sort = index
            audit.record(
                session,
                action="reorder_labels",
                user_id=actor_id,
                target_type="project",
                target_id=project,
                detail={"order": [by_id[label_id].name for label_id in order]},
            )
        return self.list_labels(project, include_archived=True)

    def _project_id(self, session: OrmSession, name: str) -> int:
        project_id = session.scalar(select(Project.id).where(Project.name == name))
        if project_id is None:
            raise NotFound(f"unknown project: {name}")
        return int(project_id)

    # endregion

    # region membership (P4)
    @property
    def strict_access(self) -> bool:
        """True when a project is only visible to its members."""
        return self.settings.project_access_mode == "strict"

    def role_for(self, auth: AuthContext, project: str) -> str:
        """The caller's effective role for one project ("" = no access).

        Global admins always have access; everyone else gets their membership role
        in strict mode, or their global role in open mode (pre-P4 behaviour).
        """
        if auth.is_admin:
            return ROLE_ADMIN
        if not self.strict_access:
            return auth.role
        with self.db.session_scope() as session:
            member = session.scalar(
                select(ProjectMember)
                .join(Project, Project.id == ProjectMember.project_id)
                .where(Project.name == project, ProjectMember.user_id == auth.user_id)
            )
            return member.role if member is not None else ""

    def require_access(self, auth: AuthContext, project: str) -> str:
        """Raise 403 unless the caller may work on ``project``."""
        role = self.role_for(auth, project)
        if not role:
            raise Forbidden(f"you are not a member of project {project!r}")
        return role

    def require_project_reviewer(self, auth: AuthContext, project: str) -> str:
        """Raise 403 unless the caller reviews *this* project."""
        role = self.require_access(auth, project)
        if role not in (ROLE_REVIEWER, ROLE_ADMIN):
            raise Forbidden(f"reviewer role required for project {project!r}")
        return role

    def accessible_projects(self, auth: AuthContext) -> set[str] | None:
        """Project names the caller may see; ``None`` means "every project"."""
        if auth.is_admin or not self.strict_access:
            return None
        with self.db.session_scope() as session:
            rows = session.execute(
                select(Project.name)
                .join(ProjectMember, ProjectMember.project_id == Project.id)
                .where(ProjectMember.user_id == auth.user_id)
            ).all()
            return {name for (name,) in rows}

    def list_members(self, project: str) -> list[dict[str, Any]]:
        with self.db.session_scope() as session:
            project_id = self._project_id(session, project)
            members = session.scalars(
                select(ProjectMember).where(ProjectMember.project_id == project_id).order_by(ProjectMember.id)
            ).all()
            return [
                {"user_id": member.user_id, "name": member.user.name, "role": member.role}
                for member in members
            ]

    def add_member(
        self, project: str, user_id: int, role: str, *, actor_id: int | None = None
    ) -> dict[str, Any]:
        """Add or re-role a member (idempotent)."""
        if role not in ROLES:
            raise ValidationFailed(f"unknown role: {role}")
        with self.db.session_scope() as session:
            project_id = self._project_id(session, project)
            user = session.get(User, user_id)
            if user is None:
                raise NotFound(f"unknown user: {user_id}")
            member = session.scalar(
                select(ProjectMember).where(
                    ProjectMember.project_id == project_id, ProjectMember.user_id == user_id
                )
            )
            if member is None:
                member = ProjectMember(project_id=project_id, user_id=user_id, role=role)
                session.add(member)
            else:
                member.role = role
            session.flush()
            audit.record(
                session,
                action="add_member",
                user_id=actor_id,
                target_type="project",
                target_id=project,
                detail={"member": user.name, "role": role},
            )
            return {"user_id": user.id, "name": user.name, "role": role}

    def remove_member(self, project: str, user_id: int, *, actor_id: int | None = None) -> None:
        """Drop a membership and audit it."""
        with self.db.session_scope() as session:
            project_id = self._project_id(session, project)
            member = session.scalar(
                select(ProjectMember).where(
                    ProjectMember.project_id == project_id, ProjectMember.user_id == user_id
                )
            )
            if member is None:
                raise NotFound(f"user {user_id} is not a member of {project!r}")
            audit.record(
                session,
                action="remove_member",
                user_id=actor_id,
                target_type="project",
                target_id=project,
                detail={"member": member.user.name},
            )
            session.delete(member)

    # endregion
