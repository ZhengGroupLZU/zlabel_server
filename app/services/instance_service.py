"""Project-scoped instances: the objects a set of annotations belongs to.

An instance is a tracked thing (a seed, a seedling, ...). The annotation documents
carry the truth — ``Result.instance_id`` on each result and ``Annotation.instances``
mapping a number to its status — while this service mirrors them into the
``instances`` / ``instance_results`` tables on every save (the same way labels grow
the per-project label registry), and exposes the CRUD the API and the admin UI use.

``number`` is the id the documents reference: it is unique per project, sorted
ascending and **never renumbered** here.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session as OrmSession

from app.adapters.storage import StorageBackend
from app.core.config import Settings
from app.core.errors import Conflict, NotFound, ValidationFailed
from app.core.logging import get_logger
from app.db.base import Database
from app.db.models import Annotation, Instance, InstanceResult, Project, Task, utcnow
from app.services import audit
from app.services.label_palette import normalize_color, pick_color

logger = get_logger("zlabel.app.instances")

#: statuses the desktop offers by default (GermStatus values + the non-seed tags);
#: the field stays free text, this is only the datalist the admin UI shows
INSTANCE_STATUS_PRESETS: tuple[str, ...] = (
    "normal_seed",
    "moldy_seed",
    "dead_seed",
    "normal_seedling",
    "abnormal_seedling",
    "dish",
    "text",
)

#: a sanity bound for the document's instance id
MAX_INSTANCE_NUMBER = 99999


def parse_instance_annotations(document: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """``{number: {"status": str, "results": [(result_id, label), ...]}}``.

    Reads the desktop's wire format: ``results`` is a mapping (or a list) of result
    documents carrying ``instance_id`` (``0`` = not part of an instance) and
    ``labels[0].name``; ``instances`` maps a number to its per-task status.
    """
    parsed: dict[int, dict[str, Any]] = {}
    results = document.get("results") or {}
    if isinstance(results, dict):
        items = list(results.items())
    elif isinstance(results, list):
        items = [(str((item or {}).get("id") or ""), item) for item in results]
    else:
        items = []
    for result_id, result in items:
        if not isinstance(result, dict):
            continue
        try:
            number = int(result.get("instance_id") or 0)
        except (TypeError, ValueError):
            continue
        if number <= 0 or number > MAX_INSTANCE_NUMBER:
            continue
        label = ""
        labels = result.get("labels") or []
        if labels and isinstance(labels[0], dict):
            label = str(labels[0].get("name") or "").strip()
        entry = parsed.setdefault(number, {"status": "", "results": []})
        entry["results"].append((str(result_id or result.get("id") or ""), label))
    statuses = document.get("instances") or {}
    if isinstance(statuses, dict):
        for raw, status in statuses.items():
            try:
                number = int(raw)
            except (TypeError, ValueError):
                continue
            if number in parsed:
                parsed[number]["status"] = str(status or "").strip()
    return parsed


class InstanceService:
    def __init__(self, db: Database, storage: StorageBackend, projects: Any, settings: Settings) -> None:
        self.db = db
        self.storage = storage
        self.projects = projects  # ProjectService (project lookup + access)
        self.settings = settings

    # region mirror (annotation saves)
    def sync_document(
        self,
        session: OrmSession,
        *,
        project_id: int,
        task: Task,
        document: dict[str, Any],
    ) -> dict[str, int]:
        """Mirror one saved document into ``instances``/``instance_results``.

        Instances are created on first sight (status from the document, colour from
        the palette) and never renumbered; an existing instance keeps its metadata
        (the admin's edits win, an empty status is filled from the document). The
        task's links are rewritten, so removed/renamed results disappear here too.
        """
        parsed = parse_instance_annotations(document)
        rows = {
            row.number: row
            for row in session.scalars(select(Instance).where(Instance.project_id == project_id)).all()
        }
        used_colors = {row.color for row in rows.values() if row.color}
        created = 0
        for number, entry in sorted(parsed.items()):
            instance = rows.get(number)
            if instance is None:
                color = pick_color(used_colors)
                instance = Instance(
                    project_id=project_id,
                    number=number,
                    status=str(entry["status"] or ""),
                    color=color,
                )
                session.add(instance)
                session.flush()  # assigns instance.id for the links below
                rows[number] = instance
                used_colors.add(color)
                created += 1
            elif not instance.status and entry["status"]:
                instance.status = str(entry["status"])

        session.execute(delete(InstanceResult).where(InstanceResult.task_id == task.id))
        linked = 0
        for number, entry in parsed.items():
            instance = rows.get(number)
            if instance is None:
                continue
            for result_id, label in entry["results"]:
                if not result_id:
                    continue
                session.add(
                    InstanceResult(instance_id=instance.id, task_id=task.id, result_id=result_id, label=label)
                )
                linked += 1
        return {"instances": created, "results": linked}

    def sync_project(self, project: str) -> dict[str, int]:
        """Re-mirror every stored annotation of a project (backfill/repair).

        Used by ``app.cli sync-instances`` for datasets annotated before the
        instance tables existed, and after a manual key migration.
        """
        import json

        stats = {"documents": 0, "instances": 0, "results": 0}
        with self.db.session_scope() as session:
            row = session.scalar(select(Project).where(Project.name == project))
            if row is None:
                raise NotFound(f"unknown project: {project}")
            project_id = row.id
            tasks = list(session.scalars(select(Task).where(Task.project_id == project_id)).all())
            for task in tasks:
                annotation = session.scalar(select(Annotation).where(Annotation.task_id == task.id))
                if annotation is None:
                    continue
                try:
                    content = self.storage.get_bytes(self.storage.anno_path(project, task.anno_id))
                    document = json.loads(content.decode("utf-8"))
                except Exception as e:  # noqa: BLE001 - one broken file must not stop the run
                    logger.warning(f"cannot read {task.rel_path!r}: {e}")
                    continue
                if not isinstance(document, dict):
                    continue
                outcome = self.sync_document(session, project_id=project_id, task=task, document=document)
                stats["documents"] += 1
                stats["instances"] += outcome["instances"]
                stats["results"] += outcome["results"]
        return stats

    # endregion

    # region reads
    def list_instances(
        self, project: str, *, include_archived: bool = False, with_stats: bool = True
    ) -> list[dict[str, Any]]:
        with self.db.session_scope() as session:
            project_id = self._project_id(session, project)
            query = select(Instance).where(Instance.project_id == project_id).order_by(Instance.number)
            if not include_archived:
                query = query.where(Instance.archived.is_(False))
            rows = list(session.scalars(query).all())
            stats = self._stats(session, [row.id for row in rows]) if with_stats else {}
        return [self._out(row, stats.get(row.id, {})) for row in rows]

    def results_of(self, project: str, number: int) -> list[dict[str, Any]]:
        """The annotations belonging to one instance (task + result + label)."""
        with self.db.session_scope() as session:
            instance = self._instance(session, project, number)
            rows = session.scalars(
                select(InstanceResult)
                .where(InstanceResult.instance_id == instance.id)
                .order_by(InstanceResult.task_id, InstanceResult.id)
            ).all()
            return [
                {
                    "task_id": row.task_id,
                    "anno_id": row.task.anno_id,
                    "rel_path": row.task.rel_path,
                    "result_id": row.result_id,
                    "label": row.label,
                }
                for row in rows
            ]

    def status_presets(self, project: str) -> list[str]:
        """The known statuses: the presets plus everything the project already uses."""
        with self.db.session_scope() as session:
            project_id = self._project_id(session, project)
            used = {
                str(status)
                for (status,) in session.execute(
                    select(Instance.status).where(Instance.project_id == project_id, Instance.status != "")
                ).all()
                if status
            }
        return list(INSTANCE_STATUS_PRESETS) + sorted(used - set(INSTANCE_STATUS_PRESETS))

    # endregion

    # region writes
    def create_instance(
        self,
        project: str,
        *,
        number: int | None = None,
        name: str = "",
        note: str = "",
        status: str = "",
        color: str = "",
        actor_id: int | None = None,
    ) -> dict[str, Any]:
        with self.db.session_scope() as session:
            project_id = self._project_id(session, project)
            rows = list(session.scalars(select(Instance).where(Instance.project_id == project_id)).all())
            existing = {row.number for row in rows}
            if number is None:
                number = max(existing) + 1 if existing else 1
            number = int(number)
            if number <= 0 or number > MAX_INSTANCE_NUMBER:
                raise ValidationFailed(f"an instance number must be between 1 and {MAX_INSTANCE_NUMBER}")
            if number in existing:
                raise Conflict(f"instance {number} already exists in {project!r}")
            used_colors = {row.color for row in rows if row.color}
            instance = Instance(
                project_id=project_id,
                number=number,
                name=str(name or "").strip(),
                note=str(note or ""),
                status=str(status or "").strip(),
                color=normalize_color(color) or pick_color(used_colors),
            )
            session.add(instance)
            session.flush()
            audit.record(
                session,
                action="create_instance",
                user_id=actor_id,
                target_type="instance",
                target_id=f"{project}:{number}",
                detail={"number": number, "name": instance.name},
            )
            return self._out(instance, {})

    def save_instances(
        self,
        project: str,
        *,
        edits: dict[int, dict[str, Any]],
        actor_id: int | None = None,
    ) -> dict[str, int]:
        """Apply the whole Instances table in one transaction (admin bulk save)."""
        changed = 0
        with self.db.session_scope() as session:
            project_id = self._project_id(session, project)
            rows = {
                row.number: row
                for row in session.scalars(select(Instance).where(Instance.project_id == project_id)).all()
            }
            for number, fields in edits.items():
                instance = rows.get(int(number))
                if instance is None:
                    raise NotFound(f"unknown instance: {number}")
                detail: dict[str, Any] = {}
                for field in ("name", "note", "status"):
                    if field in fields and fields[field] is not None:
                        value = str(fields[field]).strip() if field != "note" else str(fields[field])
                        if value != getattr(instance, field):
                            setattr(instance, field, value)
                            detail[field] = value
                if fields.get("color") is not None:
                    color = normalize_color(fields["color"])
                    if color and color != instance.color:
                        instance.color = color
                        detail["color"] = color
                if fields.get("archived") is not None and bool(fields["archived"]) != bool(instance.archived):
                    instance.archived = bool(fields["archived"])
                    detail["archived"] = bool(instance.archived)
                if detail:
                    changed += 1
                    instance.updated_at = utcnow()
                    audit.record(
                        session,
                        action="update_instance",
                        user_id=actor_id,
                        target_type="instance",
                        target_id=f"{project}:{instance.number}",
                        detail={"number": instance.number, **detail},
                    )
        return {"changed": changed}

    def update_instance(
        self,
        project: str,
        number: int,
        *,
        name: str | None = None,
        note: str | None = None,
        status: str | None = None,
        color: str | None = None,
        archived: bool | None = None,
        actor_id: int | None = None,
    ) -> dict[str, Any]:
        self.save_instances(
            project,
            edits={
                number: {
                    "name": name,
                    "note": note,
                    "status": status,
                    "color": color,
                    "archived": archived,
                }
            },
            actor_id=actor_id,
        )
        with self.db.session_scope() as session:
            return self._out(self._instance(session, project, number), {})

    def delete_instance(self, project: str, number: int, *, actor_id: int | None = None) -> None:
        """Drop the registry row and its links; the documents keep their numbers."""
        with self.db.session_scope() as session:
            instance = self._instance(session, project, number)
            links = int(
                session.scalar(
                    select(func.count())
                    .select_from(InstanceResult)
                    .where(InstanceResult.instance_id == instance.id)
                )
                or 0
            )
            audit.record(
                session,
                action="delete_instance",
                user_id=actor_id,
                target_type="instance",
                target_id=f"{project}:{number}",
                detail={"number": number, "results": links},
            )
            session.delete(instance)

    # endregion

    # region internals
    @staticmethod
    def _out(instance: Instance, stats: dict[str, Any]) -> dict[str, Any]:
        return {
            "number": instance.number,
            "name": instance.name or "",
            "note": instance.note or "",
            "status": instance.status or "",
            "color": instance.color or "#000000",
            "archived": bool(instance.archived),
            "results": int(stats.get("results", 0)),
            "tasks": int(stats.get("tasks", 0)),
            "created_at": instance.created_at,
            "updated_at": instance.updated_at,
        }

    @staticmethod
    def _stats(session: OrmSession, instance_ids: list[int]) -> dict[int, dict[str, int]]:
        if not instance_ids:
            return {}
        rows = session.execute(
            select(
                InstanceResult.instance_id,
                func.count(InstanceResult.id),
                func.count(func.distinct(InstanceResult.task_id)),
            )
            .where(InstanceResult.instance_id.in_(instance_ids))
            .group_by(InstanceResult.instance_id)
        ).all()
        return {int(iid): {"results": int(n), "tasks": int(tasks)} for iid, n, tasks in rows}

    def _project_id(self, session: OrmSession, name: str) -> int:
        project_id = session.scalar(select(Project.id).where(Project.name == name))
        if project_id is None:
            raise NotFound(f"unknown project: {name}")
        return int(project_id)

    def _instance(self, session: OrmSession, project: str, number: int) -> Instance:
        project_id = self._project_id(session, project)
        instance = session.scalar(
            select(Instance).where(Instance.project_id == project_id, Instance.number == int(number))
        )
        if instance is None:
            raise NotFound(f"unknown instance: {number}")
        return instance

    # endregion
