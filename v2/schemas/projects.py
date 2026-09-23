"""Project / label / instance / progress models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ProgressOut(BaseModel):
    total: int = 0
    draft: int = 0
    submitted: int = 0
    approved: int = 0
    rejected: int = 0
    finished: int = 0  # == approved (status-bar convenience)
    by_user: dict[str, dict[str, int]] | None = None

    @classmethod
    def of(cls, data: dict) -> ProgressOut:
        return cls(**{k: v for k, v in data.items() if k in cls.model_fields})


class ProjectOut(BaseModel):
    id: int = 0
    name: str
    display_name: str = ""
    description: str = ""
    active: bool = True
    #: the tasks form sequences (``group``/``day`` are parsed from the path)
    timeline: bool = True
    progress: ProgressOut | None = None

    @classmethod
    def of(cls, project, progress: dict | None = None) -> ProjectOut:
        return cls(
            id=int(getattr(project, "id", 0) or 0),
            name=project.name,
            display_name=project.display_name or project.name,
            description=project.description or "",
            active=bool(project.active),
            timeline=bool(getattr(project, "timeline", True)),
            progress=ProgressOut.of(progress) if progress else None,
        )


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    display_name: str = ""
    #: off = the tasks do not form sequences (no ``group``/``day`` parsing)
    timeline: bool = True


class ProjectPatch(BaseModel):
    display_name: str | None = None
    description: str | None = None
    active: bool | None = None
    timeline: bool | None = None


class ScanStats(BaseModel):
    projects: int = 0
    tasks: int = 0
    missing: int = 0
    deactivated: int = 0
    skipped: int = 0


class LabelOut(BaseModel):
    id: int
    name: str
    color: str = "#000000"
    sort: int = 0
    archived: bool = False

    @classmethod
    def of(cls, label) -> LabelOut:
        return cls(
            id=label.id, name=label.name, color=label.color, sort=label.sort, archived=bool(label.archived)
        )


class LabelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    #: empty = the server picks an unused colour from the label palette
    color: str = ""
    #: ``None`` = append at the end (the displayed ordinal id is the position)
    sort: int | None = None


class LabelOrder(BaseModel):
    """Every label of the project, in the wanted order (0 = first)."""

    order: list[int]


class InstanceOut(BaseModel):
    """One project-scoped instance (the documents' ``instance_id``)."""

    number: int
    name: str = ""
    note: str = ""
    status: str = ""
    color: str = "#000000"
    archived: bool = False
    results: int = 0
    tasks: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None


class InstanceCreate(BaseModel):
    #: ``None`` = the next free number
    number: int | None = None
    name: str = Field(default="", max_length=120)
    note: str = ""
    status: str = Field(default="", max_length=60)
    color: str = ""


class InstancePatch(BaseModel):
    name: str | None = None
    note: str | None = None
    status: str | None = None
    color: str | None = None
    archived: bool | None = None


class InstanceResultOut(BaseModel):
    """One annotation (a result inside a document) belonging to an instance."""

    task_id: int
    anno_id: str
    rel_path: str
    result_id: str
    label: str = ""


class LabelPatch(BaseModel):
    name: str | None = None
    color: str | None = None
    sort: int | None = None
    archived: bool | None = None


class MemberOut(BaseModel):
    user_id: int
    name: str
    role: str


class MemberCreate(BaseModel):
    user_id: int
    role: str = "annotator"


class MemberPatch(BaseModel):
    role: str
