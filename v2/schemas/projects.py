"""Project / label / progress models."""

from __future__ import annotations

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
    progress: ProgressOut | None = None

    @classmethod
    def of(cls, project, progress: dict | None = None) -> ProjectOut:
        return cls(
            id=int(getattr(project, "id", 0) or 0),
            name=project.name,
            display_name=project.display_name or project.name,
            description=project.description or "",
            active=bool(project.active),
            progress=ProgressOut.of(progress) if progress else None,
        )


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    display_name: str = ""


class ProjectPatch(BaseModel):
    display_name: str | None = None
    description: str | None = None
    active: bool | None = None


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
    color: str = "#000000"
    sort: int = 0


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
