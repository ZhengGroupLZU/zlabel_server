"""Task / workflow models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TaskOut(BaseModel):
    anno_id: str
    project: str
    rel_path: str
    group: str = ""
    day: int = 0
    state: str
    missing: bool = False
    has_annotation: bool = False
    version: int = 0

    claimed_by: str = ""
    claimed_at: datetime | None = None
    lease_expires_at: datetime | None = None
    submitted_at: datetime | None = None
    reviewed_by: str = ""
    reviewed_at: datetime | None = None
    review_note: str = ""
    labels: list[str] = Field(default_factory=list)
    updated_at: datetime | None = None

    @classmethod
    def of(cls, row) -> TaskOut:
        task = row.task
        return cls(
            anno_id=task.anno_id,
            project=row.project,
            rel_path=task.rel_path,
            group=task.group_name,
            day=task.day,
            state=task.state,
            missing=bool(task.missing),
            has_annotation=row.has_annotation,
            version=row.version,
            claimed_by=row.holder,
            claimed_at=task.claimed_at,
            lease_expires_at=task.lease_expires_at,
            submitted_at=task.submitted_at,
            reviewed_by=row.reviewer,
            reviewed_at=task.reviewed_at,
            review_note=task.review_note or "",
            labels=sorted(label.name for label in task.labels),
            updated_at=task.updated_at,
        )


class TaskListOut(BaseModel):
    items: list[TaskOut] = Field(default_factory=list)
    total: int = 0


class GroupOut(BaseModel):
    group: str
    count: int = 0
    tasks: list[TaskOut] = Field(default_factory=list)


class ReviewRequest(BaseModel):
    decision: Literal["approve", "reject"]
    note: str = ""


class ReopenRequest(BaseModel):
    note: str = ""
