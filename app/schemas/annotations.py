"""Annotation read/write models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SaveResponse(BaseModel):
    anno_id: str
    version: int
    state: str
    updated_at: datetime
    content_hash: str
    labels: list[str] = Field(default_factory=list)

    @classmethod
    def of(cls, anno_id: str, result) -> SaveResponse:
        return cls(
            anno_id=anno_id,
            version=result.version,
            state=result.state,
            updated_at=result.updated_at,
            content_hash=result.content_hash,
            labels=result.labels,
        )


class VersionOut(BaseModel):
    version: int
    author: str = ""
    created_at: datetime | None = None
    note: str = ""
    content_hash: str = ""
    labels: list[str] = Field(default_factory=list)

    @classmethod
    def of(cls, row) -> VersionOut:
        return cls(
            version=row.version,
            author=row.author,
            created_at=row.created_at,
            note=row.note,
            content_hash=row.content_hash,
            labels=row.labels,
        )
