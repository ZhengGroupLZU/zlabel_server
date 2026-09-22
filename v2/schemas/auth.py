"""Auth request/response models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)
    client: str = ""


class UserOut(BaseModel):
    id: int
    name: str
    role: str
    email: str = ""
    finished_count: int = 0

    @classmethod
    def of(cls, user) -> UserOut:
        return cls(
            id=user.id,
            name=user.name,
            role=user.role,
            email=user.email or "",
            finished_count=user.finished_count or 0,
        )


class LoginResponse(BaseModel):
    token: str
    expires_at: datetime
    user: UserOut
