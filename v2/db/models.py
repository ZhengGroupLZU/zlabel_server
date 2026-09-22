"""v2 schema.

Fresh database: v2 does not migrate or import v1 data. ``tasks.anno_id`` keeps the
formula shared with the desktop client — ``md5("<project>/<project-relative posix
path>")`` — and annotation *files* still live in OpenList, so history stored on
disk stays interchangeable.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from v2.db.base import Base

ROLE_ANNOTATOR = "annotator"
ROLE_REVIEWER = "reviewer"
ROLE_ADMIN = "admin"
ROLES = (ROLE_ANNOTATOR, ROLE_REVIEWER, ROLE_ADMIN)

STATE_DRAFT = "draft"
STATE_SUBMITTED = "submitted"
STATE_APPROVED = "approved"
STATE_REJECTED = "rejected"
STATES = (STATE_DRAFT, STATE_SUBMITTED, STATE_APPROVED, STATE_REJECTED)


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# association tables (kept for the v1 compat reads: labels/users per task)
link_task_label = Table(
    "link_task_label",
    Base.metadata,
    Column("task_id", ForeignKey("tasks.id", ondelete="CASCADE"), primary_key=True),
    Column("label_id", ForeignKey("labels.id", ondelete="CASCADE"), primary_key=True),
)

link_task_user = Table(
    "link_task_user",
    Base.metadata,
    Column("task_id", ForeignKey("tasks.id", ondelete="CASCADE"), primary_key=True),
    Column("user_id", ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    oplist_user_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)  # lower-cased
    email: Mapped[str] = mapped_column(String(256), default="")
    role: Mapped[str] = mapped_column(String(16), default=ROLE_ANNOTATOR)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    # scrypt hash for local accounts ("" while OpenList still owns the identity)
    password_hash: Mapped[str] = mapped_column(String(255), default="")
    finished_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)

    @property
    def is_reviewer(self) -> bool:
        return self.role in (ROLE_REVIEWER, ROLE_ADMIN)

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN


class Session(Base):
    """A server-issued session; the OpenList token lives here for file access."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    oplist_token: Mapped[str] = mapped_column(String(512), default="")
    client_info: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)

    user: Mapped[User] = relationship(lazy="joined")


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(255), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class Label(Base):
    __tablename__ = "labels"
    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_label_project_name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    color: Mapped[str] = mapped_column(String(16), default="#000000")
    sort: Mapped[int] = mapped_column(Integer, default=0)
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Task(Base):
    """One labelable image. ``state`` drives the review workflow, the claim trio
    (``claimed_by/claimed_at/lease_expires_at``) makes concurrent annotators safe."""

    __tablename__ = "tasks"
    __table_args__ = (
        Index("ix_tasks_project_state", "project_id", "state"),
        Index("ix_tasks_project_group_day", "project_id", "group_name", "day"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    anno_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # ``path`` is the absolute OpenList path, ``rel_path`` the project-relative one
    path: Mapped[str] = mapped_column(String(1024))
    rel_path: Mapped[str] = mapped_column(String(1024), default="")
    group_name: Mapped[str] = mapped_column(String(512), default="")
    day: Mapped[int] = mapped_column(Integer, default=0)
    state: Mapped[str] = mapped_column(String(16), default=STATE_DRAFT)
    missing: Mapped[bool] = mapped_column(Boolean, default=False)

    claimed_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), default=None, index=True
    )
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime, default=None, index=True)

    submitted_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    reviewed_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), default=None)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    review_note: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    project: Mapped[Project] = relationship(lazy="joined")
    claimer: Mapped[User | None] = relationship(foreign_keys=[claimed_by], lazy="joined")
    reviewer: Mapped[User | None] = relationship(foreign_keys=[reviewed_by], lazy="joined")
    labels: Mapped[list[Label]] = relationship(secondary=link_task_label, lazy="selectin")
    users: Mapped[list[User]] = relationship(secondary=link_task_user, lazy="selectin")


class Annotation(Base):
    """Metadata of the annotation currently stored in OpenList for a task."""

    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), unique=True)
    anno_id: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer, default=0)
    author_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), default=None)
    path: Mapped[str] = mapped_column(String(1024), default="")
    labels_json: Mapped[str] = mapped_column(Text, default="[]")
    content_hash: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    author: Mapped[User | None] = relationship(lazy="joined")


class AnnotationVersion(Base):
    """Append-only history of saved annotation versions."""

    __tablename__ = "annotation_versions"
    __table_args__ = (UniqueConstraint("task_id", "version", name="uq_version_task_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    author_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), default=None)
    path: Mapped[str] = mapped_column(String(1024), default="")
    labels_json: Mapped[str] = mapped_column(Text, default="[]")
    content_hash: Mapped[str] = mapped_column(String(64), default="")
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    author: Mapped[User | None] = relationship(lazy="joined")


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), default=None)
    action: Mapped[str] = mapped_column(String(64), index=True)
    target_type: Mapped[str] = mapped_column(String(32), default="")
    target_id: Mapped[str] = mapped_column(String(128), default="")
    detail_json: Mapped[str] = mapped_column(Text, default="{}")

    user: Mapped[User | None] = relationship(lazy="joined")


class ProjectMember(Base):
    """A user's role *inside one project* (P4).

    Project visibility/participation is membership-driven in "strict" access mode;
    a global admin always has access, and "open" mode keeps the pre-P4 behaviour
    (any account may work on any project) so an existing deployment is unaffected.
    """

    __tablename__ = "project_members"
    __table_args__ = (UniqueConstraint("project_id", "user_id", name="uq_member_project_user"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16), default=ROLE_ANNOTATOR)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    project: Mapped[Project] = relationship(lazy="joined")
    user: Mapped[User] = relationship(lazy="joined")
