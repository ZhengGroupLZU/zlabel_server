"""Admin UI views: a small dashboard plus the model browsers.

The auto-CRUD views cover the metadata tables; anything that has side effects
outside the database (creating a project directory, hashing a password, uploading a
frame) is deliberately **not** offered here — those go through the API/CLI, which is
also what the desktop uses. Read-only views exist for inspection.
"""

from __future__ import annotations

from typing import Any

from starlette.requests import Request
from starlette_admin.contrib.sqla import ModelView
from starlette_admin.views import CustomView
from starlette_admin.widgets import HtmlWidget

from v2.core.logging import get_logger
from v2.db.models import AuditLog, Label, Project, ProjectMember, Task, User
from v2.services.container import Services

logger = get_logger("zlabel.v2.admin")

ROLES = ("annotator", "reviewer", "admin")
STATES = ("draft", "submitted", "approved", "rejected")


def _escape(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


class DashboardView(CustomView):
    """Landing page: storage, projects, task states and the last audit entries."""

    def __init__(self, services: Services) -> None:
        # route_name must stay "index": starlette-admin derives the admin index URL from it
        super().__init__(
            menu_label="Dashboard",
            icon="fa-solid fa-gauge",
            path="/",
            route_name="index",
            widget=self.build_widget,  # a callable: rebuilt on every request
        )
        self.services = services

    async def build_widget(self, request: Request) -> HtmlWidget:  # pragma: no cover - thin HTML
        return HtmlWidget(self._cards())

    def _cards(self) -> str:
        storage = self.services.openlist
        usage = storage.usage() if hasattr(storage, "usage") else None
        projects = self.services.projects.list_projects(active_only=False)
        progress: dict[str, int] = {}
        try:
            progress = self.services.projects.progress()
        except Exception as e:  # noqa: BLE001 - the dashboard must never 500
            logger.warning(f"dashboard progress failed: {e}")

        parts = ['<div class="row row-cards">']
        cards = [
            ("Projects", f"{len(projects)} ({sum(1 for p in projects if p.active)} active)"),
            ("Frames", str(progress.get("total", 0))),
            (
                "Reviewed",
                " · ".join(f"{state} {progress.get(state, 0)}" for state in STATES),
            ),
            (
                "Storage",
                (
                    f"{usage['files']} files, {usage['bytes'] / 1024**2:.1f} MiB<br><code>{_escape(storage.root_dir)}</code>"
                    if usage
                    else f"backend: {_escape(getattr(storage, 'kind', '?'))}"
                ),
            ),
        ]
        for title, value in cards:
            parts.append(
                '<div class="col-sm-6 col-lg-3"><div class="card"><div class="card-body">'
                f'<div class="text-muted">{_escape(title)}</div>'
                f'<div class="h2 m-0">{value}</div>'
                "</div></div></div>"
            )
        parts.append("</div>")
        return "".join(parts)


class UserAdmin(ModelView):
    """Accounts: role and enable/disable. Passwords are not edited here."""

    exclude_fields_from_list = (User.password_hash,)
    exclude_fields_from_detail = (User.password_hash,)
    exclude_fields_from_create = (User.password_hash,)
    exclude_fields_from_edit = (User.password_hash,)
    fields = (User.id, User.name, User.role, User.email, User.active, User.created_at, User.last_login_at)
    searchable_fields = (User.name, User.email)
    sortable_fields = (User.id, User.name, User.role, User.created_at, User.last_login_at)
    page_size = 50

    async def can_create(
        self, request: Request
    ) -> bool:  # creation hashes a password: use the CLI / POST /api/v2/admin/users
        return False

    async def can_delete(self, request: Request) -> bool:  # disable instead (keeps task attribution)
        return False

    label = "Users"
    icon = "fa-solid fa-users"


class ProjectAdmin(ModelView):
    """Project metadata. Creation needs an OpenList/storage directory, so it is API/CLI only."""

    fields = (
        Project.id,
        Project.name,
        Project.display_name,
        Project.description,
        Project.active,
        Project.created_at,
    )
    searchable_fields = (Project.name, Project.display_name)
    sortable_fields = (Project.id, Project.name, Project.active, Project.created_at)

    async def can_create(self, request: Request) -> bool:
        return False

    async def can_delete(self, request: Request) -> bool:
        return False

    label = "Projects"
    icon = "fa-solid fa-folder-tree"


class ProjectMemberAdmin(ModelView):
    """Who works on which project (P4)."""

    fields = (
        ProjectMember.id,
        ProjectMember.project,
        ProjectMember.user,
        ProjectMember.role,
        ProjectMember.created_at,
    )
    sortable_fields = (ProjectMember.id, ProjectMember.created_at)
    label = "Members"
    icon = "fa-solid fa-user-group"


class LabelAdmin(ModelView):
    """The per-project label registry."""

    fields = (Label.id, Label.project, Label.name, Label.color, Label.sort, Label.archived)
    searchable_fields = (Label.name,)
    sortable_fields = (Label.id, Label.name, Label.sort)
    label = "Labels"
    icon = "fa-solid fa-tags"


class AuditLogAdmin(ModelView):
    """Read-only audit trail: claim/submit/review/save/label/role actions."""

    fields = (
        AuditLog.id,
        AuditLog.ts,
        AuditLog.user,
        AuditLog.action,
        AuditLog.target_type,
        AuditLog.target_id,
        AuditLog.detail_json,
    )
    sortable_fields = (AuditLog.id, AuditLog.ts, AuditLog.action)
    searchable_fields = (AuditLog.action, AuditLog.target_id)
    page_size = 100

    async def can_create(self, request: Request) -> bool:
        return False

    async def can_edit(self, request: Request) -> bool:
        return False

    async def can_delete(self, request: Request) -> bool:
        return False

    label = "Audit"
    icon = "fa-solid fa-clipboard-list"


class TaskAdmin(ModelView):
    """Read-only view of the task table (state/claim/lease per frame)."""

    fields = (
        Task.id,
        Task.project,
        Task.rel_path,
        Task.group_name,
        Task.day,
        Task.state,
        Task.claimed_by,
        Task.lease_expires_at,
        Task.submitted_at,
        Task.reviewed_at,
        Task.missing,
    )
    sortable_fields = (Task.id, Task.rel_path, Task.state, Task.updated_at)
    searchable_fields = (Task.rel_path, Task.anno_id)
    page_size = 50

    async def can_create(self, request: Request) -> bool:
        return False

    async def can_edit(self, request: Request) -> bool:
        return False

    async def can_delete(self, request: Request) -> bool:
        return False

    label = "Frames"
    icon = "fa-solid fa-images"
