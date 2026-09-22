"""Web administration UI (starlette-admin).

Mounted at ``ZLSERVER_ADMIN_PATH`` (default ``/admin``) with cookie sessions from
:class:`ZLabelAuthProvider`; everything the API exposes is still the source of truth
for side effects (project creation, uploads, password hashing), so this layer can be
replaced page by page later without touching the services.

    /admin            dashboard (storage, projects, states, audit peek)
    /admin/user       accounts (role, enable/disable; no password editing)
    /admin/project    project metadata
    /admin/projectmember  membership (P4)
    /admin/label      label registry
    /admin/task       read-only frame/task inspection
    /admin/auditlog   read-only audit trail
"""

from __future__ import annotations

from starlette_admin.contrib.sqla import Admin

from v2.admin.auth import ZLabelAuthProvider
from v2.admin.views import (
    AuditLogAdmin,
    DashboardView,
    LabelAdmin,
    ProjectAdmin,
    ProjectMemberAdmin,
    TaskAdmin,
    UserAdmin,
)
from v2.core.logging import get_logger
from v2.db.models import AuditLog, Label, Project, ProjectMember, Task, User
from v2.services.container import Services

logger = get_logger("zlabel.v2.admin")


def build_admin(services: Services) -> Admin:
    """The admin application (not mounted yet)."""
    settings = services.settings
    admin = Admin(
        services.db.engine,
        title=f"{settings.app_name} admin",
        base_url=settings.admin_path,
        auth_provider=ZLabelAuthProvider(services, base_url=settings.admin_path),
        index_view=DashboardView(services),
        secret_key=settings.secret_key or "zlabel-admin-dev-secret",
        # keep the UI in the admins' language; the client UI is separate
    )
    for view in (
        UserAdmin(User, menu_label="Users"),
        ProjectAdmin(Project, menu_label="Projects"),
        ProjectMemberAdmin(ProjectMember, menu_label="Members"),
        LabelAdmin(Label, menu_label="Labels"),
        TaskAdmin(Task, menu_label="Frames"),
        AuditLogAdmin(AuditLog, menu_label="Audit log"),
    ):
        admin.add_view(view)
    return admin


def mount_admin(app, services: Services) -> None:
    """Attach the admin UI to the FastAPI app when it is enabled."""
    if not services.settings.admin_enabled:
        logger.info("admin UI disabled (ZLSERVER_ADMIN_ENABLED=false)")
        return
    admin = build_admin(services)
    admin.mount_to(app)
    logger.info(f"admin UI mounted at {services.settings.admin_path} (admins only)")
