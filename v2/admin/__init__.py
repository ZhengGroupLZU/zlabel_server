"""Web administration UI (starlette-admin).

Mounted at ``ZLSERVER_ADMIN_PATH`` (default ``/admin``) with cookie sessions from
:class:`ZLabelAuthProvider`. The pages call the same services as the API, so audit
rows and session revocation behave identically whichever surface an admin uses.

    /admin                 dashboard: storage, progress, the project table, rescan
    /admin/users           accounts: filter, create, role/enabled, password reset
    /admin/projects        projects: filter/create; per-project overview (rename,
                           metadata), files (browse/upload/preview), members,
                           labels and frames
    /admin/files           global storage browser
    /admin/audit-log       read-only audit trail
"""

from __future__ import annotations

from starlette_admin.contrib.sqla import Admin

from v2.admin.auth import ZLabelAuthProvider
from v2.admin.views import AuditLogAdmin, DashboardView, FilesView, ProjectsView, UsersView
from v2.core.logging import get_logger
from v2.db.models import AuditLog
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
        # the UI stays in English: the desktop client's i18n is separate
    )
    for view in (
        UsersView(services),
        ProjectsView(services),
        FilesView(services),
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
