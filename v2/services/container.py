"""The service container: one instance per app, built in ``create_app``.

Routers get it through the ``get_services`` dependency; tests replace the
OpenList adapter (or the whole container) instead of monkeypatching globals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from v2.adapters.openlist import OpenListAdapter
from v2.core.config import Settings
from v2.db.base import Database
from v2.services.auth_service import AuthService
from v2.services.project_service import ProjectService


@dataclass
class Services:
    settings: Settings
    db: Database
    openlist: OpenListAdapter
    auth: AuthService = field(init=False)
    projects: ProjectService = field(init=False)

    def __post_init__(self) -> None:
        self.auth = AuthService(self.db, self.openlist, self.settings)
        self.projects = ProjectService(self.db, self.openlist, self.settings)

    @classmethod
    def build(cls, settings: Settings, db: Database, *, openlist: OpenListAdapter | None = None) -> Services:
        return cls(settings=settings, db=db, openlist=openlist or OpenListAdapter(settings))
