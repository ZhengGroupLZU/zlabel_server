"""The service container: one instance per app, built in ``create_app``.

Routers get it through the ``get_services`` dependency; tests replace the
OpenList adapter (or the whole container) instead of monkeypatching globals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from v2.adapters.identity import build_identity
from v2.adapters.inference import InferenceClient
from v2.adapters.storage import build_storage
from v2.core.config import Settings
from v2.db.base import Database
from v2.services.annotation_service import AnnotationService
from v2.services.auth_service import AuthService
from v2.services.image_store import ImageStore
from v2.services.project_service import ProjectService
from v2.services.task_service import TaskService


@dataclass
class Services:
    settings: Settings
    db: Database
    openlist: Any  # StorageBackend (OpenList or the local disk)
    auth: AuthService = field(init=False)
    projects: ProjectService = field(init=False)
    tasks: TaskService = field(init=False)
    annotations: AnnotationService = field(init=False)
    images: ImageStore = field(init=False)
    inference: InferenceClient = field(init=False)

    def __post_init__(self) -> None:
        # replaced by ``build`` (the identity provider needs the storage backend)
        self.auth = None  # type: ignore[assignment]
        self.projects = ProjectService(self.db, self.openlist, self.settings)
        self.tasks = TaskService(self.db, self.settings)
        self.annotations = AnnotationService(self.db, self.openlist, self.projects, self.tasks, self.settings)
        self.images = ImageStore(self.settings)
        self.inference = InferenceClient(self.settings)

    @classmethod
    def build(
        cls,
        settings: Settings,
        db: Database,
        *,
        openlist: Any = None,
        identity: str | None = None,
    ) -> Services:
        storage = openlist or build_storage(settings)
        provider = build_identity(settings, db=db, storage=storage, identity=identity)
        services = cls(settings=settings, db=db, openlist=storage)
        services.auth = AuthService(db, provider, settings)
        return services
