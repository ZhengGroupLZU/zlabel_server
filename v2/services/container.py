"""The service container: one instance per app, built in ``create_app``.

Routers get it through the ``get_services`` dependency; tests replace the storage
backend (or the whole container) instead of monkeypatching globals.
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
from v2.services.instance_service import InstanceService
from v2.services.project_service import ProjectService
from v2.services.task_service import TaskService


@dataclass
class Services:
    settings: Settings
    db: Database
    storage: Any  # StorageBackend (always the local disk today)
    auth: AuthService = field(init=False)
    projects: ProjectService = field(init=False)
    instances: InstanceService = field(init=False)
    tasks: TaskService = field(init=False)
    annotations: AnnotationService = field(init=False)
    images: ImageStore = field(init=False)
    inference: InferenceClient = field(init=False)

    def __post_init__(self) -> None:
        # replaced by ``build`` (the identity provider is constructed there)
        self.auth = None  # type: ignore[assignment]
        self.projects = ProjectService(self.db, self.storage, self.settings)
        self.instances = InstanceService(self.db, self.storage, self.projects, self.settings)
        self.tasks = TaskService(self.db, self.settings)
        self.annotations = AnnotationService(
            self.db, self.storage, self.projects, self.tasks, self.instances, self.settings
        )
        self.images = ImageStore(self.settings)
        self.inference = InferenceClient(self.settings)

    @classmethod
    def build(cls, settings: Settings, db: Database, *, storage: Any = None) -> Services:
        backend = storage or build_storage(settings)
        provider = build_identity(settings, db=db)
        services = cls(settings=settings, db=db, storage=backend)
        services.auth = AuthService(db, provider, settings)
        return services
