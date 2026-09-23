"""Server configuration.

Every env var uses the ``ZLSERVER_`` prefix and the optional ``.env.v2`` file
(kept separate from the deleted v1 ``.env.onnx`` so a stale file is never read by
accident). The API and the inference worker share the prefix; each process reads
the fields it knows and ignores the rest.

The database is a **fresh file** (``data/zlabel_server_v2.db``): v1 data was not
migrated and old projects are intentionally not preserved.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ZLabel Server"
    version: str = "2.0.0"

    # --- storage -----------------------------------------------------------
    database_url: str = "./data/zlabel_server_v2.db"
    upload_dir: str = "./data/uploads"
    # the server owns its datasets as a plain directory tree (mount/NAS paths are fine)
    storage_root: str = "./data/storage"
    # optional: create this admin account at startup when it does not exist yet
    bootstrap_password: str = ""
    # --- web administration UI ---------------------------------------------
    admin_enabled: bool = True
    admin_path: str = "/admin"
    # signs the admin UI's own cookies; set a long random value in production
    secret_key: str = ""

    # project access: "open" = any account may work on any project (pre-P4
    # behaviour), "strict" = only `project_members` (global admins always see all)
    project_access_mode: Literal["open", "strict"] = "open"
    # Where annotations live inside a project directory. Empty = ".zlabel/annos"
    # (the same layout as the desktop's dataset mode, so one directory can be used
    # by both sides). Old deployments moving off the historical `<project>/zlabel`
    # layout run `python -m app.cli migrate-layout` and set this explicitly.
    anno_dir: str = ""

    # --- auth / sessions ---------------------------------------------------
    session_ttl_days: int = 30
    session_cache_seconds: float = 60.0
    # optional: force this user name (lower-case) to role=admin on login
    bootstrap_admin: str = ""

    # --- task lease --------------------------------------------------------
    lease_minutes: int = 30

    # --- labels ------------------------------------------------------------
    # colour used when an annotation introduces a label the server does not know
    # yet. Empty = pick from the built-in qualitative palette (see
    # `app/services/label_palette.py`); set a colour to force one for every label.
    default_label_color: str = ""

    # --- project discovery ------------------------------------------------
    # Scanning is **manual only**: the admin Projects/Dashboard pages and
    # `POST /projects/scan` trigger it. There is no startup or periodic scan
    # (walking the storage tree is the slowest operation in the system).

    # --- inference service (separate process) ------------------------------
    inference_url: str = "http://127.0.0.1:8001"
    inference_token: str = ""  # shared secret between API and worker (both directions)
    inference_timeout: float = 180.0
    # send the task image inline with every job; False = the worker pulls it from
    # GET /api/v2/internal/images/{sha256} on an embedding miss
    inference_inline_images: bool = True

    # --- images ------------------------------------------------------------
    max_upload_bytes: int = 64 * 1024 * 1024

    model_config = SettingsConfigDict(
        env_prefix="ZLSERVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    #: layout the storage tree defaults to (identical to the desktop datasets)
    DEFAULT_ANNO_DIR: ClassVar[str] = ".zlabel/annos"

    @property
    def anno_dir_clean(self) -> str:
        """The annotation directory as a safe relative POSIX path."""
        configured = str(self.anno_dir or "") or self.DEFAULT_ANNO_DIR
        parts = [p for p in configured.replace("\\", "/").split("/") if p not in ("", ".")]
        if not parts or any(p == ".." for p in parts):
            raise ValueError(f"ZLSERVER_ANNO_DIR is not a safe relative path: {self.anno_dir!r}")
        return "/".join(parts)

    def ensure_dirs(self) -> None:
        """Create the parent directory of the sqlite file, the upload dir and the
        storage root."""
        if self.is_sqlite and ":memory:" not in self.database_url:
            Path(self.database_url.removeprefix("sqlite+pysqlite:///")).parent.mkdir(
                parents=True, exist_ok=True
            )
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.storage_root).expanduser().mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
