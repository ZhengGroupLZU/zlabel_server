"""v2 configuration.

Env vars use the ``ZLV2_`` prefix (optional ``.env.v2`` file) so a v2 process can
run next to the v1 server (``ZLSERVER_``) on the same host — that is required for
the M4 dual-run comparison.

The v2 database is a **separate, fresh file**: v2 neither migrates nor imports v1
data (old projects are intentionally not preserved).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ZLabel Server"
    version: str = "2.0.0"

    # --- storage -----------------------------------------------------------
    database_url: str = "./data/zlabel_server_v2.db"
    upload_dir: str = "./data/uploads"

    # --- auth / sessions ---------------------------------------------------
    session_ttl_days: int = 30
    session_cache_seconds: float = 60.0
    # optional: force this user name (lower-case) to role=admin on login
    bootstrap_admin: str = ""

    # --- task lease --------------------------------------------------------
    lease_minutes: int = 30

    # --- labels ------------------------------------------------------------
    # colour used when an annotation introduces a label the server does not know yet
    default_label_color: str = "#000000"

    # --- OpenList ----------------------------------------------------------
    oplist_host: str = "http://127.0.0.1:5244"
    oplist_username: str = ""
    oplist_password: str = ""
    oplist_token: str = ""  # static service token used by the background scanner
    oplist_proj_dir: str = "/zlabel_server/projects"
    oplist_proj_name: str = ""
    project_marker: str = ".zlabel-server-project-root"
    # scan OpenList into the task table once at startup / every N seconds (<=0 = off)
    scan_on_startup: bool = True
    project_scan_interval: int = 300

    # --- inference service (separate process) ------------------------------
    inference_url: str = "http://127.0.0.1:8001"
    inference_token: str = ""  # shared secret between API and worker (both directions)
    inference_timeout: float = 180.0
    # send the frame inline with every job; False = the worker pulls it from
    # GET /api/v2/internal/images/{sha256} on an embedding miss
    inference_inline_images: bool = True

    # --- images ------------------------------------------------------------
    max_upload_bytes: int = 64 * 1024 * 1024

    model_config = SettingsConfigDict(
        env_prefix="ZLV2_",
        env_file=".env.v2",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    def ensure_dirs(self) -> None:
        """Create the parent directory of the sqlite file and the upload dir."""
        if self.is_sqlite and ":memory:" not in self.database_url:
            Path(self.database_url.removeprefix("sqlite+pysqlite:///")).parent.mkdir(
                parents=True, exist_ok=True
            )
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
