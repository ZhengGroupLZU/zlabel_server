"""ASGI entrypoint: ``uv run fastapi run v2/main.py`` (port 8000 by default)."""

from __future__ import annotations

from v2.app import create_app

app = create_app()
