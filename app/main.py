"""ASGI entrypoint: ``uv run fastapi run app/main.py`` (port 8000 by default)."""

from __future__ import annotations

from app.app import create_app

app = create_app()
