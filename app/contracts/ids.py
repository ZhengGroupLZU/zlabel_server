"""Identifier contracts shared with the desktop client.

The annotation id is ``sha256("<project key>/<project-relative posix path>")``.

The *project key* is the project's stable id, not its name: the server keeps it in
``projects.key`` and mirrors it into ``<project>/.zlabel/project.json`` under
``"id"`` — the same value the desktop stores as ``Project.id``. Renaming the
directory (or the display name) therefore never invalidates annotations, and a
dataset can move between the desktop's local mode and the server unchanged.

``legacy_anno_id_for`` is the pre-key formula (``md5("<project name>/<rel>")``),
kept only so ``app.cli migrate-anno-ids`` can find files written by older releases.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def id_md5(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def id_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _relative(rel_path: str | Path) -> str:
    return str(rel_path).replace("\\", "/").lstrip("/")


def anno_id_for(project_key: str, rel_path: str | Path) -> str:
    """``sha256("<project key>/<relative posix path>")``.

    ``rel_path`` is project-relative; Windows separators are normalised so both
    platforms (and the desktop client) agree.
    """
    return id_sha256(f"{project_key}/{_relative(rel_path)}")


def legacy_anno_id_for(project_name: str, rel_path: str | Path) -> str:
    """The historical ``md5("<project name>/<rel>")`` id (migration only)."""
    return id_md5(f"{project_name}/{_relative(rel_path)}")
