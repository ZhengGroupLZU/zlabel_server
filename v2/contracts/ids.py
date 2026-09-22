"""Identifier contracts shared with the desktop client.

The annotation id must stay bit-identical to the client's
``zlabel.utils.project.anno_id_for`` and to v1's ``app.db.anno_id_for``:
``md5("<project>/<project-relative posix path>")``. Local mirrors and OpenList
files stay interchangeable only as long as this holds.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def id_md5(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def anno_id_for(project_name: str, rel_path: str | Path) -> str:
    """``md5("<project>/<relative posix path>")``.

    ``rel_path`` is project-relative; Windows separators are normalised so both
    platforms (and the desktop client) agree.
    """
    rel = str(rel_path).replace("\\", "/").lstrip("/")
    return id_md5(f"{project_name}/{rel}")
