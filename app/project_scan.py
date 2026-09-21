"""Automatic project discovery from OpenList.

A *top-level* directory under ``SETTINGS.oplist_proj_dir`` is treated as a
project only if it contains a hidden marker file (see
``SETTINGS.project_marker``, default ``.zlabel-server-project-root``). Images
are collected recursively inside each project directory.

The discovery is deliberately tolerant: it never raises for an individual
directory. It records, for each top-level directory, whether the marker is
present (project), confirmed absent (not a project), or could not be inspected
(unknown). This lets the DB sync deactivate only directories we are *sure* no
longer qualify, while leaving unknown ones untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import requests

from app.config import Settings
from app.openlist_api import OpenListAPIError, OpenListClient

# image extensions that count as labelable tasks
ALLOWED_IMAGE_EXT: tuple[str, ...] = (".png", ".jpg", ".jpeg")


@dataclass
class ScanResult:
    """Outcome of one project scan.

    Attributes:
        projects: confirmed projects, each ``{"name": str, "files": list[str]}``.
        present_dirs: every top-level directory name under ``oplist_proj_dir``.
        confirmed_missing: directories present in ``present_dirs`` but whose
            marker was confirmed absent (candidates for deactivation).
    """

    projects: list[dict[str, Any]] = field(default_factory=list)
    present_dirs: set[str] = field(default_factory=set)
    confirmed_missing: set[str] = field(default_factory=set)


def discover_projects(
    client: OpenListClient, settings: Settings
) -> ScanResult:
    """Discover projects under ``settings.oplist_proj_dir``.

    Only top-level directories are inspected; a directory is a project iff it
    contains ``settings.project_marker``. Within a project, image files are
    collected recursively.

    Raises:
        Exception: if the root directory listing itself fails (caller should
            treat this as "scan unavailable" and skip any DB sync).
    """
    proj_dir = settings.oplist_proj_dir.rstrip("/")
    dirs_resp = client.fs.dirs(proj_dir)
    present_dirs: set[str] = {d.name for d in dirs_resp.data}
    projects: list[dict[str, Any]] = []
    confirmed_missing: set[str] = set()

    for name in present_dirs:
        project_dir = f"{proj_dir}/{name}"
        status = _marker_status(client, settings, project_dir)
        if status is True:
            files = [
                f
                for f in client.fs.glob(project_dir, "*")
                if f.lower().endswith(ALLOWED_IMAGE_EXT)
            ]
            projects.append({"name": name, "files": files})
        elif status is False:
            confirmed_missing.add(name)
        # status is None (could not inspect) -> leave the project as-is in the
        # DB; we cannot assert that it no longer qualifies.

    return ScanResult(
        projects=projects,
        present_dirs=present_dirs,
        confirmed_missing=confirmed_missing,
    )


def _marker_status(
    client: OpenListClient, settings: Settings, project_dir: str
) -> bool | None:
    """Report whether ``project_dir`` contains the project marker.

    We probe the marker path directly (``fs.get``) instead of relying on a
    directory listing, because OpenList may filter dotfiles out of ``ls`` output
    (the marker is a hidden file).

    Returns:
        True if the marker is present, False if it is confirmed absent (404),
        or None if it could not be inspected (do not deactivate on None).
    """
    marker_path = f"{project_dir}/{settings.project_marker}"
    try:
        client.fs.get(marker_path)
        return True
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else None
        return False if code == 404 else None
    except OpenListAPIError as e:
        return False if e.status_code == 404 else None
    except Exception:
        return None
