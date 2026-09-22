"""Sequence grouping, parsed by the server so every client agrees.

Rules (identical to the desktop client's previous filename heuristic, but applied
to the *project-relative* path):

- ``.../species/dish/D{n}.png``  → group ``species/dish``, day ``n``
  (``dish/D{n}.png`` → group ``dish``)
- otherwise ``name_001.jpg`` style → group = the prefix, day = the number
- neither → ``("", 0)``
"""

from __future__ import annotations

import re

DAY_FILE_RE = re.compile(r"D(\d+)\.(?:png|jpe?g)", re.IGNORECASE)
TAIL_RE = re.compile(r"(.+?)[_\-\s]*(\d+)\.(?:png|jpe?g)", re.IGNORECASE)


def parse_group(rel_path: str) -> tuple[str, int]:
    """``(group_name, day)`` for a project-relative image path."""
    normalized = rel_path.replace("\\", "/").lstrip("/")
    parts = [p for p in normalized.split("/") if p]
    if not parts:
        return "", 0

    match = DAY_FILE_RE.fullmatch(parts[-1])
    if match is not None:
        day = int(match.group(1))
        if len(parts) >= 3:
            return f"{parts[-3]}/{parts[-2]}", day
        if len(parts) == 2:
            return parts[-2], day

    tail = TAIL_RE.fullmatch(normalized)
    if tail is not None:
        return tail.group(1), int(tail.group(2))
    return "", 0
