"""Auto-assigned label colours.

A curated qualitative palette beats random RGB: every entry is legible as a swatch
and the colours stay distinguishable from each other (Tableau-10 plus a few
Okabe-Ito hues). ``pick_color`` starts at a random offset — so two projects do not
look alike — and then takes the first colour the project has not used yet.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable

#: 16 distinguishable, medium-dark colours (readable on a white background)
LABEL_PALETTE: tuple[str, ...] = (
    "#e6194b",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#008080",  # teal
    "#b8860b",  # dark goldenrod
    "#1f77b4",  # tableau blue
    "#2ca02c",  # tableau green
    "#d62728",  # tableau red
    "#9467bd",  # tableau purple
    "#8c564b",  # tableau brown
    "#e377c2",  # tableau pink
    "#17becf",  # cyan
    "#bcbd22",  # olive
    "#7f7f7f",  # grey
)

_HEX_RE = re.compile(r"^[0-9a-f]{6}$")


def normalize_color(value: str | None) -> str:
    """``"#AABBCC"``/``"abc"``/``"#abc"`` → ``"#aabbcc"``; junk becomes ``""``.

    Operators type hex codes by hand, so the shorthand and a missing ``#`` are
    accepted (``"fff"`` → ``"#ffffff"``).
    """
    clean = str(value or "").strip().lower().lstrip("#")
    if re.fullmatch(r"[0-9a-f]{3}", clean):
        clean = "".join(character * 2 for character in clean)
    return f"#{clean}" if _HEX_RE.match(clean) else ""


def pick_color(used: Iterable[str] = ()) -> str:
    """A palette colour the project has not used yet (random start, then first free)."""
    taken = {normalize_color(color) for color in used}
    taken.discard("")
    start = random.randrange(len(LABEL_PALETTE))
    for offset in range(len(LABEL_PALETTE)):
        color = LABEL_PALETTE[(start + offset) % len(LABEL_PALETTE)]
        if color not in taken:
            return color
    return random.choice(LABEL_PALETTE)  # every palette colour is in use
