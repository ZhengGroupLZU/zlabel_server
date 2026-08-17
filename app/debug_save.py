"""Temporary debug hooks: dump pipeline images to a temp dir for monitoring.

REMOVE BEFORE RELEASE. Every hook call writes a numbered PNG into OUT_DIR
(zlabel_debug by default) so the full inference chain can be inspected:
01_recv (image as received) -> 02_pre (after preprocessing) ->
03_raw_mask (model mask output) -> 04_final_mask (upscaled result mask).

The active prompt (points / boxes) is drawn on every saved image; images that were
written before a prompt existed are re-saved with a ``_prompt`` suffix when the
prompt is set.

Only active when ZLSERVER_DEBUG=true/1/yes/on; otherwise all hooks are no-ops.
"""

from __future__ import annotations

import os
import threading
from itertools import count
from pathlib import Path

import cv2
import numpy as np

_ENABLED = os.environ.get("ZLSERVER_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
OUT_DIR = Path("zlabel_debug")
_lock = threading.Lock()
_counter = count(1)
_last: dict[str, tuple[np.ndarray, tuple[float, float], bool]] = {}
_prompt: tuple = (None, None, None)


def _next_id() -> int:
    with _lock:
        return next(_counter)


def _write(tag: str, im: np.ndarray, scale: tuple[float, float], prompted: bool) -> None:
    _last[tag] = (im.copy(), scale, prompted)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_DIR / f"{_next_id():06d}_{tag}.png"), im)


def save_image(tag: str, img: np.ndarray, prompt_scale: tuple[float, float] = (1.0, 1.0)) -> None:
    """Save an image/mask; float [0,1] arrays are scaled to [0,255].

    ``prompt_scale`` maps original-image pixel coords into this image's space
    (stretch scale or letterbox ratio), so the prompt can be drawn correctly.
    """
    if not _ENABLED:
        return
    if img is None or img.size == 0:
        return
    im = img.astype(np.float32, copy=False)
    if im.ndim == 3 and im.shape[2] >= 4:
        im = im[:, :, :3]
    if im.ndim == 2:
        lo, hi = im.min(), im.max()
        if hi - lo > 1e-6:
            im = (im - lo) / (hi - lo) * 255.0
        else:
            im = im * 255.0
    im = np.clip(im, 0, 255).astype(np.uint8)
    prompted = _prompt[0] is not None or _prompt[2] is not None
    if prompted:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        im = _draw_prompt(im, *_prompt, scale=prompt_scale)
    _write(tag, im, prompt_scale, prompted)


def set_prompt(points=None, labels=None, boxes=None) -> None:
    """Record the active prompt and re-save every earlier dump with it drawn.

    Call at the start of each predict. Coordinates are in the original image pixel
    space; each image is re-rendered using its own stored scale.
    """
    if not _ENABLED:
        return
    global _prompt
    _prompt = (points, labels, boxes)
    if _prompt[0] is None and _prompt[2] is None:
        return
    for tag, (im, scale, prompted) in list(_last.items()):
        if prompted:
            continue
        _write(f"{tag}_prompt", _draw_prompt(im, *_prompt, scale=scale), scale, prompted=True)


def _draw_prompt(im: np.ndarray, points, labels, boxes, scale: tuple[float, float]) -> np.ndarray:
    sx, sy = scale
    im = im.copy()
    if points:
        for i, p in enumerate(points):
            color = (0, 255, 255) if labels is None or labels[i] > 0 else (255, 0, 0)
            pt = (int(round(p[0] * sx)), int(round(p[1] * sy)))
            cv2.circle(im, pt, 6, color, -1)
            cv2.circle(im, pt, 8, (0, 0, 0), 1)
    if boxes:
        for b in boxes:
            x1, y1, x2, y2 = (int(round(float(v) * (sx if i % 2 == 0 else sy))) for i, v in enumerate(b))
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return im
