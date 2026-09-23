"""The encoded-image snapshot used by the serving cache must be complete.

Restoring a snapshot has to reproduce the masks exactly: the worker relies on it
instead of re-running the encoder for every prompt on the same task.
"""

from __future__ import annotations

import numpy as np
import pytest

from inference.sam_ort import Predictor
from tests.conftest import MODEL_DIR


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def test_loaded_flag_tracks_the_sessions(img_bgr):
    predictor = Predictor(MODEL_DIR, "EdgeSAM", backend="CPU")
    assert predictor.loaded is False
    predictor.set_image(img_bgr)
    assert predictor.loaded is True
    predictor.reset_image()
    assert predictor.loaded is False


@pytest.mark.parametrize("name", ["EdgeSAM", "SlimSAM"])
def test_snapshot_roundtrip_reproduces_the_masks(name, img_bgr):
    predictor = Predictor(MODEL_DIR, name, backend="CPU")
    predictor.set_image(img_bgr)
    baseline = predictor.predict(points=[(640, 360)], labels=[1])
    snapshot = predictor.export_image_state()

    flipped = img_bgr[:, ::-1].copy()
    predictor.set_image(flipped)
    other = predictor.predict(points=[(640, 360)], labels=[1])

    predictor.import_image_state(snapshot)
    restored = predictor.predict(points=[(640, 360)], labels=[1])

    assert mask_iou(baseline[0].mask, restored[0].mask) > 0.999
    assert mask_iou(baseline[0].mask, other[0].mask) < 0.99  # the images really differ


def test_snapshot_survives_a_box_prompt(img_bgr):
    predictor = Predictor(MODEL_DIR, "EdgeSAM", backend="CPU")
    predictor.set_image(img_bgr)
    snapshot = predictor.export_image_state()
    baseline = predictor.predict(bboxes=[[400, 200, 800, 700]])

    predictor.set_image(img_bgr[:100, :100].copy())  # clobber the state
    predictor.import_image_state(snapshot)
    restored = predictor.predict(bboxes=[[400, 200, 800, 700]])

    assert mask_iou(baseline[0].mask, restored[0].mask) > 0.999
