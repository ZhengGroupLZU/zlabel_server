"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
MODEL_DIR = ASSETS_DIR / "onnx"
TEST_IMAGE = Path(__file__).resolve().parent.parent / "notebooks" / "assets" / "zidane.jpg"
IMG_SIZE = (720, 1280)  # zidane.jpg


@pytest.fixture(scope="session")
def img_path() -> Path:
    assert TEST_IMAGE.exists(), f"test image missing: {TEST_IMAGE}"
    return TEST_IMAGE


@pytest.fixture(scope="session")
def img_bgr(img_path: Path) -> np.ndarray:
    return cv2.imread(str(img_path))


def blob_mask(shape: tuple[int, int], box=(400, 200, 800, 700)) -> np.ndarray:
    """A solid rectangular mask (255 inside the box) used as fake model output."""
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = 255
    return mask


class FakePredictor:
    """Deterministic Predictor stand-in returning blob masks."""

    def __init__(self, img_size: tuple[int, int] = IMG_SIZE, box=(400, 200, 800, 700)):
        self.img_size = img_size
        self.box = box
        self.calls: list[dict] = []
        self.set_image_calls: list[np.ndarray] = []

    def set_image(self, image: np.ndarray):
        self.set_image_calls.append(image)
        self.img_size = image.shape[:2]

    def reset_image(self):
        self.set_image_calls.clear()

    def predict(self, points=None, labels=None, bboxes=None, text=None, conf=None, iou=None):
        from app.ztypes import SamOnnxResult

        self.calls.append({"points": points, "labels": labels, "bboxes": bboxes, "text": text})
        n = 1
        if bboxes:
            n = 2
        if text:
            n = 3
        mask = blob_mask(self.img_size, box=self.box).astype(np.float32)
        return [SamOnnxResult(mask=mask.copy(), score=0.9) for _ in range(n)]


@pytest.fixture
def fake_predictor() -> FakePredictor:
    return FakePredictor()
