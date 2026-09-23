"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
MODEL_DIR = ASSETS_DIR / "onnx"
TEST_IMAGE = Path(__file__).resolve().parent / "data" / "zidane.jpg"
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
        from inference.ztypes import SamOnnxResult

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


# --- hermetic configuration ------------------------------------------------- #
# A developer's real ``.env`` holds live hosts and credentials. Capture the
# shipped defaults first, then make the whole suite ignore the file: env *vars*
# still work (tests set them explicitly), the file never leaks in.
from app.core.config import Settings as _ApiSettings  # noqa: E402
from inference.config import InferenceSettings as _WorkerSettings  # noqa: E402

ENV_FILE_DEFAULTS = {
    "api": _ApiSettings.model_config.get("env_file"),
    "worker": _WorkerSettings.model_config.get("env_file"),
}
ENV_PREFIX_DEFAULTS = {
    "api": _ApiSettings.model_config.get("env_prefix"),
    "worker": _WorkerSettings.model_config.get("env_prefix"),
}


@pytest.fixture
def shipped_env_file() -> dict[str, str | None]:
    """The env-file names the app ships with (captured before the hermetic patch).

    Defined as a fixture on purpose: importing ``tests.conftest`` from a test would
    execute the module a second time, when the patch below is already active.
    """
    return dict(ENV_FILE_DEFAULTS)


@pytest.fixture(autouse=True)
def _hermetic_env_file(monkeypatch):
    """Do not read ``.env`` (nor any other env file) during tests."""
    for cls in (_ApiSettings, _WorkerSettings):
        monkeypatch.setattr(cls, "model_config", {**cls.model_config, "env_file": None})
    yield
