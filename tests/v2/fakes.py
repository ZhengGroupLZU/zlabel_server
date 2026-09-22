"""Test doubles.

``LocalBackendHarness`` seeds the real local storage tree (the backend under test)
in the shape of the old OpenList fake, so most seeding call sites stayed unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class LocalBackendHarness:
    """Seeds the **real** local storage root and registers test accounts.

    Kept in the shape of the old OpenList fake (``add_file``/``add_dir``/``files``/
    ``users``) so the suite drives the backend under test instead of a simulation:
    every write lands in ``settings.storage_root``.
    """

    VIRTUAL_ROOT = "/zlabel_server/projects"  # the layout the tests spell their paths in

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.files: dict[str, bytes] = _FileMap(self)
        self.dirs: set[str] = {self.VIRTUAL_ROOT}
        self.users: dict[str, str] = {}  # name -> password, created on first login

    def disk_path(self, path: str) -> Path:
        """Map a test-facing (OpenList style, absolute) path onto the storage root."""
        clean = str(path).replace("\\", "/")
        if clean.startswith(self.VIRTUAL_ROOT):
            clean = clean[len(self.VIRTUAL_ROOT) :]
        return self.root / clean.lstrip("/")

    def add_file(self, path: str, data: bytes = b"data") -> str:
        target = self.disk_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self._remember_parents(path)
        return path

    def add_dir(self, path: str) -> str:
        self.disk_path(path).mkdir(parents=True, exist_ok=True)
        self.dirs.add(path.rstrip("/"))
        self._remember_parents(path)
        return path

    def images(self) -> list[str]:
        return sorted(self.files)

    def _remember_parents(self, path: str) -> None:
        parts = str(path).strip("/").split("/")
        for i in range(1, len(parts)):
            self.dirs.add("/" + "/".join(parts[:i]))


class _FileMap:
    """A live view of the harness tree.

    Keys are the virtual (OpenList style, absolute) paths the tests spell; values are
    read from disk, so the assertions keep working on paths the *server* produced
    (which are relative to the storage root).
    """

    def __init__(self, harness: LocalBackendHarness) -> None:
        self._harness = harness

    def _path(self, key: str) -> Path:
        return self._harness.disk_path(key)

    def __getitem__(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise KeyError(key)
        return path.read_bytes()

    def __setitem__(self, key: str, value: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)

    def __delitem__(self, key: str) -> None:
        path = self._path(key)
        if not path.is_file():
            raise KeyError(key)
        path.unlink()

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._path(key).is_file()

    def keys(self) -> list[str]:
        root = self._harness.root
        return sorted(
            f"{self._harness.VIRTUAL_ROOT}/{p.relative_to(root).as_posix()}"
            for p in root.rglob("*")
            if p.is_file()
        )

    def values(self) -> list[bytes]:
        return [self[k] for k in self.keys()]

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            return {k: self[k] for k in self.keys()} == other
        return NotImplemented

    def __repr__(self) -> str:
        return f"_FileMap({self.keys()!r})"

class FakeInference:
    """Stand-in for the inference worker (records jobs, returns a SamReturn)."""

    def __init__(self) -> None:
        self.jobs: list[dict] = []
        self.fail_with: Exception | None = None

    def health(self) -> dict:
        return {"status": "ok", "detail": {"model": "FakeSAM"}}

    def infer(self, job: dict) -> dict:
        if self.fail_with is not None:
            raise self.fail_with
        self.jobs.append(job)
        return {
            "anno_id": job.get("anno_id", ""),
            "status": True,
            "mode": "SAM",
            "msg": "success",
            "data": [{"x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0}],
        }


class FakeEnginePredictor:
    """Predictor stand-in for the worker engine: counts encodes/restores."""

    loaded = True

    def __init__(self, shape: tuple[int, int] = (720, 1280)) -> None:
        self.shape = shape
        self.encoded: list[tuple[int, int]] = []
        self.restored: list[dict] = []
        self.image = None

    def set_image(self, image):
        self.image = image
        self.shape = image.shape[:2]
        self.encoded.append(self.shape)
        return self

    def export_image_state(self) -> dict:
        return {"shape": self.shape, "images": len(self.encoded)}

    def import_image_state(self, state: dict) -> FakeEnginePredictor:
        self.restored.append(state)
        self.shape = state["shape"]
        return self

    def predict(self, points=None, labels=None, bboxes=None, text=None, conf=None, iou=None):
        from inference.ztypes import SamOnnxResult

        height, width = self.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        # a centred blob that always fits the image the encoder saw
        y0, y1 = height // 4, max(height // 4 + 1, height * 3 // 4)
        x0, x1 = width // 4, max(width // 4 + 1, width * 3 // 4)
        mask[y0:y1, x0:x1] = 255
        return [SamOnnxResult(mask=mask.astype(np.float32), score=0.9)]
