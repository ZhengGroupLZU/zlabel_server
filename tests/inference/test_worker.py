"""ZSamWorker tests: prompt post-processing paths with fake masks."""

from __future__ import annotations

import numpy as np
import pytest

from inference.worker import ZSamWorker
from inference.ztypes import AutoMode, Point, Polygon, Rect, ReturnType, SamOnnxResult
from tests.conftest import IMG_SIZE, blob_mask


class MaskPredictor:
    """Fake model that returns one mask per result with distinct boxes."""

    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def predict(self, points=None, labels=None, bboxes=None, text=None, conf=None, iou=None, mask_input=None):
        self.calls.append((points, labels, bboxes, text))
        return [
            SamOnnxResult(mask=blob_mask(IMG_SIZE, box=b).astype(np.float32), score=0.9) for b in self.boxes
        ]


def make_worker(model, auto_mode=AutoMode.SAM, return_type=ReturnType.RECT):
    img = np.zeros((*IMG_SIZE, 3), np.uint8)
    return ZSamWorker(
        model=model,
        anno_id="test",
        img=img,
        auto_mode=auto_mode,
        return_type=return_type,
        min_contour_area_ratio=1.0e-6,
    )


class TestRunPoint:
    def test_sam_mode_merge_one(self):
        model = MaskPredictor([(400, 200, 800, 700)])
        results = make_worker(model).run_point([Point(x=500, y=400)], [1])
        # merge_one -> single bounding rect of all contours
        assert len(results) == 1
        assert isinstance(results[0], Rect)
        assert model.calls[0][0] == [(500.0, 400.0)]

    def test_cv_mode(self):
        model = MaskPredictor([])
        worker = make_worker(model, auto_mode=AutoMode.CV)
        worker.run_point([Point(x=500, y=400)], [1])
        assert model.calls == []  # CV path does not call the model

    def test_unsupported_mode(self):
        model = MaskPredictor([])
        with pytest.raises(NotImplementedError):
            make_worker(model, auto_mode=AutoMode.SAM & AutoMode.CV).run_point([Point(x=1, y=1)], [1])


class TestRunRect:
    def test_sam_mode(self):
        model = MaskPredictor([(100, 100, 500, 500), (600, 100, 900, 500)])
        results = make_worker(model).run_rect([Rect(x=100, y=100, w=400, h=400)])
        # one box prompt -> the fake returns 2 candidate masks, only the
        # highest-scoring one is post-processed -> 1 rect
        assert len(results) == 1
        assert model.calls[0][2] == [(100, 100, 500, 500)]

    def test_cv_mode_roi(self):
        model = MaskPredictor([])
        img = np.zeros((*IMG_SIZE, 3), np.uint8)
        img[150:250, 150:250] = 255  # blob inside the ROI (100..600)
        worker = ZSamWorker(
            model=model,
            anno_id="test",
            img=img,
            auto_mode=AutoMode.CV,
            return_type=ReturnType.RECT,
            min_contour_area_ratio=1.0e-6,
        )
        rect = Rect(x=100, y=100, w=500, h=500)
        results = worker.run_rect([rect])
        assert model.calls == []
        assert len(results) == 1
        assert isinstance(results[0], Rect)
        # rect is offset back into the original image coordinates
        assert results[0].x >= 100 and results[0].y >= 100


class TestRunText:
    def test_sam_mode_multi_instances(self):
        model = MaskPredictor([(100, 100, 500, 500), (600, 100, 900, 500), (100, 600, 500, 900)])
        results = make_worker(model).run_text(["person"])
        assert model.calls[0][3] == ["person"]
        assert len(results) == 3
        assert all(isinstance(r, Rect) for r in results)

    def test_requires_sam_mode(self):
        model = MaskPredictor([(100, 100, 500, 500)])
        with pytest.raises(NotImplementedError):
            make_worker(model, auto_mode=AutoMode.CV).run_text(["person"])


class TestPostprocessMask:
    def test_return_types(self):
        model = MaskPredictor([(100, 100, 500, 500)])
        worker = make_worker(model)
        mask = blob_mask(IMG_SIZE, (100, 100, 500, 500))
        rects = worker.postprocess_mask(mask, return_type=ReturnType.RECT)
        assert len(rects) == 1 and isinstance(rects[0], Rect)
        polys = worker.postprocess_mask(mask, return_type=ReturnType.POLYGON)
        assert len(polys) == 1 and isinstance(polys[0], Polygon)
        rle = worker.postprocess_mask(mask, return_type=ReturnType.RLE)
        assert len(rle) == 1 and isinstance(rle[0], str)

    def test_roi_offset(self):
        model = MaskPredictor([])
        worker = make_worker(model)
        mask = np.zeros(IMG_SIZE, np.uint8)
        mask[200:300, 200:300] = 255
        results = worker.postprocess_mask(
            mask, roi=Rect(x=100, y=100, w=500, h=500), return_type=ReturnType.RECT
        )
        assert len(results) == 1
        # rect is offset back into the original image coordinates
        assert results[0].x >= 100 and results[0].y >= 100

    def test_empty_mask(self):
        model = MaskPredictor([])
        worker = make_worker(model)
        mask = np.zeros(IMG_SIZE, np.uint8)
        assert worker.postprocess_mask(mask, merge_one=True, return_type=ReturnType.RECT) == []


class TestMaskNormalisation:
    """The SAM family emits masks of {0,1}, not {0,255} (see ``upscale_mask``)."""

    def _worker(self, mask, **kwargs):
        import numpy as np

        from inference.worker import ZSamWorker
        from inference.ztypes import AutoMode, SamOnnxResult

        class Model:
            def predict(self, points=None, labels=None, bboxes=None, text=None, mask_input=None):
                return [SamOnnxResult(mask=mask.astype(np.float32), score=0.9)]

        return ZSamWorker(
            model=Model(),
            anno_id="a",
            img=np.zeros(mask.shape + (3,), np.uint8),
            auto_mode=AutoMode.SAM,
            return_type=kwargs.pop("return_type", __import__("inference.ztypes", fromlist=["ReturnType"]).ReturnType.POLYGON),
            min_contour_area_ratio=1e-6,
            **kwargs,
        )

    def test_zero_one_mask_is_normalised_before_thresholding(self):
        import cv2
        import numpy as np

        from inference.ztypes import Point

        mask = np.zeros((120, 120), np.uint8)
        cv2.circle(mask, (60, 60), 30, 1, -1)  # {0,1} exactly like upscale_mask
        results = self._worker(mask).run_point([Point(x=60, y=60)], [1.0])
        assert results, "a {0,1} mask was thresholded away and produced no polygon"
        pts = np.array([[p.x, p.y] for p in results[0].points], np.float32)
        assert abs(cv2.contourArea(pts)) == pytest.approx(int(np.count_nonzero(mask)), rel=0.02)

    def test_cv_mode_segments_the_darker_object(self):
        import cv2
        import numpy as np

        from inference.worker import ZSamWorker
        from inference.ztypes import AutoMode, Point, ReturnType

        img = np.full((120, 120, 3), 220, np.uint8)
        cv2.rectangle(img, (40, 40), (80, 80), (10, 10, 10), -1)

        class NoModel:
            def predict(self, **_kwargs):  # pragma: no cover
                raise AssertionError("CV mode must not run the model")

        worker = ZSamWorker(
            model=NoModel(),
            anno_id="a",
            img=img,
            auto_mode=AutoMode.CV,
            threshold=100,
            return_type=ReturnType.POLYGON,
            min_contour_area_ratio=1e-6,
        )
        results = worker.run_point([Point(x=60, y=60)], [1.0])
        assert results
        pts = np.array([[p.x, p.y] for p in results[0].points], np.float32)
        x, y, w, h = cv2.boundingRect(np.round(pts).astype(np.int32))
        assert 38 <= x <= 42 and 38 <= y <= 42 and 38 <= w <= 44
