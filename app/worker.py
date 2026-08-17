import copy
from collections.abc import Sequence

import cv2
import cv2.typing as cv2t
import numpy as np
from numpy.typing import NDArray
from rich import print  # noqa: F401

from app.logger import ZLogger
from app.sam_ort import Predictor
from app.sam_ort.postprocess import (
    contour_filter,
    nms_filter,
    reduce_contour_points,
    smooth_contour,
)
from app.ztypes import (
    AutoMode,
    Point,
    Polygon,
    Rect,
    ReturnType,
)


class ZSamWorker:
    def __init__(
        self,
        model: Predictor,
        anno_id: str,
        img: NDArray,
        auto_mode: AutoMode = AutoMode.CV,
        threshold: int = 100,
        return_type: ReturnType = ReturnType.RECT,
        min_contour_area_ratio: float = 3.0e-5,
        apply_nms: bool = True,
        iou_threshold: float = 0.5,
        contour_min_points: int = 5,
        contour_max_points: int = 100,
        contour_max_iterations: int = 10,
    ) -> None:
        """
        For point:
            auto_mode=AutoMode.SAM: use SAM to predict single point mask
            auto_mode=AutoMode.CV: use opencv to segment the whole image and return the mask
            auto_mode=AutoMode.SAM|AutoMode.CV: use SAM to predict the whole image
        For rectangle:
            auto_mode=AutoMode.SAM&AutoMode.CV: segment using opencv first, get rectangles' center point and use SAM to predict
            auto_mode=AutoMode.CV: use opencv to segment selected rectangle masks
            auto_mode=AutoMode.SAM: use SAM to predict the rectangle mask
        """
        super().__init__()
        self.auto_mode = auto_mode
        self.model = model
        self.anno_id = anno_id
        self.img = img
        self.threshold = threshold
        self.return_type = return_type
        self.min_contour_area_ratio = min_contour_area_ratio
        self.apply_nms = apply_nms
        self.iou_threshold = iou_threshold
        self.contour_min_points = contour_min_points
        self.contour_max_points = contour_max_points
        self.contour_max_iterations = contour_max_iterations
        self.shifts = [0, 0, 0, 0]
        self.logger = ZLogger("ZSamWorker")

    def run_sam(
        self,
        points: list[tuple[float, float]] | None = None,
        labels: list[float] | None = None,
        bboxes: list[tuple[float, float, float, float]] | None = None,
        texts: list[str] | None = None,
    ):
        return self.model.predict(points=points, labels=labels, bboxes=bboxes, text=texts)

    def run_text(self, texts: list[str]) -> Sequence[Rect | Polygon | str]:
        """SAM3 text prompt: detect+segment all instances of each text."""
        if self.auto_mode != AutoMode.SAM:
            raise NotImplementedError
        results: list[Rect | Polygon | str] = []
        for r in self.run_sam(texts=texts):
            results.extend(self.postprocess_mask(r.mask.astype(np.uint8)))
        return self.results_filter(results)

    def run_point(
        self,
        points: list[Point],
        labels: list[float],
    ) -> Sequence[Rect | Polygon | str]:
        final_results: list[Rect | Polygon | str] = []
        match self.auto_mode:
            # single point
            case AutoMode.SAM:
                # regard multiple points as single point
                pts = [(p.x, p.y) for p in points]
                results = self.run_sam(points=pts, labels=labels)
                # SAM returns several candidate masks per point; keep only the
                # highest-scoring one to avoid near-identical overlays.
                if results:
                    final_results.extend(
                        self.postprocess_mask(
                            results[0].mask.astype(np.uint8),
                            merge_one=True,
                            min_contour_area_ratio=self.min_contour_area_ratio,
                        )
                    )

                return final_results
            # whole image by CV
            case AutoMode.CV:
                return self.postprocess_mask(
                    self.img,
                    min_contour_area_ratio=self.min_contour_area_ratio,
                )
            # whole image by SAM
            case x if x == AutoMode.SAM | AutoMode.CV:
                raise NotImplementedError
            case _:
                raise NotImplementedError
        return final_results

    def run_rect(self, rects: list[Rect]) -> Sequence[Rect | Polygon | str]:
        results: Sequence[Rect | Polygon | str] = []
        match self.auto_mode:
            case AutoMode.SAM:
                for rect in rects:
                    box = (rect.x, rect.y, rect.x + rect.w, rect.y + rect.h)
                    # SAM returns several candidate masks per box; keep only the
                    # highest-scoring one to avoid drawing near-identical overlays.
                    candidates = self.run_sam(bboxes=[box])
                    if candidates:
                        results.extend(
                            self.postprocess_mask(
                                candidates[0].mask.astype(np.uint8),
                                min_contour_area_ratio=self.min_contour_area_ratio,
                            )
                        )
            case AutoMode.CV:
                for rect in rects:
                    r = self.postprocess_mask(
                        self.img,
                        roi=rect,
                        min_contour_area_ratio=self.min_contour_area_ratio,
                    )
                    results.extend(r)
            case x if x == AutoMode.SAM & AutoMode.CV:
                for rect in rects:
                    rects0: Sequence[Rect] = self.postprocess_mask(
                        self.img,
                        roi=rect,
                        return_type=ReturnType.RECT,
                        min_contour_area_ratio=self.min_contour_area_ratio,
                    )  # type: ignore
                    centers = [Point(x=rect.x + r.x + r.w / 2, y=rect.y + r.y + r.h / 2) for r in rects0]
                    for r in self.run_sam(points=[(p.x, p.y) for p in centers], labels=[1] * len(centers)):
                        results.extend(
                            self.postprocess_mask(
                                r.mask.astype(np.uint8),
                                min_contour_area_ratio=self.min_contour_area_ratio,
                            )
                        )
            case _:
                raise NotImplementedError
        # self.plot(result_rects)
        return self.results_filter(results)

    def postprocess_mask(
        self,
        mask: NDArray,
        merge_one: bool = False,
        roi: Rect | None = None,
        return_type: ReturnType | None = None,
        min_contour_area_ratio: float = 1.0e-6,
    ) -> Sequence[Rect | Polygon | str]:
        return_type = return_type or self.return_type
        # return RLE-encoded mask
        if return_type == ReturnType.RLE:
            return [Polygon.rle_encode(mask)]

        # for ROI, process ROI
        _mask = mask.copy()
        if _mask.ndim == 3:
            _mask = cv2.cvtColor(_mask, cv2.COLOR_BGR2GRAY)
        offset_x, offset_y = 0, 0
        if isinstance(roi, Rect):
            x, y, w, h = int(roi.x), int(roi.y), int(roi.w), int(roi.h)
            _mask = _mask[y : y + h, x : x + w]
            offset_x, offset_y = x, y
        # cv2.imwrite("mask.png", _mask)
        _mask = cv2.dilate(_mask, np.ones((3, 3), np.uint8), iterations=1)
        # Gaussian blur + re-threshold rounds the binary edge (vs the tiny 2x2 box
        # blur), which removes most of the mask stair-stepping.
        _mask = cv2.GaussianBlur(_mask, (3, 3), 0)
        # cv2.imwrite("mask_blur.png", _mask)
        _mask = cv2.erode(_mask, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(
            _mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        # Filter out small contours and apply NMS
        contours = contour_filter(
            contours,
            min_contour_area_ratio,
            self.img.shape[:2],
        )

        # contours = [cv2.approxPolyDP(c, 3, True) for c in contours]
        # smooth the contour points, then drop redundant points for a cleaner shape
        contours = [smooth_contour(c) for c in contours]
        contours = [
            reduce_contour_points(
                c,
                min_points=self.contour_min_points,
                max_points=self.contour_max_points,
                max_iterations=self.contour_max_iterations,
            )
            for c in contours
        ]

        # return rectangles
        if return_type == ReturnType.RECT:
            if merge_one:
                new_contours = []
                for c in contours:
                    new_contours.extend(list(c))
                if len(new_contours) == 0:
                    return []
                rects = [cv2.boundingRect(np.array(new_contours))]
            else:
                rects = [cv2.boundingRect(m) for m in contours]
                rects = [(x + offset_x, y + offset_y, w, h) for x, y, w, h in rects]
            return self.rect_filter(rects)  # type: ignore
        # return polygons
        elif return_type == ReturnType.POLYGON:
            polygons = []
            for contour in contours:
                _contour = contour.copy().reshape(-1, 2).astype(np.float32)
                _contour[:, 0] += offset_x
                _contour[:, 1] += offset_y
                # _contour[:, 0] = _contour[:, 0] / mask.width
                # _contour[:, 1] = _contour[:, 1] / mask.height
                polygons.append(Polygon(points=[Point(x=i[0], y=i[1]) for i in _contour]))
            return polygons
        else:
            raise NotImplementedError

    def rect_filter(self, rects: list[cv2t.Rect]) -> list[Rect]:
        """
        Filter rects by area.
        """
        areas = np.asarray([w * h for _, _, w, h in rects], dtype=np.float32)
        if len(areas) == 0:
            return []
        counts, bins = np.histogram(areas, bins="auto")
        area_most = bins[np.argmax(counts) + 1]
        # self.logger.debug(f"{areas=}, {area_most=}")
        idxs = np.where((areas > area_most * 0.3) & (areas < area_most * 8))[0]
        rects1 = [rects[i] for i in idxs]
        return [Rect(x=x, y=y, w=w, h=h) for x, y, w, h in rects1]

    def results_filter(self, results: list[Rect | Polygon | str]) -> list[Rect | Polygon | str]:
        bounding_rects = [(r.x, r.y, r.w, r.h) for r in results if isinstance(r, Rect)]
        for p in results:
            if not isinstance(p, Polygon):
                continue
            contour = np.array([[p.x, p.y] for p in p.points], dtype=np.int32).reshape(-1, 1, 2)
            x, y, w, h = cv2.boundingRect(contour)
            bounding_rects.append((x, y, w, h))
        keep_indices = nms_filter(bounding_rects, self.iou_threshold)
        new_results = [results[i] for i in keep_indices]
        new_results += [r for r in results if isinstance(r, str)]
        self.logger.debug(f"Filtering results, num results={len(results)} -> {len(new_results)}")
        return new_results

    def plot(self, rects: list[cv2t.Rect], points: list[Point] | None = None):
        im = copy.deepcopy(self.img)
        if points:
            cv2.circle(
                im,
                (int(points[0].x), int(points[0].y)),
                2,
                (0, 255, 255),
                -1,
            )
        for x, y, w, h in rects:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.imwrite("self.img.png", im)


if __name__ == "__main__":
    ...
