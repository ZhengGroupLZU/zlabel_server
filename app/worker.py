import copy
from collections.abc import Sequence

import cv2
import cv2.typing as cv2t
import numpy as np
from numpy.typing import NDArray
from rich import print  # noqa: F401

from app.logger import ZLogger
from app.sam_onnx import SamOnnxModel
from app.ztypes import (
    AutoMode,
    Point,
    Polygon,
    Rect,
    ReturnType,
    SamOnnxPrompt,
    SamOnnxResult,
)


class ZSamWorker:
    def __init__(
        self,
        model: SamOnnxModel,
        anno_id: str,
        img: NDArray,
        auto_mode: AutoMode = AutoMode.CV,
        threshold: int = 100,
        return_type: ReturnType = ReturnType.RECT,
        min_contour_area_ratio: float = 1.0e-4,
        apply_nms: bool = True,
        iou_threshold: float = 0.5,
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
        self.shifts = [0, 0, 0, 0]
        self.logger = ZLogger("ZSamWorker")

    def run_point(
        self,
        points: list[Point],
        labels: list[float],
    ) -> Sequence[Rect | Polygon | str]:
        result_rects: list[cv2t.Rect] = []
        match self.auto_mode:
            # single point
            case AutoMode.SAM:
                # regard multiple points as single point
                prompts = [
                    SamOnnxPrompt.new(p, label) for p, label in zip(points, labels)
                ]
                r = self.run_sam(self.img, prompts)
                return self.postprocess_mask(
                    r.mask,
                    merge_one=True,
                    min_contour_area_ratio=self.min_contour_area_ratio,
                    apply_nms=self.apply_nms,
                    iou_threshold=self.iou_threshold,
                )
            # whole image by CV
            case AutoMode.CV:
                return self.postprocess_mask(
                    self.img,
                    min_contour_area_ratio=self.min_contour_area_ratio,
                    apply_nms=self.apply_nms,
                    iou_threshold=self.iou_threshold,
                )
            # whole image by SAM
            case x if x == AutoMode.SAM | AutoMode.CV:
                raise NotImplementedError
            case _:
                raise NotImplementedError
        return [Rect(x=x, y=y, w=w, h=h) for x, y, w, h in result_rects]

    def run_rect(self, rects: list[Rect]) -> Sequence[Rect | Polygon | str]:
        results: Sequence[Rect | Polygon | str] = []
        match self.auto_mode:
            case AutoMode.SAM:
                for rect in rects:
                    prompts = [SamOnnxPrompt.new(rect, 0)]
                    r = self.run_sam(self.img, prompts)
                    results.extend(
                        self.postprocess_mask(
                            r.mask,
                            min_contour_area_ratio=self.min_contour_area_ratio,
                            apply_nms=self.apply_nms,
                            iou_threshold=self.iou_threshold,
                        )
                    )
            case AutoMode.CV:
                for rect in rects:
                    r = self.postprocess_mask(
                        self.img,
                        roi=rect,
                        min_contour_area_ratio=self.min_contour_area_ratio,
                        apply_nms=self.apply_nms,
                        iou_threshold=self.iou_threshold,
                    )
                    results.extend(r)
            case x if x == AutoMode.SAM & AutoMode.CV:
                for rect in rects:
                    rects0: Sequence[Rect] = self.postprocess_mask(
                        self.img,
                        roi=rect,
                        return_type=ReturnType.RECT,
                        min_contour_area_ratio=self.min_contour_area_ratio,
                        apply_nms=self.apply_nms,
                        iou_threshold=self.iou_threshold,
                    )  # type: ignore
                    centers = [
                        Point(x=rect.x + r.x + r.w / 2, y=rect.y + r.y + r.h / 2)
                        for r in rects0
                    ]
                    tmp = [SamOnnxPrompt.new(pp, 1) for pp in centers]
                    r = self.run_sam(self.img, tmp)
                    results.extend(
                        self.postprocess_mask(
                            r.mask,
                            min_contour_area_ratio=self.min_contour_area_ratio,
                            apply_nms=self.apply_nms,
                            iou_threshold=self.iou_threshold,
                        )
                    )
            case _:
                raise NotImplementedError
        # self.plot(result_rects)
        return results

    def run_sam(
        self,
        img: NDArray,
        prompts: list[SamOnnxPrompt],
    ) -> SamOnnxResult:
        if len(prompts) == 0:
            return SamOnnxResult(np.array([[]]), 0.0)
        out = self.model.predict(img, prompts)
        if isinstance(out, list):
            return out[0]
        return out

    def postprocess_mask(
        self,
        mask: NDArray,
        merge_one: bool = False,
        roi: Rect | None = None,
        return_type: ReturnType | None = None,
        min_contour_area_ratio: float = 1.0e-6,
        apply_nms: bool = True,
        iou_threshold: float = 0.5,
    ) -> Sequence[Rect | Polygon | str]:
        return_type = return_type or self.return_type
        # return RLE-encoded mask
        if return_type == ReturnType.RLE:
            return [Polygon.rle_encode(mask)]

        # for ROI, process ROI
        _mask = mask.copy()
        offset_x, offset_y = 0, 0
        if isinstance(roi, Rect):
            x, y, w, h = int(roi.x), int(roi.y), int(roi.w), int(roi.h)
            _mask = mask[y : y + h, x : x + w]
            offset_x, offset_y = x, y
        # cv2.imwrite("mask.png", _mask)
        _mask = cv2.dilate(_mask, np.ones((3, 3), np.uint8), iterations=1)
        _mask = cv2.blur(_mask, (2, 2))
        # cv2.imwrite("mask_blur.png", _mask)
        _mask = cv2.erode(_mask, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(
            _mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        # Filter out small contours and apply NMS
        contours = self.contour_filter(
            contours,
            min_area_ratio=min_contour_area_ratio,
            apply_nms=apply_nms,
            iou_threshold=iou_threshold,
        )
        # contours = [cv2.approxPolyDP(c, 3, True) for c in contours]

        # return rectangles
        if return_type == ReturnType.RECT:
            if merge_one:
                new_contours = []
                for c in contours:
                    new_contours.extend(list(c))
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
                polygons.append(
                    Polygon(points=[Point(x=i[0], y=i[1]) for i in _contour])
                )
            return polygons
        else:
            raise NotImplementedError

    def nms_contours(
        self,
        contours: list[cv2t.MatLike],
        iou_threshold: float = 0.5,
    ) -> list[cv2t.MatLike]:
        """
        Apply Non-Maximum Suppression (NMS) to contours to remove highly overlapping contours.

        Args:
            contours: List of contours to apply NMS to
            iou_threshold: IoU threshold for suppression (default: 0.5)
                Contours with IoU > threshold will be suppressed, keeping only the largest one

        Returns:
            List of contours after NMS
        """
        if not contours:
            return []

        # self.logger.debug(
        #     f"**nms_contours**, before, num_contours={len(contours)}, iou_threshold={iou_threshold}"
        # )
        # Calculate bounding boxes and areas for all contours
        boxes = []
        areas = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append([x, y, x + w, y + h])  # Convert to [x1, y1, x2, y2] format
            areas.append(w * h)

        # Sort contours by area in descending order
        indices = np.argsort(areas)[::-1]

        # Initialize list of kept indices
        keep_indices = []

        while len(indices) > 0:
            # Get the index of the largest remaining contour
            current = indices[0]
            keep_indices.append(current)

            if len(indices) == 1:
                break

            # Get the rest of the indices
            rest_indices = indices[1:]

            # Calculate IoU between the current contour and all remaining contours
            current_box = boxes[current]
            ious = []

            for idx in rest_indices:
                box = boxes[idx]

                # Calculate intersection
                x1 = max(current_box[0], box[0])
                y1 = max(current_box[1], box[1])
                x2 = min(current_box[2], box[2])
                y2 = min(current_box[3], box[3])

                # Calculate intersection area
                if x2 <= x1 or y2 <= y1:
                    intersection = 0
                else:
                    intersection = (x2 - x1) * (y2 - y1)

                # Calculate union area
                union = areas[current] + areas[idx] - intersection

                # Calculate IoU
                iou = intersection / union if union > 0 else 0
                ious.append(iou)

            # Keep only contours with IoU <= threshold
            indices = [
                rest_indices[i] for i, iou in enumerate(ious) if iou <= iou_threshold
            ]

        # Return the kept contours
        # self.logger.debug(
        #     f"**nms_contours**, after, num_contours={len(keep_indices)}, iou_threshold={iou_threshold}"
        # )
        return [contours[i] for i in keep_indices]

    def contour_filter(
        self,
        contours: Sequence[cv2t.MatLike],
        min_area_ratio: float = 1.0e-6,
        apply_nms: bool = True,
        iou_threshold: float = 0.5,
    ) -> list[cv2t.MatLike]:
        """
        Filter contours by area to remove small contours.
        Optionally apply Non-Maximum Suppression (NMS) to remove overlapping contours.

        Args:
            contours: List of contours to filter
            min_area_ratio: Minimum area ratio relative to image area (default: 0.001 = 0.1%)
            apply_nms: Whether to apply NMS to remove overlapping contours (default: True)
            iou_threshold: IoU threshold for NMS (default: 0.5)

        Returns:
            Filtered list of contours
        """
        if not contours:
            return []

        # Calculate image area
        img_area = self.img.shape[0] * self.img.shape[1]
        min_area = img_area * min_area_ratio

        # Filter contours by area
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Skip if area is too small
            if area < min_area:
                self.logger.debug(
                    f"[filtered] contour area {area}, min_area={min_area}"
                )
                continue

            filtered_contours.append(contour)

        # Apply NMS if requested
        if apply_nms and filtered_contours:
            filtered_contours = self.nms_contours(filtered_contours, iou_threshold)

        return filtered_contours

    def rect_filter(self, rects: list[cv2t.Rect]) -> list[Rect]:
        """
        Filter rects by area.
        """
        areas = np.asarray([w * h for _, _, w, h in rects], dtype=np.float32)
        counts, bins = np.histogram(areas, bins="auto")
        area_most = bins[np.argmax(counts) + 1]
        # self.logger.debug(f"{areas=}, {area_most=}")
        idxs = np.where((areas > area_most * 0.3) & (areas < area_most * 8))[0]
        rects1 = [rects[i] for i in idxs]
        return [Rect(x=x, y=y, w=w, h=h) for x, y, w, h in rects1]

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
    # img = cv2.imread("5704.png")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    # model = SamOnnxModel(
    #     "assets/sam_vit_b_encoder_quantized.onnx",
    #     "assets/sam_vit_b_decoder_quantized.onnx",
    # )
    # model.encode(np.asarray(img, dtype=np.uint8))
    # points = [
    #     Point(x=150, y=85),
    # ]
    # labels = [1]
    # worker = ZSamWorker(
    #     model,
    #     "TEST_RESULT_ID",
    #     img,
    #     AutoMode.SAM,
    #     threshold=100,
    #     return_type=ReturnType.POLYGON,
    # )
    # results = worker.run_point(points, [1])
    # if results and isinstance(results[0], str):
    #     mask = Polygon.rle_decode(results[0], shape=img.shape[:2])
    #     mask[mask==1] = 255
    #     cv2.imwrite("mask.png", mask)
    # elif results and isinstance(results[0], Rect):
    #     x, y, w, h = int(results[0].x), int(results[0].y), int(results[0].w), int(results[0].h)
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
    #     cv2.imwrite("mask.png", img)
    # elif results and isinstance(results[0], Polygon):
    #     points = [np.array([[int(p.x), int(p.y)] for p in r.points]) for r in results]
    #     cv2.drawContours(img, points, -1, (255, 0, 0), cv2.FILLED)
    #     cv2.imwrite("mask.png", img)
    # else:
    #     raise NotImplementedError
