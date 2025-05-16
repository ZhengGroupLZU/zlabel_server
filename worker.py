import copy
from enum import Flag
from typing import List, Optional, Tuple

import cv2
import cv2.typing as cv2t
import numpy as np
from numpy.typing import NDArray
from rich import print

from data_struct import Point, Rect
from ztypes import SamOnnxPrompt, SamOnnxResult

from sam_onnx import SamOnnxModel


class AutoMode(Flag):
    SAM = 1
    CV = 2
    SAM_AND_CV = 1 & 2
    MANUAL = 3


class ZSamWorker(object):
    def __init__(
        self,
        model: SamOnnxModel,
        anno_id: str,
        img: NDArray,
        auto_mode: AutoMode = AutoMode.CV,
        threshold: int = 100,
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
        self.shifts = [0, 0, 0, 0]

    def run_point(self, points: List[Point], labels: List[float]) -> List[Rect]:
        result_rects: List[cv2t.Rect] = []
        match self.auto_mode:
            # single point
            case AutoMode.SAM:
                # regard multiple points as single point
                prompts = [
                    SamOnnxPrompt.new(p, label) for p, label in zip(points, labels)
                ]
                r = self.run_sam(self.img, prompts)
                result_rects = self.rects_cv(r.mask, merge_one=True)
            # whole image by CV
            case AutoMode.CV:
                result_rects = self.rects_cv(self.img)
            # whole image by SAM
            case x if x == AutoMode.SAM | AutoMode.CV:
                raise NotImplementedError
            case _:
                raise NotImplementedError
        return [Rect(x=x, y=y, w=w, h=h) for x, y, w, h in result_rects]

    def run_rect(self, rects: List[Rect]) -> List[Rect]:
        result_rects: List[cv2t.Rect] = []
        match self.auto_mode:
            case AutoMode.SAM:
                for rect in rects:
                    prompts = [SamOnnxPrompt.new(rect, 0)]
                    r = self.run_sam(self.img, prompts)
                    result_rects.extend(self.rects_cv(r.mask))
            case AutoMode.CV:
                for rect in rects:
                    x, y, x1, y1 = [int(i) for i in rect.to_list_x1y1()]
                    rects1 = np.array(
                        [
                            [r[0] + x, r[1] + y, r[2], r[3]]
                            for r in self.rects_cv(self.img[y:y1, x:x1])
                        ],
                        dtype=int,
                    )
                    result_rects.extend(rects1)
            case x if x == AutoMode.SAM & AutoMode.CV:
                for rect in rects:
                    x, y, x1, y1 = [int(i) for i in rect.to_list_x1y1()]
                    rects0 = self.rects_cv(self.img[y:y1, x:x1])
                    centers = [
                        Point(x=x + xx + ww / 2, y=y + yy + hh / 2)
                        for xx, yy, ww, hh in rects0
                    ]
                    tmp = [SamOnnxPrompt.new(pp, 1) for pp in centers]
                    r = self.run_sam(self.img, tmp)
                    result_rects.extend(self.rects_cv(r.mask))
            case _:
                raise NotImplementedError
        # self.plot(result_rects)
        return [Rect(x=x, y=y, w=w, h=h) for x, y, w, h in result_rects]

    def run_sam(
        self,
        img: NDArray,
        prompts: List[SamOnnxPrompt],
    ) -> SamOnnxResult:
        if len(prompts) == 0:
            return SamOnnxResult(np.array([[]]), 0.0)
        out = self.model.predict(img, prompts)
        return out

    def rects_cv(self, img: NDArray, merge_one: bool = False) -> List[cv2t.Rect]:
        # img = cv2.blur(img, (2, 2))
        canny_out = cv2.Canny(img, self.threshold, self.threshold * 2)
        contours, _ = cv2.findContours(
            canny_out,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        # contours = [cv2.approxPolyDP(c, 3, True) for c in contours]
        if merge_one:
            new_contours = []
            for c in contours:
                new_contours.extend(list(c))
            rects = [cv2.boundingRect(np.array(new_contours))]
        else:
            rects = [cv2.boundingRect(m) for m in contours]
        return self.rect_filter(rects)  # type: ignore

    def rect_filter(self, rects: List[cv2t.Rect]) -> List[cv2t.Rect]:
        areas = np.asarray([w * h for _, _, w, h in rects], dtype=np.float32)
        counts, bins = np.histogram(areas, bins="auto")
        area_most = bins[np.argmax(counts) + 1]
        # print(f"{areas=}, {area_most=}")
        idxs = np.where((areas > area_most * 0.3) & (areas < area_most * 8))[0]
        return [rects[i] for i in idxs]

    def plot(self, rects: List[cv2t.Rect], points: List[Point] | None = None):
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
    # img = cv2.imread("401.png")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # model = SamOnnxModel(
    #     "data/sam_vit_l_encoder_quantized.onnx",
    #     "data/sam_vit_l_decoder_quantized.onnx",
    # )
    # points = [
    #     (150, 85),
    # ]
    # labels = [1]
    # worker = ZSamWorker(
    #     model,
    #     "TEST_RESULT_ID",
    #     [Label.default()],
    #     img,
    #     points,
    #     ResultType.POINT,
    #     AutoMode.SAM,
    #     threshold=100,
    #     parent=None,
    # )
    # r = worker.run()
    # print(r)
