from collections import OrderedDict
from typing import Literal

import cv2 as cv
import numpy as np
import torch
from ultralytics.engine.results import Results
from ultralytics.models.sam import Predictor, SAM2Predictor, SAM3Predictor, SAM3SemanticPredictor

from .logger import ZLogger
from .ztypes import (
    SamOnnxEncodedInput,
    SamOnnxResult,
)

MODEL_NAME_TO_CLASS = {
    "SAM": Predictor,
    "MobileSAM": Predictor,
    "SAM2": SAM2Predictor,
    "SAM3": SAM3SemanticPredictor,
}


class ZSAM:
    def __init__(
        self,
        model_path: str,
        minor_model_path: str | None = None,
        minor_model_name: Literal["SAM", "MobileSAM", "SAM2", "SAM3"] = "MobileSAM",
        cv_interpolation: int = cv.INTER_LINEAR,
        img_size: int = 1024,
        cache_size: int = 241,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        confidence: float = 0.25,
        fp16: bool = True,
    ):
        self.confidence: float = confidence
        self.img_size: int = img_size
        self.input_size = (img_size, img_size)
        self.logger = ZLogger("ZSAM")
        self.img = None
        self._cache: OrderedDict[str, SamOnnxEncodedInput] = OrderedDict()
        self.cv_interpolation = cv_interpolation
        self.cache_size: int = cache_size
        self.mean = np.array(mean or [123.675, 116.28, 103.53])
        self.std = np.array(std or [[58.395, 57.12, 57.375]])
        self.fp16: bool = fp16
        self.model_path = model_path
        self.minor_model_path = minor_model_path
        self.minor_model_class = MODEL_NAME_TO_CLASS.get(minor_model_name, Predictor)

        self.model: Predictor | None = None
        self.minor_model: Predictor | None = None

    def load_model(self, model_path: str, **kwargs) -> tuple[Predictor, Predictor | None]:
        overrides = {
            "conf": self.confidence,
            "task": "segment",
            "mode": "predict",
            "imgsz": self.img_size,
            "model": model_path,
            "half": self.fp16,  # Use FP16 for faster inference
            "save": False,  # Do not save predictions
        }
        self.model = Predictor(overrides={**overrides, **(kwargs or {})})
        if self.minor_model_path is not None:
            self.minor_model = self.minor_model_class(
                overrides={
                    **overrides,
                    **(kwargs or {}),
                    "model": self.minor_model_path,
                },
            )
        return self.model, self.minor_model

    def postprocess_results(self, infer_results: list[Results]) -> list[SamOnnxResult]:
        results = []
        for infer_result in infer_results:
            boxes = infer_result.boxes
            masks = infer_result.masks
            assert boxes is not None
            assert masks is not None
            for i in range(len(boxes)):
                bbox = boxes[i].xywh
                if isinstance(bbox, torch.Tensor):
                    bbox = bbox.cpu().numpy()
                bbox = bbox.ravel()
                assert len(bbox) == 4, f"bbox must have 4 elements, but got {len(bbox)}"

                mask = masks[i].data
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                mask = mask.squeeze().astype(np.float32) * 255
                assert len(mask.shape) == 2, f"mask must be 2D, but got {mask.shape}"

                results.append(
                    SamOnnxResult(
                        mask=mask,
                        box=(bbox[0], bbox[1], bbox[2], bbox[3]),
                        score=boxes[i].conf.item(),
                    )
                )
        return results

    def set_image(self, image: np.ndarray, minor: bool = False):
        self.img = image
        if self.model is None:
            self.load_model(self.model_path)
        if self.model is not None:
            self.model.set_image(image)
        if minor:
            if self.minor_model is not None:
                self.minor_model.set_image(image)

    def predict(
        self,
        img: np.ndarray,
        bboxes: list[tuple[float, float, float, float]] | None = None,
        points: list[tuple[float, float]] | None = None,
        labels: list[int] | None = None,
    ) -> list[SamOnnxResult]:
        self.set_image(img, minor=points is not None)

        if self.model is None:
            self.load_model(self.model_path)

        if len(bboxes or []) > 0 and self.model is not None:
            infer_results = self.model(bboxes=bboxes, labels=labels)
        elif len(points or []) > 0 and self.minor_model is not None:
            infer_results = self.minor_model(points=points, labels=labels)
        else:
            raise ValueError("At least one of bboxes or points must be provided.")

        # cv_img = infer_results[0].plot(line_width=2, font_size=16)
        # cv.imwrite("predicted.png", cv_img)

        results = self.postprocess_results(infer_results)
        return results


class ZSAM2(ZSAM):
    def load_model(self, model_path: str, **kwargs) -> tuple[Predictor, Predictor | None]:
        overrides = {
            "conf": self.confidence,
            "task": "segment",
            "mode": "predict",
            "imgsz": self.img_size,
            "model": model_path,
            "half": self.fp16,  # Use FP16 for faster inference
            "save": False,  # Do not save predictions
        }
        self.model = SAM2Predictor(overrides={**overrides, **(kwargs or {})})
        if self.minor_model_path is not None:
            self.minor_model = self.minor_model_class(
                overrides={
                    **overrides,
                    **(kwargs or {}),
                    "model": self.minor_model_path,
                },
            )
        return self.model, self.minor_model


class ZSAM3(ZSAM):
    def __init__(
        self,
        model_path: str,
        minor_model_path: str | None = None,
        minor_model_name: Literal["SAM", "MobileSAM", "SAM2", "SAM3"] = "MobileSAM",
        cv_interpolation: int = cv.INTER_LINEAR,
        img_size: int = 1024,
        cache_size: int = 241,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        confidence: float = 0.25,
        fp16: bool = True,
        semantic: bool = True,
    ):
        self.semantic = semantic

        super().__init__(
            model_path=model_path,
            minor_model_path=minor_model_path,
            minor_model_name=minor_model_name,
            cv_interpolation=cv_interpolation,
            img_size=img_size,
            cache_size=cache_size,
            mean=mean,
            std=std,
            confidence=confidence,
            fp16=fp16,
        )

    def load_model(self, model_path: str, **kwargs) -> tuple[Predictor, Predictor | None]:
        overrides = {
            "conf": self.confidence,
            "task": "segment",
            "mode": "predict",
            "imgsz": self.img_size,
            "model": model_path,
            "half": self.fp16,  # Use FP16 for faster inference
            "save": False,  # Do not save predictions
        }
        if self.semantic:
            self.model = SAM3SemanticPredictor(overrides={**overrides, **(kwargs or {})})
        else:
            self.model = SAM3Predictor(overrides={**overrides, **(kwargs or {})})

        if self.minor_model_path is not None:
            self.minor_model = self.minor_model_class(
                overrides={
                    **overrides,
                    **(kwargs or {}),
                    "model": self.minor_model_path,
                    "half": False,
                },
            )
        return self.model, self.minor_model
