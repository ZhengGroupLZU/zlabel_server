from enum import Enum
from typing import List, Tuple
from pydantic import BaseModel
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from data_struct import Point, Rect


@dataclass
class SamOnnxResult(object):
    mask: NDArray[np.float32]
    score: float


class PromptType(Enum):
    POINT = "point"
    RECTANGLE = "rectangle"


@dataclass
class SamOnnxPrompt(object):
    type_: PromptType
    # (x, y) for point, (x0, y0, x1, y1) for rectangle
    point: Tuple[float, float] | Tuple[float, float, float, float]
    label: float

    @staticmethod
    def new(p: Point | Rect, label):
        match p:
            case Point():
                prompt = SamOnnxPrompt(PromptType.POINT, (p.x, p.y), label)
            case Rect():
                prompt = SamOnnxPrompt(
                    PromptType.RECTANGLE,
                    (p.x, p.y, p.x1, p.y1),
                    label,
                )
            case _:
                raise ValueError
        return prompt


@dataclass
class SamOnnxEncodedInput(object):
    image_embedding: NDArray[np.float32]
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
