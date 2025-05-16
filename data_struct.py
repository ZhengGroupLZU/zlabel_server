from fastapi import FastAPI, status, Form, UploadFile, File, Depends, Request
from pydantic import BaseModel, ValidationError
from fastapi.exceptions import HTTPException
from fastapi.encoders import jsonable_encoder
from typing import Optional, List


class SamReturn(BaseModel):
    anno_id: str
    status: bool = False
    mode: str
    msg: str = ""
    rects: List["Rect"] | None = None


class Point(BaseModel):
    x: float
    y: float


class Rect(BaseModel):
    x: float
    y: float
    w: float
    h: float

    def to_list(self):
        return [self.x, self.y, self.w, self.h]

    def to_list_x1y1(self):
        return [self.x, self.y, self.x1, self.y1]

    @property
    def x1(self):
        return self.x + self.w

    @property
    def y1(self):
        return self.y + self.h


class Annotation(BaseModel):
    id: str
    points: List[Point] | None = None
    labels: List[float] | None = None
    rects: List[Rect] | None = None


def annotation_checker(data: str = Form(...)):
    try:
        return Annotation.model_validate_json(data)
    except ValidationError as e:
        raise HTTPException(
            detail=jsonable_encoder(e.errors()),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )
