from io import BytesIO
from typing import List
import subprocess

import numpy as np
from fastapi import Depends, FastAPI, File, Form, UploadFile
from PIL import Image
from rich import print

from data_struct import Annotation, Point, Rect, SamReturn, annotation_checker
from sam_onnx import SamOnnxModel, EdgeSam
from worker import AutoMode, ZSamWorker

# ENCODER_PATH = "assets/sam_vit_h_encoder_quantized.onnx"
# DECODER_PATH = "assets/sam_vit_h_decoder_quantized.onnx"
ENCODER_PATH = "assets/edge_sam_3x_encoder.onnx"
DECODER_PATH = "assets/edge_sam_3x_decoder.onnx"

app = FastAPI()
# SAM_MODEL = SamOnnxModel(ENCODER_PATH, DECODER_PATH)
SAM_MODEL = EdgeSam(ENCODER_PATH, DECODER_PATH)


@app.get("/")
async def root():
    return {"message": "Welcome!"}


@app.get("/api/v1/how-many-finished")
async def how_many_finished():
    dir0 = "/home/rainy/dev/datasets/labelspace/seeds_data"
    cmds = [
        ["find", f"{dir0}/exported_pngs", "-name", "*.png"],
        ["find", f"{dir0}/exported_pngs_label", "-name", "*.zlabel"],
        ["find", f"{dir0}/exported_pngs_label", "-name", "*.zlabel", "-size", "+1k"],
    ]
    wc = ["wc", "-l"]
    nums = []
    for cmd in cmds:
        find = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        n = subprocess.check_output(wc, stdin=find.stdout)
        nums.append(n.decode("utf-8").replace("\n", ""))
    return {"all": nums[0], "labeled": nums[1], "verified": nums[2]}


@app.post("/api/v1/setimage")
async def set_image(image: UploadFile = File(...)):
    content = await image.read()
    img = Image.open(BytesIO(content))
    SAM_MODEL.encode(np.asarray(img, dtype=np.uint8))
    return {"status": True}


@app.post("/api/v1/predict")
async def predict(
    image: UploadFile = File(...),
    anno: Annotation = Depends(annotation_checker),
    threshold: int = Form(100),
    mode: int = Form(1),
):
    """
    mode: 1 -> SAM, 2 -> CV, 1&2 -> SAM_AND_CV
    """
    # print(anno, image.file)
    content = await image.read()
    img = Image.open(BytesIO(content))
    auto_mode = AutoMode(mode)
    return _predict(
        anno.id,
        img,
        anno.points,
        anno.labels,
        anno.rects,
        threshold,
        auto_mode,
    )


@app.post("/api/v2/predict")
async def predict_v2(
    anno: Annotation = Depends(annotation_checker),
    threshold: int = Form(100),
    mode: int = Form(1),
    image_name: str = Form(...),
):
    ...


def _predict(
    anno_id: str,
    img: Image.Image,
    points: List[Point] | None,
    labels: List[float] | None,
    rects: List[Rect] | None,
    threshold: int,
    auto_mode: AutoMode,
):
    status = False
    msg = ""
    worker_result = None
    match (points, labels, rects):
        case (p, l, r) if p is not None and l is not None and len(p) == len(l):
            worker = ZSamWorker(
                model=SAM_MODEL,
                anno_id=anno_id,
                img=np.asarray(img, dtype=np.uint8),
                auto_mode=auto_mode,
                threshold=threshold,
            )
            worker_result = worker.run_point(p, l)
            status = True
            msg = "success"
        case (p, l, r) if r is not None:
            worker = ZSamWorker(
                model=SAM_MODEL,
                anno_id=anno_id,
                img=np.asarray(img, dtype=np.uint8),
                auto_mode=auto_mode,
                threshold=threshold,
            )
            worker_result = worker.run_rect(r)
            status = True
            msg = "success"
        case _:
            status = False
            msg = f"Either points/label/rects is None or len(points) != len(labels), {anno_id=}"
    return SamReturn(
        anno_id=anno_id,
        status=status,
        msg=msg,
        mode=auto_mode.name,
        rects=worker_result,
    )
