import asyncio
from io import BytesIO
import json
from pathlib import Path
from typing import List, Annotated
import subprocess

import numpy as np
import requests
from fastapi import Depends, FastAPI, File, Form, UploadFile, Header, status, Response
from PIL import Image
from rich import print

from data_struct import Annotation, Point, Rect, SamReturn, annotation_checker
from sam_onnx import SamOnnxModel, EdgeSam
from worker import AutoMode, ZSamWorker
from app.db import get_tasks, insert_link_table

ALIST_HOST = "http://127.0.0.1:5244"
VERSION = "1.0.0"
# ENCODER_PATH = "assets/sam_vit_h_encoder_quantized.onnx"
# DECODER_PATH = "assets/sam_vit_h_decoder_quantized.onnx"
ENCODER_PATH = "assets/edge_sam_3x_encoder.onnx"
DECODER_PATH = "assets/edge_sam_3x_decoder.onnx"

app = FastAPI()
# SAM_MODEL = SamOnnxModel(ENCODER_PATH, DECODER_PATH)
SAM_MODEL = EdgeSam(ENCODER_PATH, DECODER_PATH)

IMAGE_CACHE = {}


USERS = [
    "lyl",
    "wzh",
    "zhq",
    "csy",
    "open_labeling",
]


@app.get("/")
async def root():
    return {"message": "Welcome to SamServer!", "version": f"v{VERSION}"}


@app.post("/api/v1/login", status_code=status.HTTP_200_OK, response_class=Response)
async def login(username: str = Form(...), password: str = Form("123456")):
    if username not in USERS:
        payload = json.dumps({"username": "open_labeling", "password": "123456"})
    else:
        payload = json.dumps({"username": username, "password": password})
    url = f"{ALIST_HOST}/api/auth/login"
    headers = {
        "User-Agent": f"SAMServer/v{VERSION}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(url, data=payload, headers=headers)
        if resp.status_code != 200 or resp.json()["code"] != 200:
            return Response(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=resp.content,
                media_type="application/json",
            )
        return Response(
            json.dumps({"token": resp.json()["data"]["token"]}),
            media_type="application/json",
        )
    except Exception as e:
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=str(e),
            media_type="text/plain",
        )


@app.get(
    "/api/v1/get_image/{name}",
    responses={
        200: {"content": {"image/png": {}}},
        500: {"content": {"application/json": {}}},
    },
    response_class=Response,
)
async def get_image(name: str, authorization: str = Header(None)):
    url = f"{ALIST_HOST}/p/datasets/seeds_data/exported_pngs/{name}"
    headers = {
        "User-Agent": f"SAMServer/v{VERSION}",
        "Authorization": authorization,
    }
    if name in IMAGE_CACHE:
        return Response(content=IMAGE_CACHE[name], media_type="image/png")
    try:
        resp = requests.get(url, headers=headers, stream=True)
        if resp.status_code != 200 or resp.headers["Content-Type"] != "image/png":
            return Response(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=resp.content,
                media_type="application/json",
            )

        IMAGE_CACHE[name] = resp.content
        await _set_model_image(resp.content)

        return Response(content=resp.content, media_type="image/png")
    except Exception as e:
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=str(e),
            media_type="application/json",
        )


@app.put("/api/v1/save_zlabel")
async def save_zlabel(
    zlabel: bytes = Form(...),
    username: str = Form(...),
    filename: str = Form(...),
    authorization: str = Header(None),
):
    url = f"{ALIST_HOST}/api/fs/put"
    headers = {
        "User-Agent": f"SAMServer/v{VERSION}",
        "Authorization": authorization,
        "Content-Type": "text/plain",
        "File-Path": "",
    }

    anno = json.loads(zlabel.decode("utf-8"))
    insert_link_table(
        anno["id"],
        user_name=username,
    )

    msg = []
    resp = None
    headers[
        "File-Path"
    ] = f"/datasets/seeds_data/exported_pngs_label/{Path(filename).name}"
    try:
        resp = requests.put(url, data=zlabel, headers=headers)
        if resp.status_code == 200:
            if resp.json()["message"] == "success":
                msg.append({"status": True, "msg": "success"})
        else:
            msg.append({"status": False, "msg": resp.text})
    except Exception as e:
        msg.append({"status": False, "msg": str(e)})
    return Response(content=json.dumps(msg), media_type="application/json")


@app.get("/api/v1/get_zlabel/{name}", response_class=Response)
async def get_zlabel(name: str, authorization: str = Header(None)):
    url = f"{ALIST_HOST}/d/datasets/seeds_data/exported_pngs_label/{name}"
    headers = {
        "User-Agent": f"SAMServer/v{VERSION}",
        "Authorization": authorization,
    }
    try:
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200 or resp.json().get("code", 200) != 200:
            return Response(
                status_code=status.HTTP_404_NOT_FOUND,
                content=resp.content,
                media_type="application/json",
            )
        return Response(content=resp.content, media_type="application/json")
    except Exception as e:
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=str(e),
            media_type="text/plain",
        )


@app.get("/api/v1/get_tasks")
async def _get_tasks(num: int = 30, finished: int = 1):
    """
    finished: -1: all, 0: unfinished, 1: finished
    """
    tasks = get_tasks(num, finished)
    res = [
        {
            "id": task.id,
            "anno_id": task.anno_id,
            "filename": task.filename,
            "labels": [label.name for label in task.labels],
            "finished": task.finished,
        }
        for task in tasks
    ]
    return res


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


@app.post("/api/v0/predict")
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


@app.post("/api/v1/predict")
async def predict_v1(
    anno: Annotation = Depends(annotation_checker),
    threshold: int = Form(100),
    mode: int = Form(1),
    image_name: str = Form(...),
    authorization: str = Header(None),
):
    resp = await get_image(image_name, authorization)
    if resp.status_code != 200:
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=resp.body,
            media_type="application/json",
        )
    try:
        img = Image.open(BytesIO(resp.body))
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
    except Exception as e:
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=str(e),
            media_type="application/json",
        )


async def _set_model_image(img: bytes):
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        SAM_MODEL.encode,
        np.asarray(Image.open(BytesIO(img)), dtype=np.uint8),
    )


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
