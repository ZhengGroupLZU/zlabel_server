import asyncio
import json
import traceback
from collections import OrderedDict
from collections.abc import Callable
from io import BytesIO

import numpy as np
import requests
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    Query,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from PIL import Image
from rich import print

import app.db as db
from app.config import SETTINGS
from app.openlist_api import OpenListAPIError, OpenListClient
from app.sam_onnx import EdgeSam, SamOnnxModel
from app.worker import AutoMode, ReturnType, ZSamWorker
from app.ztypes import Annotation, Point, Rect, SamReturn, annotation_checker

oplist_client = OpenListClient(SETTINGS.oplist_host)
app = FastAPI()


SAM_MODEL = (
    SamOnnxModel(SETTINGS.encoder_path, SETTINGS.decoder_path)
    if SETTINGS.model_name == "SAM"
    else EdgeSam(SETTINGS.encoder_path, SETTINGS.decoder_path)
)


IMAGE_CACHE = OrderedDict()


async def _cache_image(image_id: str, image: bytes):
    if len(IMAGE_CACHE) >= SETTINGS.image_cache_size:
        IMAGE_CACHE.popitem(last=False)
    IMAGE_CACHE[image_id] = image


def oplist_client_try_run(func: Callable[..., JSONResponse | Response]):
    def wrapper(*args, **kwargs) -> JSONResponse | Response:
        try:
            return func(*args, **kwargs)
        except OpenListAPIError as e:
            print(traceback.format_exc())
            return JSONResponse(
                status_code=e.status_code or status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": str(e), "data": None},
                media_type="application/json",
            )
        except requests.exceptions.HTTPError as e:
            print(traceback.format_exc())
            return JSONResponse(
                status_code=e.response.status_code,
                content={"message": str(e), "data": None},
                media_type="application/json",
            )
        except Exception as e:
            print(traceback.format_exc())
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": str(e), "data": None},
                media_type="application/json",
            )

    return wrapper


@app.get("/")
async def root():
    return {"message": "Welcome to SamServer!", "version": f"v{SETTINGS.version}"}


@app.api_route(
    "/api/v1/login",
    methods=["GET", "POST"],
    response_class=JSONResponse,
)
async def login(username: str = Query(...), password: str = Query(...)):
    @oplist_client_try_run
    def login_func():
        response = oplist_client.auth.login(username, password)
        return JSONResponse(
            {
                "message": "success",
                "data": {"username": username, "token": response.data.token},
            },
            status_code=status.HTTP_200_OK,
            media_type="application/json",
        )

    return login_func()


@app.api_route(
    "/api/v1/me",
    methods=["GET", "POST"],
    response_class=JSONResponse,
)
async def get_current_user(authorization: str = Header(None)):
    @oplist_client_try_run
    def get_current_user_func():
        oplist_client.set_token(authorization)
        resp = oplist_client.auth.get_current_user()
        return JSONResponse(
            {
                "message": "success",
                "data": resp.data.model_dump(),
            },
            media_type="application/json",
        )

    return get_current_user_func()


@app.api_route(
    "/api/v1/list_projects",
    methods=["GET", "POST"],
    response_class=JSONResponse,
)
async def list_projects(authorization: str = Header(None)):
    @oplist_client_try_run
    def list_projects_func():
        oplist_client.set_token(authorization)
        resp = oplist_client.fs.ls(SETTINGS.oplist_proj_dir)
        return JSONResponse(
            {
                "message": "success",
                "data": resp.data.model_dump(),
            },
            media_type="application/json",
        )

    return list_projects_func()


@app.api_route(
    "/api/v1/list_files",
    methods=["GET", "POST"],
    status_code=status.HTTP_200_OK,
    response_class=JSONResponse,
)
async def list_files(
    project: str,
    authorization: str = Header(None),
):
    @oplist_client_try_run
    def list_files_func():
        oplist_client.set_token(authorization)
        resp = oplist_client.fs.ls(f"{SETTINGS.oplist_proj_dir}/{project}")

        return JSONResponse(
            {
                "message": "success",
                "data": resp.data.model_dump(),
            },
            media_type="application/json",
        )

    return list_files_func()


@app.api_route(
    "/api/v1/get_file_info",
    methods=["GET", "POST"],
)
async def get_file_info(
    path: str,
    authorization: str = Header(None),
):
    @oplist_client_try_run
    def get_file_info_func():
        oplist_client.set_token(authorization)
        resp = oplist_client.fs.get(path)

        return JSONResponse(
            {
                "message": "success",
                "data": resp.model_dump(),
            },
            media_type="application/json",
        )

    return get_file_info_func()


@app.api_route(
    "/api/v1/get_image",
    methods=["GET", "POST"],
    response_class=Response,
)
async def get_image(
    name: str,
    authorization: str = Header(None),
):
    path = f"{SETTINGS.oplist_proj_dir}/{SETTINGS.oplist_proj_name}/{name}"
    if path in IMAGE_CACHE:
        return Response(content=IMAGE_CACHE[path], media_type="image/png")
    try:
        oplist_client.set_token(authorization)
        img_bytes = oplist_client.fs.get_file_bytes(path)

        await _cache_image(path, img_bytes)
        await _set_model_image(img_bytes)

        return Response(
            content=img_bytes,
            status_code=status.HTTP_200_OK,
            media_type="image/png",
        )
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": str(e), "data": None},
            media_type="application/json",
        )


@app.put("/api/v1/save_zlabel")
async def save_zlabel(
    zlabel: bytes = Form(...),
    username: str = Form(...),
    filename: str = Form(...),
    authorization: str = Header(None),
):
    @oplist_client_try_run
    def save_zlabel_func():
        oplist_client.set_token(authorization)
        file_path = f"{SETTINGS.oplist_zlabel_save_dir}/{filename}"
        resp = oplist_client.fs.stream_upload(file_path, BytesIO(zlabel), as_task=False)
        if resp.code == 200 and resp.message == "success":
            msg = {"message": "success", "data": None}
        else:
            msg = {"message": resp.message, "data": None}

        # save to local database
        anno = json.loads(zlabel.decode("utf-8"))
        db.insert_link_table(
            anno["id"],
            user_name=username,
        )

        return JSONResponse(
            content=msg,
            status_code=status.HTTP_200_OK,
            media_type="application/json",
        )

    return save_zlabel_func()


@app.api_route(
    "/api/v1/get_zlabel",
    methods=["GET", "POST"],
    response_class=JSONResponse,
)
async def get_zlabel(name: str, authorization: str = Header(None)):
    @oplist_client_try_run
    def get_zlabel_func():
        oplist_client.set_token(authorization)

        file_bytes = oplist_client.fs.get_file_bytes(f"{SETTINGS.oplist_zlabel_save_dir}/{name}")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=json.loads(file_bytes.decode("utf-8")),
        )

    return get_zlabel_func()


async def refresh_tasks(authorization: str):
    allowed_image_ext = [".png", ".jpg", ".jpeg"]
    oplist_client.set_token(authorization)
    project_list = []

    projects = oplist_client.fs.dirs(SETTINGS.oplist_proj_dir)
    for project in projects.data:
        img_files = oplist_client.fs.ls(f"{SETTINGS.oplist_proj_dir}/{project.name}")
        img_files_filtered = []
        for img_file in img_files.data.get_files():
            if any(img_file.name.lower().endswith(ext) for ext in allowed_image_ext):
                img_files_filtered.append(img_file.name)
        project_list.append(
            {
                "name": project.name,
                "files": img_files_filtered,
            }
        )
        db.create_or_update_projects(project_list)


@app.get("/api/v1/get_tasks")
async def get_tasks(num: int = 30, finished: int = -1, authorization: str = Header(None)):
    """
    finished: -1: all, 0: unfinished, 1: finished
    """
    await refresh_tasks(authorization)
    tasks = db.get_tasks(num, finished)
    res = [
        {
            "id": task.id,
            "project_id": task.project_id,
            "anno_id": task.anno_id,
            "filename": task.filename,
            "labels": [label.name for label in task.labels],
            "finished": task.finished,
        }
        for task in tasks
    ]
    return JSONResponse(
        content={"message": "success", "data": res},
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@app.get("/api/v1/how-many-finished")
async def how_many_finished():
    n = db.how_many_finished()
    return JSONResponse(
        content={"message": "success", "data": n},
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@app.post("/api/v1/set_image")
async def set_image(image: UploadFile = File(...)):
    content = await image.read()
    img = Image.open(BytesIO(content))
    SAM_MODEL.encode(np.asarray(img, dtype=np.uint8))
    return JSONResponse(
        content={"message": "success", "data": None},
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@app.post("/api/v1/predict")
async def predict_v1(
    anno: Annotation = Depends(annotation_checker),
    threshold: int = Form(100),
    mode: int = Form(1),
    image_name: str = Form(...),
    authorization: str = Header(None),
    return_type: int = Form(1),  # RECT = 1 POLYGON = 2 RLE = 3
):
    resp = await get_image(image_name, authorization)
    if resp.status_code != 200:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=resp.body,
            media_type="application/json",
        )

    @oplist_client_try_run
    def predict_func():
        img = Image.open(BytesIO(resp.body))
        result = _predict(
            anno.id,
            img,
            anno.points,
            anno.labels,
            anno.rects,
            threshold,
            AutoMode(mode),
            ReturnType(return_type),
        )
        return JSONResponse(
            content={"message": "success", "data": result.model_dump()},
            status_code=status.HTTP_200_OK,
            media_type="application/json",
        )

    return predict_func()


async def _set_model_image(img: bytes) -> None:
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        SAM_MODEL.encode,
        np.asarray(Image.open(BytesIO(img)), dtype=np.uint8),
    )


def _predict(
    anno_id: str,
    img: Image.Image,
    points: list[Point] | None,
    labels: list[float] | None,
    rects: list[Rect] | None,
    threshold: int,
    auto_mode: AutoMode,
    return_type: ReturnType,
) -> SamReturn:
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
                return_type=return_type,
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
                return_type=return_type,
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
        mode=auto_mode.name,  # type: ignore
        data=worker_result,
    )
