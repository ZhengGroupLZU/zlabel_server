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
from rich import print  # noqa: F401

import app.db as db
from app.config import SETTINGS
from app.logger import ZLogger
from app.openlist_api import OpenListAPIError, OpenListClient
from app.sam_ort import Predictor
from app.worker import AutoMode, ReturnType, ZSamWorker
from app.ztypes import Annotation, Point, Rect, SamReturn, annotation_checker

oplist_client = OpenListClient(SETTINGS.oplist_host)
app = FastAPI()

logger = ZLogger("ZLabelServer")

SAM_MODEL = Predictor(
    model_dir=SETTINGS.model_dir,
    model_name=SETTINGS.model_name,
    backend=SETTINGS.ort_backend,
    threads=SETTINGS.ort_threads,
    conf=SETTINGS.sam3_conf,
    iou=SETTINGS.sam3_iou,
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
            logger.debug(traceback.format_exc())
            return JSONResponse(
                status_code=e.status_code or status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": str(e), "data": None},
                media_type="application/json",
            )
        except requests.exceptions.HTTPError as e:
            logger.debug(traceback.format_exc())
            return JSONResponse(
                status_code=e.response.status_code,
                content={"message": str(e), "data": None},
                media_type="application/json",
            )
        except Exception as e:
            logger.debug(traceback.format_exc())
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
    path = f"{name}"
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
        logger.debug(traceback.format_exc())
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


@app.api_route(
    "/api/v1/refresh_tasks",
    methods=["GET", "POST"],
    response_class=JSONResponse,
)
async def refresh_tasks(username: str = Query(...), password: str = Query(...)):
    response = oplist_client.auth.login(username, password)
    if response.data.token is None:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"message": "failed", "data": "Unauthorized"},
            media_type="application/json",
        )
    allowed_image_ext = [".png", ".jpg", ".jpeg"]
    oplist_client.set_token(response.data.token)
    project_list = []

    projects = oplist_client.fs.dirs(SETTINGS.oplist_proj_dir)
    for project in projects.data:
        img_files = oplist_client.fs.glob(f"{SETTINGS.oplist_proj_dir}/{project.name}", "*")
        # logger.debug(img_files)
        img_files_filtered = []
        for img_file in img_files:
            if any(img_file.lower().endswith(ext) for ext in allowed_image_ext):
                img_files_filtered.append(img_file)
        project_list.append(
            {
                "name": project.name,
                "files": img_files_filtered,
            }
        )
        db.create_or_update_projects(project_list)

    return JSONResponse(
        content={"message": "success", "data": None},
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@app.get("/api/v1/get_tasks")
async def get_tasks(
    project_id: int = -1,
    num: int = 30,
    finished: int = -1,
    random: bool = True,
    authorization: str = Header(None),
):
    """
    finished: -1: all, 0: unfinished, 1: finished
    """
    # await refresh_tasks(authorization)

    try:
        oplist_client.set_token(authorization)
        resp = oplist_client.auth.get_current_user()
        if not resp.data.id:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"message": "failed", "data": "Unauthorized"},
                media_type="application/json",
            )
        tasks = db.get_tasks(project_id, num, finished, random)
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
    except Exception as e:
        logger.debug(traceback.format_exc())
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": str(e), "data": None},
            media_type="application/json",
        )


@app.get("/api/v1/get_projects")
async def get_projects():
    projects = db.get_projects()
    res = [
        {
            "id": project.id,
            "name": project.name,
        }
        for project in projects
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
async def set_image(image: UploadFile = File(...), image_name: str = Form("")):
    content = await image.read()
    if image_name:
        await _cache_image(image_name, content)
    await _set_model_image(content)
    return JSONResponse(
        content={"message": "success", "data": None},
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@app.post("/api/v1/predict")
async def predict_v1(
    anno: Annotation = Depends(annotation_checker),
    image: UploadFile | None = File(None),
    threshold: int = Form(100),
    mode: int = Form(1),
    image_name: str = Form(...),
    authorization: str = Header(None),
    return_type: int = Form(1),  # RECT = 1 POLYGON = 2 RLE = 3
):
    if image is not None:
        # direct image upload: cache the bytes (so get_image returns them) and
        # set the model image; no oplist round-trip needed
        content = await image.read()
        await _cache_image(image_name, content)
        await _set_model_image(content)
    else:
        resp = await get_image(image_name, authorization)
        if resp.status_code != 200:
            return resp
        content = resp.body

    @oplist_client_try_run
    def predict_func():
        img = Image.open(BytesIO(content))
        result = _predict(
            anno.id,
            img,
            anno.points,
            anno.labels,
            anno.rects,
            anno.texts,
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
        SAM_MODEL.set_image,
        _to_bgr(np.asarray(Image.open(BytesIO(img)), dtype=np.uint8)),
    )


def _to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert an RGB array (PIL convention) to BGR (OpenCV convention)."""
    return img[..., ::-1].copy()


def _predict(
    anno_id: str,
    img: Image.Image,
    points: list[Point] | None,
    labels: list[float] | None,
    rects: list[Rect] | None,
    texts: list[str] | None,
    threshold: int,
    auto_mode: AutoMode,
    return_type: ReturnType,
) -> SamReturn:
    status = False
    msg = ""
    worker_result = None
    img_bgr = _to_bgr(np.asarray(img, dtype=np.uint8))
    match (points, labels, rects, texts):
        case (_, _, _, t) if t is not None:
            worker = ZSamWorker(
                model=SAM_MODEL,
                anno_id=anno_id,
                img=img_bgr,
                auto_mode=auto_mode,
                threshold=threshold,
                return_type=return_type,
                min_contour_area_ratio=SETTINGS.min_contour_area_ratio,
                contour_min_points=SETTINGS.contour_min_points,
                contour_max_points=SETTINGS.contour_max_points,
                contour_max_iterations=SETTINGS.contour_max_iterations,
            )
            worker_result = worker.run_text(t)
            status = True
            msg = "success"
        case (p, l, r, _) if p is not None and l is not None and len(p) == len(l):
            worker = ZSamWorker(
                model=SAM_MODEL,
                anno_id=anno_id,
                img=img_bgr,
                auto_mode=auto_mode,
                threshold=threshold,
                return_type=return_type,
                min_contour_area_ratio=SETTINGS.min_contour_area_ratio,
                contour_min_points=SETTINGS.contour_min_points,
                contour_max_points=SETTINGS.contour_max_points,
                contour_max_iterations=SETTINGS.contour_max_iterations,
            )
            worker_result = worker.run_point(p, l)
            status = True
            msg = "success"
        case (p, l, r, _) if r is not None:
            worker = ZSamWorker(
                model=SAM_MODEL,
                anno_id=anno_id,
                img=img_bgr,
                auto_mode=auto_mode,
                threshold=threshold,
                return_type=return_type,
                min_contour_area_ratio=SETTINGS.min_contour_area_ratio,
                contour_min_points=SETTINGS.contour_min_points,
                contour_max_points=SETTINGS.contour_max_points,
                contour_max_iterations=SETTINGS.contour_max_iterations,
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
