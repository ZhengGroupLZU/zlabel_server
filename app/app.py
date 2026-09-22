import asyncio
import json
import time
import traceback
from collections import OrderedDict
from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from io import BytesIO

import numpy as np
import requests
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
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
from app.project_scan import ScanResult, discover_projects
from app.sam_ort import Predictor
from app.worker import AutoMode, ReturnType, ZSamWorker
from app.ztypes import Annotation, Point, Rect, SamReturn, annotation_checker

oplist_client = OpenListClient(SETTINGS.oplist_host)

logger = ZLogger("ZLabelServer")


# --------------------------------------------------------------------------- #
# Project auto-discovery
# --------------------------------------------------------------------------- #
# minimum seconds between scans triggered on GET /api/v1/get_projects, to avoid
# hammering OpenList when the client polls; the periodic background scan always
# runs on its own interval.
_PROJECT_SCAN_THROTTLE = 2.0
_project_scan_lock = asyncio.Lock()
_last_project_scan = 0.0
_scan_unavailable = False


def _ensure_service_token() -> str:
    """Return a valid OpenList token for the background scanner.

    Precedence: an already-set token, then a configured static token
    (``ZLSERVER_OPLIST_TOKEN``), then a username/password login. The static
    token avoids needing a real OpenList user account for the scanner.
    """
    if oplist_client.token:
        return oplist_client.token
    if SETTINGS.oplist_token:
        oplist_client.set_token(SETTINGS.oplist_token)
        return oplist_client.token
    if not SETTINGS.oplist_username or not SETTINGS.oplist_password:
        raise RuntimeError("no OpenList token and no ZLSERVER_OPLIST_USERNAME/PASSWORD configured")
    resp = oplist_client.auth.login(SETTINGS.oplist_username, SETTINGS.oplist_password)
    oplist_client.set_token(resp.data.token)
    return oplist_client.token


def _scan_and_sync() -> ScanResult:
    """Run one project scan and persist the result (blocks; runs in a thread)."""
    _ensure_service_token()
    result = discover_projects(oplist_client, SETTINGS)
    db.sync_projects_from_scan(result.projects, result.present_dirs, result.confirmed_missing)
    return result


async def _maybe_scan(force: bool = False) -> None:
    """Run a project scan unless one ran very recently."""
    global _last_project_scan, _scan_unavailable
    now = time.monotonic()
    if not force and now - _last_project_scan < _PROJECT_SCAN_THROTTLE:
        return
    async with _project_scan_lock:
        if not force and time.monotonic() - _last_project_scan < _PROJECT_SCAN_THROTTLE:
            return
        try:
            await asyncio.to_thread(_scan_and_sync)
        except Exception as e:
            # avoid spamming the log every interval while OpenList/auth is down
            if not _scan_unavailable:
                _scan_unavailable = True
                logger.warning(f"project scan unavailable: {e}")
            logger.debug(traceback.format_exc())
        else:
            if _scan_unavailable:
                _scan_unavailable = False
                logger.warning("project scan recovered")
        _last_project_scan = time.monotonic()


async def _periodic_project_scan() -> None:
    """Background task that rescans projects every ``project_scan_interval``."""
    while True:
        await asyncio.sleep(SETTINGS.project_scan_interval)
        await _maybe_scan(force=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    tasks: list[asyncio.Task] = []
    if SETTINGS.project_scan_interval > 0:
        tasks.append(asyncio.create_task(_periodic_project_scan()))
    # initial scan at startup (fire-and-forget; must not block startup)
    tasks.append(asyncio.create_task(_maybe_scan(force=True)))
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task


app = FastAPI(lifespan=lifespan)

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
            code = getattr(e.response, "status_code", None) or status.HTTP_500_INTERNAL_SERVER_ERROR
            return JSONResponse(
                status_code=code,
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


AUTH_CACHE_TTL_SECONDS = 60.0
_auth_cache: dict[str, float] = {}


async def require_user(authorization: str = Header(None)) -> str:
    """FastAPI dependency: the request must carry a valid OpenList token.

    Verification hits OpenList, so a positive result is cached briefly; that
    keeps ``/predict`` (called once per click) cheap.
    """
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing token")
    now = time.monotonic()
    if now - _auth_cache.get(authorization, 0.0) < AUTH_CACHE_TTL_SECONDS:
        return authorization

    def _check() -> bool:
        oplist_client.set_token(authorization)
        resp = oplist_client.auth.get_current_user()
        return bool(resp.data and resp.data.id)

    try:
        ok = await asyncio.to_thread(_check)
    except Exception as e:
        logger.warning(f"Token verification failed: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token") from e
    if not ok:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")
    if len(_auth_cache) > 1024:  # keep the cache bounded
        _auth_cache.clear()
    _auth_cache[authorization] = now
    return authorization


def annotation_timestamp(anno: dict) -> datetime | None:
    """``updated_at`` (falling back to ``created_at``) of an annotation."""
    raw = anno.get("updated_at") or anno.get("created_at")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except ValueError:
        return None


def detect_conflict(file_path: str, incoming: dict) -> dict | None:
    """Optimistic locking: refuse to overwrite a *newer* stored annotation."""
    try:
        raw = oplist_client.fs.get_file_bytes(file_path)
    except OpenListAPIError as e:
        if e.status_code == 404:
            return None  # nothing stored yet
        raise
    try:
        existing = json.loads(raw.decode("utf-8"))
    except Exception:
        return None  # unreadable stored copy: let the client re-save it

    incoming_ts = annotation_timestamp(incoming)
    existing_ts = annotation_timestamp(existing)
    if incoming_ts is None or existing_ts is None:
        return None  # legacy annotations without timestamps
    try:
        newer = incoming_ts < existing_ts
    except TypeError:  # naive vs tz-aware mix
        return None
    if not newer:
        return None
    updated_by = existing.get("updated_by") or {}
    return {
        "updated_at": existing_ts.isoformat(),
        "updated_by": updated_by.get("name", "") if isinstance(updated_by, dict) else "",
    }


def extract_label_names(anno: dict) -> list[str]:
    """Label names used by an annotation (client Annotation.results[*].labels)."""
    names: list[str] = []
    results = anno.get("results") or {}
    for result in results.values() if isinstance(results, dict) else results:
        for label in (result or {}).get("labels") or []:
            name = (label or {}).get("name", "")
            if name and name not in names:
                names.append(name)
    return names


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
    except OpenListAPIError as e:
        # pass OpenList's status through: a missing image must be a 404 (the
        # desktop distinguishes "not there" from "server broken")
        logger.debug(traceback.format_exc())
        return JSONResponse(
            status_code=e.status_code or status.HTTP_500_INTERNAL_SERVER_ERROR,
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


@app.put("/api/v1/save_zlabel")
async def save_zlabel(
    zlabel: bytes = Form(...),
    username: str = Form(...),
    filename: str = Form(...),
    project: str = Form(""),
    force: bool = Form(False),
    authorization: str = Depends(require_user),
):
    @oplist_client_try_run
    def save_zlabel_func():
        oplist_client.set_token(authorization)
        file_path = f"{SETTINGS.zlabel_save_dir(project)}/{filename}"

        try:
            anno = json.loads(zlabel.decode("utf-8"))
        except Exception as e:
            logger.error(f"save_zlabel invalid annotation json, {file_path=}, {e=}")
            return JSONResponse(
                content={"message": f"invalid annotation json: {e}", "data": None},
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                media_type="application/json",
            )

        if not force:
            conflict = detect_conflict(file_path, anno)
            if conflict is not None:
                logger.info(f"save_zlabel conflict, {file_path=}, server={conflict['updated_at']}")
                return JSONResponse(
                    content={"message": "annotation conflict", "data": conflict},
                    status_code=status.HTTP_409_CONFLICT,
                    media_type="application/json",
                )

        resp = oplist_client.fs.stream_upload(file_path, BytesIO(zlabel), as_task=False)
        if not (resp.code == 200 and resp.message == "success"):
            # never mark the task finished when the file was not stored
            logger.error(f"save_zlabel upload failed, {file_path=}, {resp.message=}")
            return JSONResponse(
                content={"message": resp.message, "data": None},
                status_code=status.HTTP_502_BAD_GATEWAY,
                media_type="application/json",
            )

        # persist only after the annotation is really stored
        db.insert_link_table(
            anno.get("id", ""),
            label_names=extract_label_names(anno),
            user_name=username,
        )
        return JSONResponse(
            content={"message": "success", "data": None},
            status_code=status.HTTP_200_OK,
            media_type="application/json",
        )

    return save_zlabel_func()


@app.api_route(
    "/api/v1/get_zlabel",
    methods=["GET", "POST"],
    response_class=JSONResponse,
)
async def get_zlabel(
    name: str,
    project: str = Query(""),
    authorization: str = Header(None),
):
    @oplist_client_try_run
    def get_zlabel_func():
        oplist_client.set_token(authorization)

        file_bytes = oplist_client.fs.get_file_bytes(
            f"{SETTINGS.zlabel_save_dir(project)}/{name}"
        )
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
    oplist_client.set_token(response.data.token)

    # marker-based discovery (only top-level dirs with the marker become projects)
    result = discover_projects(oplist_client, SETTINGS)
    db.sync_projects_from_scan(result.projects, result.present_dirs, result.confirmed_missing)

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
    # trigger a project scan; the response is always the active project list,
    # even when OpenList is temporarily unavailable.
    await _maybe_scan()
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


@app.get("/api/v1/labels")
async def get_labels(project: str = ""):
    """Labels attached to a project's annotations (all labels when empty)."""
    return JSONResponse(
        content={"message": "success", "data": db.get_labels(project)},
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@app.get("/api/v1/how-many-finished")
async def how_many_finished(project: str = ""):
    """``{"finished": n, "total": m}`` for a project (or every active project)."""
    return JSONResponse(
        content={"message": "success", "data": db.get_progress(project)},
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )


@app.post("/api/v1/set_image")
async def set_image(
    image: UploadFile = File(...),
    image_name: str = Form(""),
    authorization: str = Depends(require_user),  # noqa: ARG001 (auth gate only)
):
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
    authorization: str = Depends(require_user),
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
    """Set the model image and *wait* for the embedding.

    It used to be fire-and-forget, so a predict right after a set-image raced the
    embedding (the model could still run on the previous frame).
    """
    arr = _to_bgr(np.asarray(Image.open(BytesIO(img)), dtype=np.uint8))
    await asyncio.to_thread(SAM_MODEL.set_image, arr)


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
