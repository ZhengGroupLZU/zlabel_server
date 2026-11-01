"""Pull mode (``INLINE_IMAGES=false``) needs a free event loop while the worker answers.

The worker fetches the task image back from the API *while the API is waiting for it*, so
the predict handler must not block the event loop. It is a sync handler on purpose:
FastAPI runs those in a threadpool. This test starts the real app on a socket with a
stand-in worker that pulls the job's ``image_url``, which is exactly the production
sequence - an ``async def`` handler deadlocks here and the pull times out (503).
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import requests
import uvicorn
from PIL import Image

from app.app import create_app
from app.core.config import Settings
from app.db.base import Database

TOKEN = "pull-mode-token"
ADMIN, PASSWORD = "pulladmin", "pull-password"
PROJECT = "pullproj"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class PullingWorker:
    """Minimal /infer endpoint that pulls the image back from the API, like the real one."""

    def __init__(self, port: int) -> None:
        self.outcome = ""
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                url = body.get("image_url") or ""
                try:
                    resp = requests.get(
                        f"{body['_api']}{url}",
                        headers={"Authorization": f"Bearer {TOKEN}"},
                        timeout=5.0,
                    )
                except Exception as e:  # noqa: BLE001 - the failure this test is about
                    outer.outcome = f"pull failed: {type(e).__name__}"
                    self._reply(503, {"code": "inference_unavailable", "message": outer.outcome})
                    return
                outer.outcome = f"pulled {len(resp.content)} bytes"
                self._reply(
                    200,
                    {
                        "anno_id": body["anno_id"],
                        "status": True,
                        "mode": "SAM",
                        "msg": "success",
                        "data": [{"points": [{"x": 1, "y": 1}]}],
                    },
                )

            def _reply(self, code: int, payload: dict) -> None:
                raw = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.server.shutdown()


class PullingInference:
    """Client-side stand-in for the API's InferenceClient: forwards the job to the worker."""

    def __init__(self, worker_url: str, api_url: str) -> None:
        self.worker_url = worker_url
        self.api_url = api_url

    def infer(self, job: dict) -> dict:
        job = {**job, "_api": self.api_url}
        return requests.post(f"{self.worker_url}/infer", json=job, timeout=20).json()


@pytest.fixture
def live_api(tmp_path):
    """The app served on a real socket (a TestClient cannot be called back into)."""
    storage = tmp_path / "storage"
    (storage / PROJECT / "images" / "dish01").mkdir(parents=True)
    Image.fromarray(np.full((64, 96, 3), 130, np.uint8)).save(storage / PROJECT / "images" / "dish01" / "D1.png")
    api_port = free_port()
    worker = PullingWorker(free_port())
    settings = Settings(
        database_url=f"sqlite+pysqlite:///{tmp_path / 'pull.db'}",
        upload_dir=str(tmp_path / "uploads"),
        storage_root=str(storage),
        bootstrap_admin=ADMIN,
        bootstrap_password=PASSWORD,
        inference_url=f"http://127.0.0.1:{worker.server.server_address[1]}",
        inference_token=TOKEN,
        inference_inline_images=False,
        inference_timeout=20.0,
    )
    database = Database(settings.database_url)
    database.create_all()
    app = create_app(settings, database=database)
    base = f"http://127.0.0.1:{api_port}"
    app.state.services.inference = PullingInference(settings.inference_url, base)

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=api_port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            requests.get(f"{base}/api/v2/health", timeout=0.5)
            break
        except requests.RequestException:
            time.sleep(0.05)
    yield base, worker
    server.should_exit = True
    thread.join(timeout=10)
    worker.stop()


def test_a_predict_can_pull_its_own_image_back(live_api):
    base, worker = live_api
    login = requests.post(f"{base}/api/v2/auth/login", json={"username": ADMIN, "password": PASSWORD}, timeout=10)
    assert login.status_code == 200, login.text
    headers = {"Authorization": f"Bearer {login.json()['token']}"}
    requests.post(f"{base}/api/v2/projects/scan", headers=headers, json={}, timeout=20)
    items = requests.get(f"{base}/api/v2/projects/{PROJECT}/tasks", headers=headers, timeout=10).json()["items"]
    task = items[0]

    started = time.perf_counter()
    resp = requests.post(
        f"{base}/api/v2/projects/{PROJECT}/predict",
        data={
            "data": json.dumps(
                {"anno_id": task["anno_id"], "rel_path": task["rel_path"], "points": [{"x": 5, "y": 5}],
                 "labels": [1], "mode": 1, "return_type": 2}
            )
        },
        headers=headers,
        timeout=30,
    )
    elapsed = time.perf_counter() - started

    # the handler blocks on the worker, which blocks on us: it only works when the
    # event loop is free, i.e. when the handler is a sync function (threadpool)
    assert resp.status_code == 200, resp.text
    assert worker.outcome.startswith("pulled"), worker.outcome
    assert elapsed < 10, f"predict took {elapsed:.1f}s - the pull probably deadlocked"


def test_predict_stays_a_sync_handler():
    """Guards the invariant above: a sync handler leaves the loop free for the pull."""
    import inspect

    from fastapi.routing import APIRoute

    from app.api.v2 import predict as predict_module

    routes = [r for r in predict_module.router.routes if isinstance(r, APIRoute) and r.path.endswith("/predict")]
    assert routes
    assert not inspect.iscoroutinefunction(routes[0].endpoint), (
        "predict must stay a sync handler: an async one blocks the event loop while the "
        "inference worker pulls its image back from this process (INLINE_IMAGES=false)"
    )
