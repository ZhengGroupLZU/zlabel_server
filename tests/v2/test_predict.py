"""Stateless prediction: the job always carries its own frame."""

from __future__ import annotations

import base64
import json

from tests.v2.fakes import FakeInference
from tests.v2.test_projects import seed
from v2.core.errors import InferenceUnavailable

PREDICT = "/api/v2/projects/projA/predict"


def bootstrap(client, auth_headers, ol, services) -> dict:
    seed(ol, "projA", files=("images/dish01/D1.png", "images/dish01/D2.png"))
    headers = auth_headers(client)
    client.post("/api/v2/projects/scan", headers=headers)
    items = client.get("/api/v2/projects/projA/tasks", headers=headers).json()["items"]
    fake = FakeInference()
    services.inference = fake
    return {"headers": headers, "items": items, "inference": fake}


def payload(anno_id: str, **extra) -> str:
    body = {
        "anno_id": anno_id,
        "points": [{"x": 10, "y": 20}],
        "labels": [1],
        "threshold": 100,
        "mode": 1,
        "return_type": 2,
    }
    body.update(extra)
    return json.dumps(body)


def test_predict_with_an_uploaded_frame(client, auth_headers, ol, services):
    ctx = bootstrap(client, auth_headers, ol, services)
    first = ctx["items"][0]

    resp = client.post(
        PREDICT,
        data={"data": payload(first["anno_id"])},
        files={"image": ("frame.png", b"frame-one")},
        headers=ctx["headers"],
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] is True and resp.json()["data"]

    job = ctx["inference"].jobs[-1]
    assert job["anno_id"] == first["anno_id"]
    assert base64.b64decode(job["image_b64"]) == b"frame-one"
    assert job["prompts"]["points"] == [{"x": 10, "y": 20}]
    assert job["return_type"] == 2


def test_each_predict_sends_its_own_frame(client, auth_headers, ol, services):
    """The v1 bug was predicting on whichever frame was loaded last."""
    ctx = bootstrap(client, auth_headers, ol, services)
    first, second = ctx["items"][0], ctx["items"][1]

    client.post(
        PREDICT,
        data={"data": payload(first["anno_id"])},
        files={"image": ("a.png", b"aaa")},
        headers=ctx["headers"],
    )
    client.post(
        PREDICT,
        data={"data": payload(second["anno_id"])},
        files={"image": ("b.png", b"bbb")},
        headers=ctx["headers"],
    )
    sent = [base64.b64decode(job["image_b64"]) for job in ctx["inference"].jobs]
    assert sent == [b"aaa", b"bbb"]
    assert [job["anno_id"] for job in ctx["inference"].jobs] == [first["anno_id"], second["anno_id"]]


def test_predict_can_pull_the_frame_from_openlist(client, auth_headers, ol, services):
    ctx = bootstrap(client, auth_headers, ol, services)
    resp = client.post(
        PREDICT,
        data={"data": payload(ctx["items"][0]["anno_id"], rel_path="images/dish01/D1.png")},
        headers=ctx["headers"],
    )
    assert resp.status_code == 200
    assert base64.b64decode(ctx["inference"].jobs[-1]["image_b64"]) == b"png"  # the fake stores b"png"


def test_predict_can_reuse_an_uploaded_digest(client, auth_headers, ol, services):
    ctx = bootstrap(client, auth_headers, ol, services)
    upload = client.put(
        "/api/v2/projects/projA/images/local/frame.png",
        files={"file": ("frame.png", b"cached-frame")},
        headers=ctx["headers"],
    )
    digest = upload.json()["sha256"]

    resp = client.post(
        PREDICT,
        data={"data": payload(ctx["items"][0]["anno_id"], image_sha256=digest)},
        headers=ctx["headers"],
    )
    assert resp.status_code == 200
    assert ctx["inference"].jobs[-1]["image_sha256"] == digest


def test_predict_errors(client, auth_headers, ol, services):
    ctx = bootstrap(client, auth_headers, ol, services)
    headers = ctx["headers"]

    assert client.post(PREDICT, data={"data": "not json"}, headers=headers).status_code == 422
    assert client.post(PREDICT, data={"data": payload("deadbeef")}, headers=headers).status_code == 404
    no_image = client.post(PREDICT, data={"data": payload(ctx["items"][0]["anno_id"])}, headers=headers)
    assert no_image.status_code == 422 and "no image" in no_image.json()["message"]
    assert client.post(PREDICT, data={"data": payload(ctx["items"][0]["anno_id"])}).status_code == 401


def test_worker_unavailable_is_503(client, auth_headers, ol, services):
    ctx = bootstrap(client, auth_headers, ol, services)
    ctx["inference"].fail_with = InferenceUnavailable("worker down")

    resp = client.post(
        PREDICT,
        data={"data": payload(ctx["items"][0]["anno_id"])},
        files={"image": ("frame.png", b"x")},
        headers=ctx["headers"],
    )
    assert resp.status_code == 503 and resp.json()["code"] == "inference_unavailable"
