"""Worker engine: per-frame embeddings, crop handling, queue and degradation."""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from inference.config import InferenceSettings
from tests.v2.fakes import FakeEnginePredictor
from v2.core.errors import ValidationFailed, WorkerBusy
from v2.inference_worker.engine import EmbeddingCache, InferenceEngine
from v2.inference_worker.schemas import InferJob


def png_bytes(size=(64, 96), color=(10, 20, 30)) -> bytes:
    """PIL size is (width, height); numpy shapes below are (height, width)."""
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def job(sha: str = "a" * 64, *, image: bytes | None = None, **extra) -> InferJob:
    body = {
        "anno_id": "anno1",
        "image_sha256": sha,
        "image_b64": base64.b64encode(image or png_bytes()).decode("ascii"),
        "prompts": {"points": [{"x": 10, "y": 20}], "labels": [1]},
        **extra,
    }
    return InferJob(**body)


@pytest.fixture
def engine() -> InferenceEngine:
    settings = InferenceSettings(embedding_cache_size=2, queue_size=2, max_concurrency=1)
    return InferenceEngine(settings, predictor=FakeEnginePredictor())


# region embedding cache
def test_same_frame_is_encoded_once(engine):
    engine.run(job("a" * 64))
    engine.run(job("a" * 64))
    engine.run(job("a" * 64))

    assert engine.predictor.encoded == [(96, 64)]  # encoded once (h, w)
    assert engine.metrics.cache_misses == 1
    assert engine.metrics.cache_hits == 2
    assert engine.metrics.jobs == 3
    assert engine.metrics.hit_rate == pytest.approx(2 / 3, abs=1e-4)


def test_different_frames_are_encoded_separately(engine):
    engine.run(job("a" * 64))
    engine.run(job("b" * 64))
    assert len(engine.predictor.encoded) == 2
    assert engine.metrics.cache_misses == 2


def test_switching_back_restores_the_snapshot(engine):
    engine.run(job("a" * 64))
    engine.run(job("b" * 64))
    engine.run(job("a" * 64))  # cache size 2 keeps both

    assert len(engine.predictor.encoded) == 2  # still just two encodes
    assert engine.predictor.restored[-1]["shape"] == (96, 64)
    assert engine.metrics.cache_hits == 1


def test_lru_eviction_re_encodes(engine):
    engine.cache = EmbeddingCache(1)
    engine.run(job("a" * 64))
    engine.run(job("b" * 64))
    engine.run(job("a" * 64))
    assert len(engine.predictor.encoded) == 3
    assert engine.metrics.cache_hits == 0


def test_results_follow_the_requested_frame(engine):
    """Two frames in a row: each result belongs to its own job."""
    first = engine.run(job("a" * 64, image=png_bytes((64, 96))))
    second = engine.run(job("b" * 64, image=png_bytes((32, 200))))
    assert first["anno_id"] == second["anno_id"] == "anno1"
    assert engine.predictor.encoded == [(96, 64), (200, 32)]


# endregion


# region crop, pull, modes, queue
def test_crop_box_shifts_prompts_and_results(engine):
    crop = [10, 20, 50, 60]  # 40x40 inside the 64x96 frame
    engine.run(
        job("a" * 64, crop_box=crop, return_type=2, prompts={"points": [{"x": 30, "y": 40}], "labels": [1]})
    )
    assert engine.predictor.encoded == [(40, 40)]  # the encoder saw the crop

    engine.cache.clear()
    result = engine.run(
        job("b" * 64, crop_box=crop, return_type=1, prompts={"rects": [{"x": 1, "y": 2, "w": 3, "h": 4}]})
    )
    rect = result["data"][0]
    # the detector worked in crop space; the answer is translated back
    assert rect["x"] >= 10 and rect["y"] >= 20


def test_crop_is_part_of_the_cache_key(engine):
    engine.run(job("a" * 64, crop_box=[0, 0, 32, 32]))
    engine.run(job("a" * 64))  # same frame, no crop -> must encode again
    assert len(engine.predictor.encoded) == 2


def test_image_pull_path():
    calls: list[str] = []
    points = {"points": [{"x": 1, "y": 2}], "labels": [1]}

    def fetch(url, **kwargs):
        calls.append(url)
        assert kwargs["headers"]["Authorization"] == "Bearer secret"
        return type("Resp", (), {"status_code": 200, "content": png_bytes()})()

    settings = InferenceSettings(api_base_url="http://api.test", token="secret")
    engine = InferenceEngine(settings, predictor=FakeEnginePredictor(), fetch=fetch)
    pulled = InferJob(
        anno_id="anno1",
        image_sha256="c" * 64,
        image_url="/api/v2/internal/images/ccc",
        prompts=points,
    )
    assert engine.run(pulled)["status"] is True
    assert calls == ["http://api.test/api/v2/internal/images/ccc"]


def test_image_pull_failure_is_503(engine):
    def boom(*_args, **_kwargs):
        raise __import__("requests").exceptions.ConnectionError("no route")

    engine._fetch = boom
    with pytest.raises(Exception) as exc:  # InferenceUnavailable
        engine.run(
            InferJob(
                anno_id="a",
                image_sha256="d" * 64,
                image_url="/x",
                prompts={"points": [{"x": 1, "y": 2}], "labels": [1]},
            )
        )
    assert exc.value.code == "inference_unavailable"  # type: ignore[attr-defined]


def test_mode_validation(engine):
    rects = {"rects": [{"x": 1, "y": 2, "w": 8, "h": 8}]}
    with pytest.raises(ValidationFailed):
        engine.run(job("a" * 64, mode=3))  # points + SAM&CV is not implemented
    with pytest.raises(ValidationFailed):
        engine.run(job("b" * 64, mode=99, prompts=rects))
    with pytest.raises(ValidationFailed):
        engine.run(job("c" * 64, return_type=7, prompts=rects))
    with pytest.raises(ValidationFailed):
        engine.run(job("d" * 64, mode=2, prompts={"texts": ["seed"]}))  # texts need SAM
    with pytest.raises(ValidationFailed):
        engine.run(InferJob(anno_id="a", image_b64=base64.b64encode(png_bytes()).decode()))
    with pytest.raises(ValidationFailed):
        engine.run(job("e" * 64, image=None, image_b64="not-base64!!"))

    # mode 0 is the legacy "both toggles on" alias of SAM+CV, valid for rect prompts
    result = engine.run(job("f" * 64, mode=0, return_type=1, prompts=rects))
    assert result["status"] is True


def test_queue_full_is_503():
    settings = InferenceSettings(queue_size=0)
    engine = InferenceEngine(settings, predictor=FakeEnginePredictor())
    with pytest.raises(WorkerBusy):
        engine.run(job("a" * 64))


def test_health_and_metrics(engine):
    engine.run(job("a" * 64))
    engine.run(job("a" * 64))
    health = engine.health()
    assert health["model"] and health["loaded"] is True
    assert health["cache"]["entries"] == 1 and health["cache"]["hit_rate"] == 0.5
    assert health["queue"] == {"waiting": 0, "size": 2, "concurrency": 1}

    metrics = engine.metrics_snapshot()
    assert metrics["jobs"] == 2 and metrics["cache_hits"] == 1
    assert metrics["latency_ms"]["p50"] >= 0

    engine.metrics.errors += 1
    assert engine.metrics_snapshot()["errors"] == 1


# endregion


# region worker HTTP API
def worker_client(**settings_kwargs):
    from fastapi.testclient import TestClient

    from v2.inference_worker.main import create_worker_app

    settings = InferenceSettings(embedding_cache_size=2, queue_size=4, **settings_kwargs)
    engine = InferenceEngine(settings, predictor=FakeEnginePredictor())
    return TestClient(create_worker_app(settings, engine)), engine


def test_worker_requires_the_shared_token():
    client, _ = worker_client(token="secret")
    body = {
        "anno_id": "a",
        "image_b64": base64.b64encode(png_bytes()).decode(),
        "prompts": {"points": [{"x": 1, "y": 2}]},
    }
    assert client.post("/infer", json=body).status_code == 401
    assert client.post("/infer", json=body, headers={"Authorization": "Bearer nope"}).status_code == 401

    unconfigured, _ = worker_client(token="")
    assert unconfigured.post("/infer", json=body).status_code == 403


def test_worker_infer_roundtrip_and_health():
    client, engine = worker_client(token="secret")
    headers = {"Authorization": "Bearer secret"}
    body = {
        "anno_id": "anno1",
        "image_sha256": "a" * 64,
        "image_b64": base64.b64encode(png_bytes()).decode(),
        "prompts": {"points": [{"x": 5, "y": 6}], "labels": [1]},
        "mode": 1,
        "return_type": 2,
    }
    resp = client.post("/infer", json=body, headers=headers)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["status"] is True and payload["data"]
    assert payload["anno_id"] == "anno1"

    health = client.get("/health").json()  # liveness needs no token
    assert health["status"] == "ok" and health["loaded"] is True
    assert health["cache"]["entries"] == 1
    assert health["queue"]["size"] == 4

    assert client.get("/metrics").status_code == 401
    metrics = client.get("/metrics", headers=headers).json()
    assert metrics["jobs"] == 1 and metrics["cache_misses"] == 1

    # a second identical request is served from the cache
    client.post("/infer", json=body, headers=headers)
    assert client.get("/metrics", headers=headers).json()["cache_hits"] == 1
    assert engine.predictor.encoded == [(96, 64)]


def test_worker_rejects_bad_payloads():
    client, _ = worker_client(token="secret")
    headers = {"Authorization": "Bearer secret"}
    assert client.post("/infer", json={"prompts": {}}, headers=headers).status_code == 422
    assert client.post("/infer", json={}, headers=headers).status_code == 422


# endregion


def test_crop_box_moves_the_prompts_into_crop_space(engine):
    """The detector works in crop space, so prompts must be shifted with it."""
    captured: dict = {}
    original = engine.predictor.predict

    def record(points=None, labels=None, bboxes=None, text=None, **_ignored):
        captured["points"] = points
        return original(points=points, labels=labels, bboxes=bboxes, text=text)

    engine.predictor.predict = record
    engine.run(
        job("a" * 64, crop_box=[10, 20, 50, 60], prompts={"points": [{"x": 30, "y": 45}], "labels": [1]})
    )
    assert captured["points"] == [(20.0, 25.0)]  # (30-10, 45-20)


def test_state_filter_accepts_a_list():
    """The client asks for several states at once (draft,rejected)."""
    from v2.services.task_service import _split_states

    assert _split_states("draft,rejected") == ["draft", "rejected"]
    assert _split_states(" approved ") == ["approved"]
    assert _split_states(None) == []
