"""Task reads (caching, per-user ACL) and uploads (content-addressed store)."""

from __future__ import annotations

import io

from PIL import Image

from app.services.image_store import sha256_bytes
from tests.app.fakes import LocalBackendHarness
from tests.app.test_projects import seed

ROOT = "/zlabel_server/projects"
PROJ = "projA"


def _png(size: tuple[int, int] = (1024, 512), color=(10, 200, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _served_size(content: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(content)) as image:
        return image.size


def _seed_image(harness: LocalBackendHarness, content: bytes, rel: str = "images/dish01/D1.png") -> str:
    """Put a *real* image in the storage tree (``seed`` writes placeholders)."""
    harness.add_file(f"{harness.VIRTUAL_ROOT}/{PROJ}/{rel}", content)
    return rel


def _image_url(rel: str) -> str:
    return f"/api/v2/projects/{PROJ}/images/{rel}"


def test_get_image_uses_etag_cache(client, auth_headers, harness: LocalBackendHarness):
    seed(harness, "projA", files=("images/dish01/D1.png",))
    headers = auth_headers(client)
    client.post("/api/v2/projects/scan", headers=headers)

    resp = client.get("/api/v2/projects/projA/images/images/dish01/D1.png", headers=headers)
    assert resp.status_code == 200
    etag = resp.headers["ETag"]
    assert resp.headers["X-Image-Sha256"] and resp.headers["Content-Type"] == "image/png"

    cached = client.get(
        "/api/v2/projects/projA/images/images/dish01/D1.png",
        headers={**headers, "If-None-Match": etag},
    )
    assert cached.status_code == 304 and cached.content == b""


def test_get_image_missing_is_404(client, auth_headers, harness):
    seed(harness, "projA", files=("a.png",))
    headers = auth_headers(client)
    client.post("/api/v2/projects/scan", headers=headers)
    missing = client.get("/api/v2/projects/projA/images/nope.png", headers=headers)
    assert missing.status_code == 404 and missing.json()["code"] == "not_found"


def test_images_require_a_session(client, harness):
    seed(harness, "projA", files=("a.png",))
    assert client.get("/api/v2/projects/projA/images/a.png").status_code == 401


def test_upload_is_content_addressed(client, auth_headers, harness, services):
    seed(harness, "projA", files=("a.png",))
    headers = auth_headers(client)
    client.post("/api/v2/projects/scan", headers=headers)

    first = client.put(
        "/api/v2/projects/projA/images/local/task.png",
        files={"file": ("task.png", b"task-bytes")},
        headers=headers,
    )
    assert first.status_code == 200, first.text
    digest = first.json()["sha256"]
    assert services.images.has(digest)

    again = client.put(
        "/api/v2/projects/projA/images/local/task.png",
        files={"file": ("task.png", b"task-bytes")},
        headers=headers,
    )
    assert again.json()["sha256"] == digest  # same bytes -> same digest

    served = client.get(f"/api/v2/images/{digest}", headers=headers)
    assert served.status_code == 200 and served.content == b"task-bytes"
    assert client.get("/api/v2/images/deadbeef", headers=headers).status_code == 404


def test_upload_size_limit(client, auth_headers, harness, services):
    seed(harness, "projA", files=("a.png",))
    headers = auth_headers(client)
    client.post("/api/v2/projects/scan", headers=headers)
    services.settings.max_upload_bytes = 4

    resp = client.put(
        "/api/v2/projects/projA/images/big.png", files={"file": ("big.png", b"0123456789")}, headers=headers
    )
    assert resp.status_code == 413 and resp.json()["code"] == "payload_too_large"


# region max_side (downscaled task images)
def test_get_image_max_side_serves_a_downscaled_copy(client, auth_headers, harness: LocalBackendHarness):
    original = _png((1024, 512))
    rel = _seed_image(harness, original)
    headers = auth_headers(client)
    client.post("/api/v2/projects/scan", headers=headers)

    resp = client.get(_image_url(rel), params={"max_side": 256}, headers=headers)

    assert resp.status_code == 200
    assert resp.headers["Content-Type"] == "image/png"
    assert resp.headers["X-Image-Original-Size"] == "1024x512"
    assert _served_size(resp.content) == (256, 128)
    # the digest still identifies the *stored* file, not the variant
    assert resp.headers["X-Image-Sha256"] == sha256_bytes(original)


def test_get_image_max_side_zero_and_absent_serve_the_stored_bytes(
    client, auth_headers, harness: LocalBackendHarness
):
    original = _png((1024, 512))
    rel = _seed_image(harness, original)
    headers = auth_headers(client)
    client.post("/api/v2/projects/scan", headers=headers)

    plain = client.get(_image_url(rel), headers=headers)
    explicit = client.get(_image_url(rel), params={"max_side": 0}, headers=headers)

    assert plain.content == original == explicit.content
    assert "X-Image-Original-Size" not in plain.headers
    assert plain.headers["ETag"] == explicit.headers["ETag"]  # one cache entry for the original


def test_get_image_max_side_never_upscales(client, auth_headers, harness: LocalBackendHarness):
    original = _png((256, 128))
    rel = _seed_image(harness, original)
    headers = auth_headers(client)
    client.post("/api/v2/projects/scan", headers=headers)

    resp = client.get(_image_url(rel), params={"max_side": 4096}, headers=headers)

    assert resp.status_code == 200 and resp.content == original
    assert "X-Image-Original-Size" not in resp.headers
    # the ETag still names the requested variant: the client caches per resolution
    assert resp.headers["ETag"].endswith('-m4096"')


def test_get_image_variant_etag_is_per_resolution(client, auth_headers, harness: LocalBackendHarness):
    rel = _seed_image(harness, _png((1024, 512)))
    headers = auth_headers(client)
    client.post("/api/v2/projects/scan", headers=headers)

    full = client.get(_image_url(rel), headers=headers)
    small = client.get(_image_url(rel), params={"max_side": 256}, headers=headers)
    assert small.headers["ETag"] != full.headers["ETag"]

    # a conditional request is answered per variant, without decoding anything
    cached = client.get(
        _image_url(rel),
        params={"max_side": 256},
        headers={**headers, "If-None-Match": small.headers["ETag"]},
    )
    assert cached.status_code == 304 and cached.content == b""
    # the stored file's own ETag must not validate the variant
    stale = client.get(
        _image_url(rel),
        params={"max_side": 256},
        headers={**headers, "If-None-Match": full.headers["ETag"]},
    )
    assert stale.status_code == 200 and stale.content == small.content


def test_get_image_variant_is_cached_on_disk(client, auth_headers, harness: LocalBackendHarness, monkeypatch):
    from app.services import image_scaler

    rel = _seed_image(harness, _png((1024, 512)))
    headers = auth_headers(client)
    client.post("/api/v2/projects/scan", headers=headers)
    calls: list[int] = []
    real = image_scaler.render_variant

    def _counting(content: bytes, max_side: int, **kwargs):
        calls.append(max_side)
        return real(content, max_side, **kwargs)

    monkeypatch.setattr(image_scaler, "render_variant", _counting)
    first = client.get(_image_url(rel), params={"max_side": 256}, headers=headers)
    second = client.get(_image_url(rel), params={"max_side": 256}, headers=headers)
    assert calls == [256]
    assert first.content == second.content


def test_get_image_max_side_is_capped_by_the_server(
    client, auth_headers, harness: LocalBackendHarness, services
):
    rel = _seed_image(harness, _png((1024, 512)))
    headers = auth_headers(client)
    client.post("/api/v2/projects/scan", headers=headers)
    services.settings.image_max_side_limit = 128

    resp = client.get(_image_url(rel), params={"max_side": 8192}, headers=headers)

    assert _served_size(resp.content) == (128, 64)
    assert resp.headers["X-Image-Original-Size"] == "1024x512"
    assert resp.headers["ETag"].endswith('-m128"')  # the effective size, not the asked one


def test_get_image_max_side_falls_back_on_a_broken_file(client, auth_headers, harness: LocalBackendHarness):
    rel = _seed_image(harness, b"png")
    headers = auth_headers(client)
    client.post("/api/v2/projects/scan", headers=headers)

    resp = client.get(_image_url(rel), params={"max_side": 256}, headers=headers)

    assert resp.status_code == 200 and resp.content == b"png"
    assert "X-Image-Original-Size" not in resp.headers


def test_get_image_max_side_requires_a_session(client, auth_headers, harness: LocalBackendHarness):
    rel = _seed_image(harness, _png((1024, 512)))
    headers = auth_headers(client)
    client.post("/api/v2/projects/scan", headers=headers)

    assert client.get(_image_url(rel), params={"max_side": 256}).status_code == 401


# endregion
