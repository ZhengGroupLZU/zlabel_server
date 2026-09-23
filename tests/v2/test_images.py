"""Task reads (caching, per-user ACL) and uploads (content-addressed store)."""

from __future__ import annotations

from tests.v2.fakes import LocalBackendHarness
from tests.v2.test_projects import seed

ROOT = "/zlabel_server/projects"


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
