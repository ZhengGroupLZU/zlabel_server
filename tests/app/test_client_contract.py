"""Contract test: the desktop client's API client against the real v2 ASGI app.

The desktop checkout normally sits above this repo (``zlabel_server/`` lives inside
it in a dev tree). ``zlabel.utils.api_helper`` is loaded standalone — the real
package pulls in PySide6 — and its ``requests`` calls are routed into the test
client, so every URL, query parameter, header and body the desktop produces is
checked against the server for real.

Run it from the desktop repo (``uv run pytest zlabel_server/tests/app``) or from
this repo when the desktop tree is present next door.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from urllib.parse import parse_qsl, urlparse

import pytest
import requests

from tests.app.fakes import FakeInference
from tests.app.test_projects import seed

CLIENT_ROOT = Path(__file__).resolve().parents[3]
API_HELPER = CLIENT_ROOT / "zlabel" / "utils" / "api_helper.py"

pytestmark = pytest.mark.skipif(
    not API_HELPER.exists(), reason=f"desktop checkout not found at {CLIENT_ROOT}"
)

PROJ = "projA"


@pytest.fixture(scope="module")
def client_api():
    """Import the desktop's api_helper without importing the PySide6 package tree."""
    if "zlabel.utils.api_helper" in sys.modules:
        return sys.modules["zlabel.utils.api_helper"]

    def load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    for pkg_name, rel in (("zlabel", ""), ("zlabel.utils", "utils")):
        package = types.ModuleType(pkg_name)
        package.__path__ = [str(CLIENT_ROOT / "zlabel" / rel)]  # type: ignore[attr-defined]
        sys.modules[pkg_name] = package
    load("zlabel.utils.logger", CLIENT_ROOT / "zlabel" / "utils" / "logger.py")
    return load("zlabel.utils.api_helper", API_HELPER)


class _RequestsToTestClient:
    """``requests``-shaped shim that turns calls into TestClient requests."""

    def __init__(self, test_client) -> None:
        self.client = test_client
        # the client uses both ``requests.RequestException`` and ``requests.exceptions``
        self.RequestException = requests.RequestException
        self.exceptions = requests.exceptions
        self.calls: list[tuple[str, str]] = []
        self.timeouts: list[tuple[str, str, float | None]] = []

    def _call(self, method: str, url: str, **kwargs):
        parsed = urlparse(url)
        params = kwargs.pop("params", None)
        if parsed.query:
            merged = dict(parse_qsl(parsed.query))
            merged.update(params or {})
            params = merged
        timeout = kwargs.pop("timeout", None)
        data = kwargs.pop("data", None)
        content = kwargs.pop("content", None)
        if isinstance(data, (bytes, bytearray)):
            content, data = bytes(data), None
        request_kwargs = {"params": params, "headers": kwargs.pop("headers", None), **kwargs}
        if content is not None:
            request_kwargs["content"] = content
        if data is not None:
            request_kwargs["data"] = data
        # TestClient deprecates its own timeout argument; the desktop's timeout
        # values are recorded instead.
        self.timeouts.append((method, parsed.path, timeout))
        self.calls.append((method, parsed.path))
        return self.client.request(method, parsed.path, **request_kwargs)

    def request(self, method: str, url: str, **kwargs):
        return self._call(method.upper(), url, **kwargs)

    def get(self, url: str, **kwargs):
        return self._call("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self._call("POST", url, **kwargs)

    def put(self, url: str, **kwargs):
        return self._call("PUT", url, **kwargs)


@pytest.fixture
def api(client, client_api, monkeypatch):
    """A logged-in desktop client pointed at the test app."""
    shim = _RequestsToTestClient(client)
    monkeypatch.setattr(client_api, "requests", shim)
    api = client_api.ZLServerApiClient("rainy", "secret", "http://testserver")
    return api


def _png(size=(64, 64), color=(1, 2, 3)) -> bytes:
    """A real PNG: get_image hands the bytes to PIL."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _login(api, username: str = "rainy", password: str = "secret"):
    assert api.login(username, password), api.last_login_error
    return api


def test_login_carries_user_role_and_capabilities(api):
    _login(api)
    assert api.user_token
    assert api.user["name"] == "rainy" and api.role == "admin"
    assert api.server_version.startswith("2.")
    assert api.supports("tasks.claim") and api.supports("annotations.versions")
    assert api.headers["Authorization"] == f"Bearer {api.user_token}"


def test_scan_projects_tasks_and_versions(api, harness):
    _login(api)
    seed(harness, PROJ, files=("images/dish01/D1.png", "images/dish01/D2.png"))
    assert api.scan(PROJ) is True  # admin may scan

    projects = api.get_projects()
    assert [p["name"] for p in projects] == [PROJ]
    assert isinstance(projects[0]["id"], int) and projects[0]["progress"]["total"] == 2

    page = api.get_tasks(PROJ, states="draft,rejected", order="sequence")
    assert page["total"] == 2
    first = page["items"][0]
    assert first["rel_path"] == "images/dish01/D1.png"
    assert first["group"] == "images/dish01" and first["day"] == 1
    assert api.version_of(first["anno_id"]) in (0, None)  # seeded from the listing


def test_fetch_all_asks_the_server_for_every_task(api, harness):
    """The File dock's "all" entry sends ``limit=0`` (see ``FETCH_ALL``), which must
    not be a validation error: a whole sequence group has to arrive in one page."""
    _login(api)
    files = tuple(f"images/dish01/D{i}.png" for i in range(1, 9))
    seed(harness, PROJ, files=files)
    assert api.scan(PROJ) is True

    page = api.get_tasks(PROJ, limit=0, order="sequence")
    assert page is not None, "the server refused limit=0"
    assert page["total"] == len(files) and len(page["items"]) == len(files)
    # the explicit combo values must keep working too (1000 used to be a 422)
    assert len(api.get_tasks(PROJ, limit=1000)["items"]) == len(files)


def test_annotation_roundtrip_and_optimistic_locking(api, harness, client_api, tmp_path):
    _login(api)
    seed(harness, PROJ, files=("images/dish01/D1.png",))
    api.scan(PROJ)
    item = api.get_tasks(PROJ)["items"][0]
    anno_id = item["anno_id"]

    assert api.get_zlabel(f"{anno_id}.zlabel", PROJ).not_found  # never annotated

    document = tmp_path / f"{anno_id}.zlabel"
    document.write_text('{"results": {"r1": {"labels": [{"name": "Root"}]}}}', encoding="utf-8")
    assert api.save_zlabel(str(document), project=PROJ).ok
    assert api.version_of(anno_id) == 1

    fetched = api.get_zlabel(f"{anno_id}.zlabel", PROJ)
    assert fetched.ok and "Root" in fetched.text
    assert api.version_of(anno_id) == 1

    # a second save on top of v1 succeeds and bumps the version
    assert api.save_zlabel(str(document), project=PROJ).ok
    assert api.version_of(anno_id) == 2

    # a stale base version is refused with the server's details
    api._versions[anno_id] = 1
    conflict = api.save_zlabel(str(document), project=PROJ)
    assert conflict.status == 409 and "v2" in conflict.message
    assert api.version_of(anno_id) is None  # must reload before retrying

    # a reviewer may force it, an annotator may not (the client's conflict dialog
    # only offers "overwrite" to reviewers, but the server has to enforce it too)
    harness.users["bob"] = "pw-padding"
    bob = client_api.ZLServerApiClient("bob", "pw-padding", api.sam_api)
    assert bob.login("bob", "pw-padding"), bob.last_login_error
    assert bob.role == "annotator"
    assert bob.save_zlabel(str(document), force=True, project=PROJ).status == 403
    assert api.save_zlabel(str(document), force=True, project=PROJ).ok
    assert api.version_of(anno_id) == 3  # forced save still creates a normal version


def test_annotator_gets_rbac_instead_of_errors(api, harness, client_api):
    """The client must degrade gracefully when the role is not allowed to scan."""
    _login(api)  # the first account of a fresh database becomes admin
    harness.users["bob"] = "pw-padding"
    bob = client_api.ZLServerApiClient("bob", "pw-padding", api.sam_api)
    assert bob.login("bob", "pw-padding"), bob.last_login_error
    assert bob.role == "annotator" and bob.is_reviewer is False
    assert bob.scan(PROJ) is False  # 403 -> no scan, not an exception


def test_labels_and_progress(api, harness):
    _login(api)
    seed(harness, PROJ, files=("a.png",))
    api.scan(PROJ)
    assert api.get_labels(PROJ) == []
    progress = api.get_progress(PROJ)
    assert progress["total"] == 1 and progress["draft"] == 1 and progress["finished"] == 0
    # an empty project name never triggers a request
    assert api.get_labels("") is None and api.get_progress("") is None


def test_labels_created_from_the_desktop_get_a_palette_colour_and_append(api, harness):
    """``create_label`` must not pin the colour or the position.

    The desktop used to send ``color="#000000", sort=0`` for every new label, which
    disabled the server palette and gave every label the same ordinal (the value the
    COCO/YOLO exports use as the class id).
    """
    from app.services.label_palette import LABEL_PALETTE

    _login(api)
    seed(harness, PROJ, files=("a.png",))
    api.scan(PROJ)

    for name in ("Seed", "Root", "Shoot"):
        assert api.create_label(PROJ, name) is not None

    labels = api.get_labels(PROJ)
    assert [label["name"] for label in labels] == ["Seed", "Root", "Shoot"]
    assert [label["sort"] for label in labels] == [0, 1, 2]  # appended, not all 0
    colors = [label["color"] for label in labels]
    assert all(color in LABEL_PALETTE for color in colors), colors
    assert len(set(colors)) == 3  # and each one is a different, unused colour

    # an explicit colour/position is still honoured (the admin UI and scripts)
    assert api.create_label(PROJ, "Dish", "#123456", 7) is not None
    dish = next(label for label in api.get_labels(PROJ) if label["name"] == "Dish")
    assert (dish["color"], dish["sort"]) == ("#123456", 7)


def test_get_image_returns_a_pil_image(api, harness):
    _login(api)
    seed(harness, PROJ, files=())
    harness.add_file(f"/zlabel_server/projects/{PROJ}/images/dish01/D1.png", _png())
    api.scan(PROJ)

    image = api.get_image("images/dish01/D1.png", project=PROJ)
    assert image is not None and image.size == (64, 64)
    assert api.get_image("missing.png", project=PROJ) is None


def test_predict_contract(api, harness, services):
    """What the desktop sends must be what the worker receives."""
    _login(api)
    seed(harness, PROJ, files=("images/dish01/D1.png",))
    api.scan(PROJ)
    anno_id = api.get_tasks(PROJ)["items"][0]["anno_id"]

    fake = FakeInference()
    services.inference = fake
    from PIL import Image

    task = Image.new("RGB", (32, 32), (1, 2, 3))
    result = api.predict(
        anno_id,
        image=task,
        project=PROJ,
        points=[{"x": 5.0, "y": 6.0}],
        labels=[1.0],
        threshold=100,
        mode=1,
        return_type=2,
    )
    assert result["status"] is True
    job = fake.jobs[-1]
    assert job["prompts"]["points"] == [{"x": 5.0, "y": 6.0}]
    assert job["mode"] == 1 and job["return_type"] == 2
    assert job["image_sha256"] and job["image_b64"]

    # server-side tasks are referenced, not re-uploaded
    api.predict(anno_id, project=PROJ, rel_path="images/dish01/D1.png", points=[{"x": 1.0, "y": 2.0}])
    assert fake.jobs[-1]["image_b64"]


def test_predict_by_uploaded_digest_costs_no_image_bytes(api, harness, services, client):
    """C7: upload once, then predict by ``image_sha256``.

    ``_resolve_image``'s digest branch had no coverage at all, and it is what the
    desktop uses for every predict after the first one of an image: the image must not
    travel from the client again, while the job keeps the *same* ``image_sha256`` /
    job id so the worker's embedding cache still hits.
    """
    import json

    from PIL import Image

    _login(api)
    seed(harness, PROJ, files=("images/dish01/D1.png",))
    api.scan(PROJ)
    anno_id = api.get_tasks(PROJ)["items"][0]["anno_id"]

    fake = FakeInference()
    services.inference = fake
    task = Image.new("RGB", (32, 32), (9, 8, 7))

    # first predict: the image travels with the request
    api.predict(anno_id, image=task, project=PROJ, points=[{"x": 1.0, "y": 2.0}], labels=[1.0])
    with_bytes = fake.jobs[-1]
    assert with_bytes["image_b64"]

    digest = api.upload_image("images/dish01/D1.png", task, project=PROJ)
    assert digest  # the upload ack carries the reference

    # a later predict references it: no file part at all, so the bytes cannot travel
    payload = {
        "anno_id": anno_id,
        "mode": 1,
        "return_type": 1,
        "image_sha256": digest,
        "points": [{"x": 1.0, "y": 2.0}],
        "labels": [1.0],
    }
    raw = client.post(
        f"/api/v2/projects/{PROJ}/predict", data={"data": json.dumps(payload)}, headers=api.headers
    )
    assert raw.status_code == 200, raw.text
    by_digest = fake.jobs[-1]
    # the worker sees the same picture and the same cache key, whatever the transport
    assert by_digest["image_sha256"] == digest == with_bytes["image_sha256"]
    assert by_digest["job_id"] == with_bytes["job_id"]

    # an unknown digest is a 404, which is how the desktop knows to upload again
    unknown = api.predict(anno_id, project=PROJ, image_sha256="0" * 64)
    assert unknown["status"] is False and unknown["http_status"] == 404

    # with the non-inline transport the API -> worker hop stops carrying bytes too
    services.settings.inference_inline_images = False
    assert (
        api.predict(anno_id, project=PROJ, image_sha256=digest, points=[{"x": 1.0, "y": 2.0}])["status"]
        is True
    )
    assert fake.jobs[-1]["image_b64"] is None
    assert fake.jobs[-1]["image_url"].endswith(f"/internal/images/{digest}")
    services.settings.inference_inline_images = True

    # a different image gets a different digest, so a stale entry cannot leak one
    # task's pixels into another task's prediction
    other = api.upload_image("images/dish01/D1.png", Image.new("RGB", (32, 32), (1, 1, 1)), project=PROJ)
    assert other and other != digest
    api.predict(anno_id, project=PROJ, image_sha256=other, points=[{"x": 1.0, "y": 2.0}])
    assert fake.jobs[-1]["image_sha256"] == other
    assert fake.jobs[-1]["job_id"] != with_bytes["job_id"]  # other image, other embedding


def test_logout_revokes_the_session(api):
    _login(api)
    api.logout()
    assert api.user_token == ""
    assert api.get_projects() is None  # 401 -> treated as "server refused"


def test_revoked_session_is_renewed_automatically(api, harness, client, client_api, tmp_path):
    """An admin revoking a session must not break the running desktop.

    The server answers 401 ``session_stale`` for a dead-but-known session, and the
    desktop replays the call once with a fresh token (``_RetryingRequests``) - no
    "Get Tasks Failed", no read-only task, no manual re-login.
    """
    _login(api)  # rainy: the first account, so the admin
    harness.users["bob"] = "pw-padding"
    bob = client_api.ZLServerApiClient("bob", "pw-padding", api.sam_api)
    assert bob.login("bob", "pw-padding"), bob.last_login_error

    seed(harness, PROJ, files=("images/dish01/D1.png",))
    api.scan(PROJ)
    assert bob.get_tasks(PROJ)["total"] == 1
    bob_id = next(
        u["id"] for u in client.get("/api/v2/auth/users", headers=api.headers).json() if u["name"] == "bob"
    )
    dead_token = bob.user_token

    # the admin changes bob's role, which revokes every session he holds
    changed = client.patch(f"/api/v2/admin/users/{bob_id}", json={"role": "reviewer"}, headers=api.headers)
    assert changed.status_code == 200
    stale = client.get("/api/v2/auth/me", headers={"Authorization": f"Bearer {dead_token}"})
    assert stale.status_code == 401 and stale.json()["code"] == "session_stale"

    # bob's next call heals itself: one renewal, then the replay succeeds
    page = bob.get_tasks(PROJ)
    assert page is not None and page["total"] == 1
    assert bob.user_token != dead_token
    assert bob.role == "reviewer"  # the fresh login carried the new role
    assert bob.supports("tasks.claim")  # and the health probe ran again

    # a write path is replayed too, and the optimistic-locking cache survives it
    anno_id = page["items"][0]["anno_id"]
    assert bob.claim(anno_id).ok
    document = tmp_path / f"{anno_id}.zlabel"
    document.write_text('{"results": {}}', encoding="utf-8")
    assert bob.save_zlabel(str(document), project=PROJ).ok
    assert bob.version_of(anno_id) == 1


def test_task_calls_do_not_use_the_control_timeout(api, harness, client, client_api, monkeypatch):
    """Images/predict stay unbounded; control calls keep the 10s timeout."""
    routed = _RequestsToTestClient(client)
    monkeypatch.setattr(client_api, "requests", routed)
    _login(api)
    seed(harness, PROJ, files=())
    harness.add_file(f"/zlabel_server/projects/{PROJ}/a.png", _png())
    api.scan(PROJ)

    api.get_projects()
    assert routed.timeouts[-1][2] == 10.0
    api.get_image("a.png", project=PROJ)
    assert routed.timeouts[-1][2] is None


def _ready_task(api, harness, tmp_path) -> tuple[str, str]:
    """One project + one task with a saved annotation; returns (anno_id, document path)."""
    seed(harness, PROJ, files=("images/dish01/D1.png",))
    api.scan(PROJ)
    anno_id = api.get_tasks(PROJ)["items"][0]["anno_id"]
    document = tmp_path / f"{anno_id}.zlabel"
    document.write_text('{"results": {"r1": {"labels": [{"name": "Root"}]}}}', encoding="utf-8")
    assert api.save_zlabel(str(document), project=PROJ).ok
    return anno_id, str(document)


def test_claim_lease_and_review_roundtrip(api, harness, client_api, tmp_path):
    """The whole workflow the desktop drives: claim -> save -> submit -> review."""
    _login(api)
    harness.users["bob"] = "pw-padding"
    reviewer = client_api.ZLServerApiClient("bob", "pw-padding", api.sam_api)
    assert reviewer.login("bob", "pw-padding"), reviewer.last_login_error
    assert reviewer.role == "annotator"  # only the first account is admin

    anno_id, document = _ready_task(api, harness, tmp_path)

    # the task carries the annotator's lease; a second client is refused with detail
    claim = api.claim(anno_id)
    assert claim.ok and claim.task["claimed_by"] == "rainy"
    assert claim.task["lease_expires_at"]

    taken = reviewer.claim(anno_id)
    assert taken.status == 409 and taken.claimed_by == "rainy" and taken.lease_expires_at
    assert reviewer.claim(anno_id, force=True).status == 403  # annotators cannot force

    # renewing, then handing in
    assert api.heartbeat(anno_id).ok
    assert api.submit(anno_id).ok
    assert api.heartbeat(anno_id).status == 409  # the lease was released on submit

    # a non-reviewer cannot decide
    assert reviewer.review(anno_id, "approve").status == 403
    rejected = api.review(anno_id, "reject", "wrong dish")
    assert (
        rejected.ok and rejected.task["state"] == "rejected" and rejected.task["review_note"] == "wrong dish"
    )

    # rework, resubmit, approve, reopen
    assert api.save_zlabel(document, project=PROJ).ok
    assert api.submit(anno_id).ok
    approved = api.review(anno_id, "approve")
    assert approved.ok and approved.task["state"] == "approved"
    reopened = api.reopen(anno_id, "another look")
    assert reopened.ok and reopened.task["state"] == "draft"

    # progress + my stats reflect it
    assert api.get_progress(PROJ)["draft"] == 1
    assert api.my_stats(PROJ)["draft"] == 1
    task = api.get_task(anno_id)
    assert task["state"] == "draft" and task["version"] >= 2


def test_version_history_contract(api, harness, tmp_path):
    """Every save is listed, and an old document can be fetched back."""
    _login(api)
    anno_id, document = _ready_task(api, harness, tmp_path)
    assert api.save_zlabel(document, project=PROJ).ok  # v2
    assert api.save_zlabel(document, project=PROJ).ok  # v3

    versions = api.annotation_versions(anno_id, PROJ)
    assert [v["version"] for v in versions] == [3, 2, 1]
    assert all(v["author"] == "rainy" for v in versions)

    first = api.annotation_version(anno_id, 1, PROJ)
    assert first is not None and "Root" in first
    current = api.annotation_version(anno_id, 3, PROJ)
    assert current is not None and "Root" in current
    assert api.annotation_version(anno_id, 99, PROJ) is None


def test_lease_expiry_and_takeover(client, api, harness, client_api, tmp_path):
    """Two desktop clients, one task: the loser is told, not silently overwritten.

    This is the flow the GUI drives: open a task -> lease -> work; a lease that
    lapses can be taken over, and the original holder's save is refused with the
    holder details so the client can re-claim or go read-only.
    """
    from datetime import timedelta

    from sqlalchemy import select

    from app.db.models import Task, utcnow

    _login(api)
    harness.users["bob"] = "pw-padding"
    second = client_api.ZLServerApiClient("bob", "pw-padding", api.sam_api)
    assert second.login("bob", "pw-padding"), second.last_login_error

    anno_id, document = _ready_task(api, harness, tmp_path)
    assert api.claim(anno_id).ok

    # the first client's lease lapses (the server TTL is ZLSERVER_LEASE_MINUTES)
    with client.app.state.db.session_scope() as session:
        task = session.scalar(select(Task).where(Task.anno_id == anno_id))
        task.claimed_by = None
        task.lease_expires_at = utcnow() - timedelta(minutes=1)
    assert api.heartbeat(anno_id).status == 409  # no live lease to renew any more

    # the second annotator picks the task up and saves
    taken = second.claim(anno_id)
    assert taken.ok and taken.task["claimed_by"] == "bob"
    assert second.save_zlabel(document, project=PROJ).ok

    # the first client is refused, with the new holder in the detail
    stale = api.save_zlabel(document, project=PROJ)
    assert stale.status == 409 and "bob" in stale.message  # names the new holder
    detail = api.claim(anno_id)
    assert detail.status == 409 and detail.claimed_by == "bob"

    # and a reviewer can always take over
    forced = api.claim(anno_id, force=True)
    assert forced.ok and forced.task["claimed_by"] == "rainy"
