"""The adapter is the only OpenList boundary: it must translate errors correctly."""

from __future__ import annotations

import pytest
import requests

from tests.v2.fakes import FakeOpenList
from v2.adapters.openlist import OpenListAdapter, is_image
from v2.core.config import Settings
from v2.core.errors import (
    Forbidden,
    NotFound,
    SessionStale,
    Unauthorized,
    UpstreamError,
)
from v2.vendor.openlist_api import OpenListAPIError

PROJ = "/zlabel_server/projects/projA"


@pytest.fixture
def ol() -> FakeOpenList:
    fake = FakeOpenList()
    fake.add_file(f"{PROJ}/images/dish01/D1.png", b"png-bytes")
    fake.add_file(f"{PROJ}/images/dish01/notes.txt", b"txt")
    fake.add_file(f"{PROJ}/zlabel/abc.zlabel", b"{}")
    return fake


@pytest.fixture
def adapter(ol: FakeOpenList) -> OpenListAdapter:
    return OpenListAdapter(Settings(oplist_proj_dir="/zlabel_server/projects"), client_factory=ol.client)


def test_login_and_read(adapter: OpenListAdapter, ol: FakeOpenList):
    token = adapter.login("rainy", "secret")
    assert token in ol.tokens
    assert adapter.get_bytes(f"{PROJ}/images/dish01/D1.png", token) == b"png-bytes"


def test_login_rejects_bad_credentials(adapter: OpenListAdapter):
    with pytest.raises(Unauthorized):
        adapter.login("rainy", "wrong")


def test_current_user(adapter: OpenListAdapter):
    token = adapter.login("rainy", "secret")
    assert adapter.current_user(token) == {"id": "id-rainy", "name": "rainy", "email": "rainy@example.com"}


def test_invalid_token_is_session_stale(adapter: OpenListAdapter):
    with pytest.raises(SessionStale):
        adapter.get_bytes(f"{PROJ}/images/dish01/D1.png", "bogus")


def test_missing_object_is_not_found(adapter: OpenListAdapter):
    token = adapter.login("rainy", "secret")
    with pytest.raises(NotFound) as exc:
        adapter.get_bytes(f"{PROJ}/images/dish01/nope.png", token)
    assert exc.value.code == "not_found"
    assert adapter.exists(f"{PROJ}/images/dish01/nope.png", token) is False
    assert adapter.exists(f"{PROJ}/images/dish01/D1.png", token) is True


def test_glob_images_filters_other_extensions(adapter: OpenListAdapter):
    token = adapter.login("rainy", "secret")
    assert adapter.glob_images(PROJ, token) == [f"{PROJ}/images/dish01/D1.png"]


def test_upload_roundtrip_and_etag(adapter: OpenListAdapter):
    token = adapter.login("rainy", "secret")
    target = f"{PROJ}/zlabel/new.zlabel"
    payload = b'{"v":1}'
    adapter.put_bytes(target, payload, token)
    assert adapter.get_bytes(target, token) == payload
    info = adapter.file_info(target, token)
    assert info.size == len(payload)
    assert info.etag == f'"{len(payload)}-2026-01-01T00:00:00Z"'


def test_ensure_dir_is_idempotent(adapter: OpenListAdapter, ol: FakeOpenList):
    token = adapter.login("rainy", "secret")
    adapter.ensure_dir(f"{PROJ}/zlabel/_history", token)
    adapter.ensure_dir(f"{PROJ}/zlabel/_history", token)  # must not raise
    assert f"{PROJ}/zlabel/_history" in ol.dirs


def test_service_token_prefers_the_static_token(ol: FakeOpenList):
    adapter = OpenListAdapter(
        Settings(oplist_proj_dir="/zlabel_server/projects", oplist_token="static-token"),
        client_factory=ol.client,
    )
    assert adapter.service_token() == "static-token"
    assert ("login", "rainy") not in ol.calls


def test_service_token_falls_back_to_credentials(ol: FakeOpenList):
    adapter = OpenListAdapter(
        Settings(
            oplist_proj_dir="/zlabel_server/projects", oplist_username="rainy", oplist_password="secret"
        ),
        client_factory=ol.client,
    )
    token = adapter.service_token()
    assert ol.user_of(token) == "rainy"
    assert ("login", "rainy") in ol.calls


def test_service_token_requires_configuration(ol: FakeOpenList):
    adapter = OpenListAdapter(Settings(oplist_proj_dir="/zlabel_server/projects"), client_factory=ol.client)
    with pytest.raises(UpstreamError):
        adapter.service_token()


def test_transport_errors_become_upstream(adapter: OpenListAdapter, ol: FakeOpenList):
    token = adapter.login("rainy", "secret")
    ol.fail_on(f"{PROJ}/zlabel/abc.zlabel", requests.exceptions.ConnectionError("boom"))
    with pytest.raises(UpstreamError):
        adapter.get_bytes(f"{PROJ}/zlabel/abc.zlabel", token)


def test_forbidden_and_api_errors_are_mapped(adapter: OpenListAdapter, ol: FakeOpenList):
    token = adapter.login("rainy", "secret")
    ol.fail_on(f"{PROJ}/zlabel/abc.zlabel", OpenListAPIError("no permission", status_code=403))
    with pytest.raises(Forbidden):
        adapter.get_bytes(f"{PROJ}/zlabel/abc.zlabel", token)
    ol.fail_on(f"{PROJ}/zlabel/abc.zlabel", OpenListAPIError("storage offline", status_code=500))
    with pytest.raises(UpstreamError):
        adapter.get_bytes(f"{PROJ}/zlabel/abc.zlabel", token)


def test_is_image():
    assert is_image("a.PNG") and is_image("a.jpeg") and not is_image("a.txt")


def test_forbidden_messages_name_the_operation(adapter: OpenListAdapter, ol: FakeOpenList):
    """A 403 must say *what* OpenList refused, not just "permission denied"."""
    token = adapter.login("rainy", "secret")
    ol.fail_on(f"{PROJ}/zlabel/x.zlabel", OpenListAPIError("permission denied", status_code=403))
    with pytest.raises(Forbidden) as exc:
        adapter.put_bytes(f"{PROJ}/zlabel/x.zlabel", b"{}", token)
    assert "permission denied" in exc.value.message
    assert "upload(" in exc.value.message  # names the operation
