"""OpenList error mapping: a missing object must surface as a 404."""

from __future__ import annotations

import pytest
import requests

from app.openlist_api.client import BaseClient
from app.openlist_api.exceptions import NotFoundError, OpenListAPIError


class _Resp:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (str(payload) if payload is not None else "")
        self.reason = "Error"
        self.content = b""

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self):
        if not 200 <= self.status_code < 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)


def _client() -> BaseClient:
    return BaseClient.__new__(BaseClient)  # skip __init__ (no HTTP session needed)


@pytest.mark.parametrize(
    "payload",
    [
        {"message": "failed to getobj: object not found", "data": None},
        {"message": "failed get storage: storage not found; rawPath: /x.png", "data": None},
    ],
)
def test_missing_object_over_500_maps_to_404(payload):
    with pytest.raises(NotFoundError) as exc:
        _client()._handle_response(_Resp(500, payload))
    assert exc.value.status_code == 404


def test_other_server_error_stays_an_error():
    with pytest.raises(OpenListAPIError) as exc:
        _client()._handle_response(_Resp(500, {"message": "backend exploded", "data": None}))
    assert exc.value.status_code == 500
    assert not isinstance(exc.value, NotFoundError)


def test_plain_http_404_maps_to_404():
    with pytest.raises(NotFoundError):
        _client()._handle_response(_Resp(404, None, text="not found"))


def test_non_json_error_is_not_a_404():
    with pytest.raises(OpenListAPIError) as exc:
        _client()._handle_response(_Resp(502, None, text="bad gateway"))
    assert not isinstance(exc.value, NotFoundError)


def test_body_code_500_with_not_found_maps_to_404():
    """OpenList answers with HTTP 200 + ``{"code": 500, "message": "...not found"}``."""
    resp = _Resp(200, {"code": 500, "message": "failed to get obj: object not found", "data": None})

    with pytest.raises(NotFoundError) as exc:
        _client()._handle_response(resp)

    assert exc.value.status_code == 404


def test_body_code_500_without_not_found_stays_a_server_error():
    resp = _Resp(200, {"code": 500, "message": "disk exploded", "data": None})

    with pytest.raises(OpenListAPIError) as exc:
        _client()._handle_response(resp)

    assert exc.value.status_code == 500
    assert not isinstance(exc.value, NotFoundError)
