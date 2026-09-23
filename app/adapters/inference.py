"""HTTP client for the inference worker (a separate process).

The API never loads a model: it sends a job (image + prompts) and returns the
worker's :class:`SamReturn` payload. That is what removes v1's "predict ran on
whichever task was loaded last" bug — every job carries its own image.

The job contract is documented in ``docs/architecture-v2.md`` §8.
"""

from __future__ import annotations

from typing import Any

import requests

from app.core.config import Settings
from app.core.errors import InferenceUnavailable, UpstreamError
from app.core.logging import get_logger

logger = get_logger("zlabel.app.inference")


class InferenceClient:
    def __init__(self, settings: Settings) -> None:
        self.url = (settings.inference_url or "").rstrip("/")
        self.token = settings.inference_token
        self.timeout = settings.inference_timeout

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    @property
    def configured(self) -> bool:
        return bool(self.url)

    def health(self) -> dict[str, Any]:
        if not self.configured:
            return {"status": "unconfigured"}
        try:
            resp = requests.get(f"{self.url}/health", timeout=2.0, headers=self._headers())
        except requests.RequestException as e:
            return {"status": "unavailable", "message": str(e)}
        if resp.status_code != 200:
            return {"status": "error", "http": resp.status_code}
        return {"status": "ok", "detail": resp.json()}

    def infer(self, job: dict[str, Any]) -> dict[str, Any]:
        """Run one job; raises ``inference_unavailable`` (503) when the worker is down."""
        if not self.configured:
            raise InferenceUnavailable("no inference worker is configured")
        try:
            resp = requests.post(f"{self.url}/infer", json=job, timeout=self.timeout, headers=self._headers())
        except requests.RequestException as e:
            logger.warning(f"inference worker unreachable: {e}")
            raise InferenceUnavailable(f"inference worker unreachable: {e}") from e
        if resp.status_code == 200:
            return resp.json()
        try:
            detail = resp.json()
        except Exception:  # noqa: BLE001
            detail = (resp.text or "")[:200]
        if resp.status_code >= 500:
            raise InferenceUnavailable(f"inference worker failed ({resp.status_code})", detail=detail)
        raise UpstreamError(f"inference request rejected ({resp.status_code})", detail=detail)
