"""核心API客户端实现。

提供了与OpenList API交互的基础HTTP请求处理功能。
"""

from json import JSONDecodeError
from typing import Any

import requests

from .exceptions import NotFoundError, OpenListAPIError


class BaseClient:
    """基础API客户端类。

    提供HTTP请求的封装和错误处理。

    Attributes:
        base_url: API基础URL
        timeout: 请求超时时间（秒）
        token: 认证token（可选）
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        token: str | None = None,
    ):
        """初始化客户端。

        Args:
            base_url: API基础URL，例如 "https://api.example.com"
            timeout: 请求超时时间（秒），默认30秒
            token: 认证token，可选
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.token = token
        self.session = requests.Session()

    def set_token(self, token: str) -> None:
        """设置认证token。

        Args:
            token: JWT token
        """
        self.token = token

    def _get_headers(self) -> dict[str, str]:
        """获取请求头。

        Returns:
            包含认证信息的请求头字典
        """
        headers = {
            "Content-Type": "application/json",
        }
        if self.token:
            headers["Authorization"] = self.token
        return headers

    def raise_for_status(self, response: requests.Response) -> None:
        """Raises :class:`HTTPError`, if one occurred."""

        try:
            status_code = response.json().get("code", response.status_code)
            reason = response.json().get("message", response.reason)
        except JSONDecodeError:
            status_code = response.status_code
            if isinstance(response.reason, bytes):
                # We attempt to decode utf-8 first because some servers
                # choose to localize their reason strings. If the string
                # isn't utf-8, we fall back to iso-8859-1 for all other
                # encodings. (See PR #3538)
                try:
                    reason = response.reason.decode("utf-8")
                except UnicodeDecodeError:
                    reason = response.reason.decode("iso-8859-1")
            else:
                reason = response.reason
        except Exception:
            status_code = 500
            reason = "Unknown Error"

        if 400 <= status_code <= 600:
            # OpenList reports a missing object as code=500 + "object not found"
            # (often with HTTP 200): callers rely on a 404 for "not there yet"
            if "not found" in str(reason).lower():
                raise NotFoundError(str(reason), status_code=404, response=response)
            raise OpenListAPIError(reason, status_code=status_code, response=response)

    @staticmethod
    def _to_openlist_error(response: requests.Response) -> OpenListAPIError:
        """Translate an HTTP error into an OpenList error.

        OpenList reports a missing object as **HTTP 500** with a
        "object not found" / "storage not found" message; callers (the desktop
        client, the task scanner) rely on those being 404s.
        """
        try:
            payload = response.json()
            message = str(payload.get("message", "") or "")
            code = payload.get("code") or response.status_code
        except Exception:
            message = (response.text or "")[:200]
            code = response.status_code
        if "not found" in message.lower():
            return NotFoundError(message or "not found", status_code=404, response=response)
        return OpenListAPIError(message or str(response.reason), status_code=code, response=response)

    def _handle_response(self, response: requests.Response) -> Any:
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            raise self._to_openlist_error(response) from e
        self.raise_for_status(response)
        return response.json()

    def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        """发起HTTP请求。

        Args:
            method: HTTP方法 (GET, POST, PUT, DELETE等)
            endpoint: API端点路径
            **kwargs: requests库支持的其他参数

        Returns:
            解析后的响应数据
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()

        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))

        response = self.session.request(
            method=method,
            url=url,
            headers=headers,
            timeout=self.timeout,
            **kwargs,
        )
        return self._handle_response(response)

    def get(self, endpoint: str, **kwargs) -> Any:
        """发起GET请求。"""
        return self._request("GET", endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs) -> Any:
        """发起POST请求。"""
        return self._request("POST", endpoint, **kwargs)

    def put(self, endpoint: str, **kwargs) -> Any:
        """发起PUT请求。"""
        return self._request("PUT", endpoint, **kwargs)

    def delete(self, endpoint: str, **kwargs) -> Any:
        """发起DELETE请求。"""
        return self._request("DELETE", endpoint, **kwargs)
