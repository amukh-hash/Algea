from __future__ import annotations

from typing import Any, Callable


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: Any = None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class APIRouter:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def _decorator(self, *args, **kwargs):
        def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn

        return _wrap

    get = _decorator
    put = _decorator
    post = _decorator
