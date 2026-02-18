"""Request-ID + structured logging middleware for the API."""
from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("algaie.api")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach X-Request-ID to every request and log structured timings."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4())[:12])
        request.state.request_id = request_id

        path = request.url.path
        method = request.method

        logger.info("req_start request_id=%s method=%s path=%s", request_id, method, path)
        t0 = time.monotonic()

        try:
            response = await call_next(request)
        except Exception:
            elapsed = round((time.monotonic() - t0) * 1000, 1)
            logger.exception(
                "req_error request_id=%s method=%s path=%s duration_ms=%s",
                request_id, method, path, elapsed,
            )
            raise

        elapsed = round((time.monotonic() - t0) * 1000, 1)
        response.headers["X-Request-ID"] = request_id

        log_fn = logger.warning if response.status_code >= 400 else logger.info
        log_fn(
            "req_end request_id=%s method=%s path=%s status=%s duration_ms=%s",
            request_id, method, path, response.status_code, elapsed,
        )
        return response
