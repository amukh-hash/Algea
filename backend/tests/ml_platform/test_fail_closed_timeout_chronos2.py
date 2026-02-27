from datetime import datetime

import pytest

from backend.app.ml_platform.inference_gateway.client import InferenceGatewayClient, InferenceTimeoutError
from backend.app.ml_platform.inference_gateway.protocol import InferenceRequestBase
from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer


def test_fail_closed_timeout() -> None:
    server = InferenceGatewayServer.__new__(InferenceGatewayServer)
    server._handlers = {"chronos2_forecast": lambda _req: {"outputs": {}, "model_version": "v"}}
    server.endpoints = {"chronos2_forecast": True}
    server.latency_p95_ms = {}
    server.model_status = {}
    orig = server.infer

    def slow(endpoint, req):
        resp = orig(endpoint, req)
        resp.latency_ms = 300
        return resp

    server.infer = slow  # type: ignore[assignment]
    client = InferenceGatewayClient(server, timeout_ms=10)
    req = InferenceRequestBase(asof=datetime.utcnow(), universe_id="u", features_hash="h", payload={})
    with pytest.raises(InferenceTimeoutError):
        client.call("chronos2_forecast", req, critical=True)
