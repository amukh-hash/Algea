from datetime import datetime

import pytest

from backend.app.ml_platform.inference_gateway.client import InferenceGatewayClient, InferenceTimeoutError
from backend.app.ml_platform.inference_gateway.protocol import InferenceRequestBase
from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer


def test_fail_closed_rl_policy_timeout():
    server = InferenceGatewayServer.__new__(InferenceGatewayServer)
    server._handlers = {"rl_policy_act": lambda _req: {"outputs": {}, "model_version": "v"}}
    server.endpoints = {"rl_policy_act": True}
    server.latency_p95_ms = {}
    server.model_status = {}
    orig = server.infer

    def slow(endpoint, req):
        resp = orig(endpoint, req)
        resp.latency_ms = 999
        return resp

    server.infer = slow  # type: ignore[assignment]
    client = InferenceGatewayClient(server, timeout_ms=10)
    req = InferenceRequestBase(asof=datetime.utcnow(), universe_id="vrp", features_hash="h", payload={})
    with pytest.raises(InferenceTimeoutError):
        client.call("rl_policy_act", req, critical=True)
