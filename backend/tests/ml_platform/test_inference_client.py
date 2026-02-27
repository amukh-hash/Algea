from datetime import datetime
import time

import pytest

from backend.app.ml_platform.inference_gateway.client import (
    InferenceGatewayClient,
    InferenceTimeoutError,
)
from backend.app.ml_platform.inference_gateway.protocol import InferenceRequestBase
from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer


def test_timeout_fail_closed() -> None:
    server = InferenceGatewayServer()

    def slow(_req: InferenceRequestBase) -> dict:
        time.sleep(0.03)
        return {"outputs": {"score": 1}}

    server.register("smoe_rank", slow)
    client = InferenceGatewayClient(server, timeout_ms=5)
    req = InferenceRequestBase(datetime.utcnow(), "u", "h")
    with pytest.raises(InferenceTimeoutError):
        client.call("smoe_rank", req, critical=True)


def test_noncritical_drop_policy() -> None:
    server = InferenceGatewayServer()
    server.register("smoe_rank", lambda _req: {"outputs": {"score": 1}})
    client = InferenceGatewayClient(server, timeout_ms=200)
    req = InferenceRequestBase(datetime.utcnow(), "u", "h")
    assert client.call("smoe_rank", req, critical=False) is not None
