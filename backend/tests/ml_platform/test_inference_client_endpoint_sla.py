from datetime import datetime

import pytest

from backend.app.ml_platform.inference_gateway.client import InferenceGatewayClient, InferenceTimeoutError
from backend.app.ml_platform.inference_gateway.protocol import InferenceRequestBase


class _SlowServer:
    def infer(self, endpoint, req):
        return type("R", (), {"model_name": endpoint, "model_version": "v1", "outputs": {}, "uncertainty": 0.0, "calibration_score": 0.0, "ood_score": 0.0, "latency_ms": 75.0, "warnings": []})()


def test_endpoint_specific_sla_budget_is_used():
    client = InferenceGatewayClient(_SlowServer(), timeout_ms=200, endpoint_timeouts_ms={"smoe_rank": 50})
    req = InferenceRequestBase(asof=datetime.utcnow(), universe_id="u", features_hash="h")
    with pytest.raises(InferenceTimeoutError):
        client.call("smoe_rank", req, critical=True)


def test_other_endpoints_use_default_timeout():
    client = InferenceGatewayClient(_SlowServer(), timeout_ms=200, endpoint_timeouts_ms={"smoe_rank": 50})
    req = InferenceRequestBase(asof=datetime.utcnow(), universe_id="u", features_hash="h")
    assert client.call("chronos2_forecast", req, critical=True) is not None
