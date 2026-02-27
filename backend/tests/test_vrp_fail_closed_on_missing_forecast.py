import pytest

from backend.app.ml_platform.inference_gateway.client import InferenceGatewayClient
from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer
from backend.app.strategies.vrp.vrp_sleeve import VRPSleeve


def test_vrp_fail_closed_missing_forecast():
    s = InferenceGatewayServer.__new__(InferenceGatewayServer)
    s._handlers = {}
    s.endpoints = {}
    s.latency_p95_ms = {}
    s.model_status = {}
    sleeve = VRPSleeve(InferenceGatewayClient(s, timeout_ms=10))
    with pytest.raises(KeyError):
        sleeve.generate_targets("2026-01-01", "SPY", {7: 0.2}, {7: {"rv_hist_20": 0.1}}, "t")
