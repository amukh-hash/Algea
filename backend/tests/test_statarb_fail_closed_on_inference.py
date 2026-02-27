import pytest

from backend.app.ml_platform.inference_gateway.client import InferenceGatewayClient
from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer
from backend.app.strategies.statarb.sleeve import StatArbSleeve


def test_statarb_fail_closed_on_inference():
    s = InferenceGatewayServer.__new__(InferenceGatewayServer)
    s._handlers = {}
    s.endpoints = {}
    s.latency_p95_ms = {}
    s.model_status = {}
    sleeve = StatArbSleeve(InferenceGatewayClient(s, timeout_ms=1))
    with pytest.raises(KeyError):
        sleeve.generate_targets("2026-01-01", ["XLF"], [[0.1]], "x")
