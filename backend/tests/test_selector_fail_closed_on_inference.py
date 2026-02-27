import pytest

from backend.app.ml_platform.inference_gateway.client import InferenceGatewayClient
from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer
from backend.app.strategies.selector.selector_sleeve import SelectorSleeve


def test_selector_fail_closed_on_inference():
    server = InferenceGatewayServer.__new__(InferenceGatewayServer)
    server._handlers = {}
    server.endpoints = {}
    server.latency_p95_ms = {}
    server.model_status = {}
    client = InferenceGatewayClient(server, timeout_ms=1)
    sleeve = SelectorSleeve(client)
    with pytest.raises(KeyError):
        sleeve.generate_targets("2026-01-01", ["A"], [[0.1]], "t", {})
