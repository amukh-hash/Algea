from datetime import datetime

import pytest

from backend.app.ml_platform.inference_gateway.client import InferenceGatewayClient, InferenceTimeoutError
from backend.app.ml_platform.inference_gateway.protocol import InferenceRequestBase


class _SlowServer:
    def infer(self, endpoint, req):
        return type("R", (), {"model_name": "x", "model_version": "v1", "outputs": {}, "uncertainty": 0.0, "calibration_score": 0.0, "ood_score": 0.0, "latency_ms": 9999, "warnings": []})()


def test_fail_closed_vol_surface_grid_timeout():
    c = InferenceGatewayClient(_SlowServer(), timeout_ms=10)
    with pytest.raises(InferenceTimeoutError):
        c.call("vol_surface_grid_forecast", InferenceRequestBase(asof=datetime.now(), universe_id="u", features_hash="h"), critical=True)
