from datetime import datetime

from backend.app.ml_platform.inference_gateway.client import InferenceGatewayClient
from backend.app.ml_platform.inference_gateway.protocol import InferenceRequestBase


class _EchoServer:
    def infer(self, endpoint, req):
        return type("R", (), {"model_name": endpoint, "model_version": req.model_version or "alias", "outputs": {}, "uncertainty": 0.0, "calibration_score": 0.0, "ood_score": 0.0, "latency_ms": 1.0, "warnings": []})()


def test_inference_client_pinned_version():
    c = InferenceGatewayClient(_EchoServer())
    req = InferenceRequestBase(asof=datetime.now(), universe_id="u", features_hash="h")
    resp = c.call_pinned("chronos2_forecast", req, model_version="v123")
    assert resp.model_version == "v123"
