from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.orchestrator.job_defs import _build_inference_client


class _Server:
    def infer(self, endpoint, req):
        raise AssertionError("not called")


def test_orchestrator_builds_client_with_configured_sla_budgets():
    cfg = MLPlatformConfig()
    client = _build_inference_client({}, _Server())
    assert client.endpoint_timeouts_ms == cfg.inference_endpoint_timeouts_ms()
    assert client.timeout_ms == max(cfg.inference_endpoint_timeouts_ms().values())
