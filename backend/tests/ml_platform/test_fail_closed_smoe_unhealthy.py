import pytest

from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer


def test_fail_closed_smoe_unhealthy(tmp_path):
    cfg = MLPlatformConfig(registry_db_path=tmp_path / "none.sqlite", model_root=tmp_path / "none_models", trace_root=tmp_path / "traces")
    s = InferenceGatewayServer(cfg)
    # No alias seeded for selector_smoe
    with pytest.raises(RuntimeError):
        s.get_ready()
