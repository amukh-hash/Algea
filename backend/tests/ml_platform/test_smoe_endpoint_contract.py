import pytest
from pathlib import Path

from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer
from backend.app.ml_platform.models.selector_smoe.types import SMoERankRequest
from backend.app.ml_platform.registry.store import ModelRegistryStore


@pytest.mark.asyncio
async def test_smoe_endpoint_contract(tmp_path: Path):
    cfg = MLPlatformConfig(registry_db_path=tmp_path / "r.sqlite", model_root=tmp_path / "models", trace_root=tmp_path / "traces")
    store = ModelRegistryStore(cfg.registry_db_path, cfg.model_root)
    store.publish_version("selector_smoe", "v1", "x", {"rank_ic": 0.6, "calibration_score": 0.7}, {"n_experts": 4, "top_k": 1}, {"feature_schema": {}, "drift_baseline": {}, "calibration": {"calibration_score": 0.7}})
    store.set_alias("selector_smoe", "prod", "v1")

    s = InferenceGatewayServer(cfg)
    out = await s.smoe_http_rank(SMoERankRequest(asof="2026-01-01", symbols=["AAPL"], feature_matrix=[[0.1, 0.2]], trace_id="t", market_context={}))
    assert out["model_name"] == "selector_smoe"
    assert "scores" in out and "AAPL" in out["scores"]
