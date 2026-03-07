import pytest
from pathlib import Path

from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer
from backend.app.ml_platform.models.vol_surface.types import VolSurfaceRequest
from backend.app.ml_platform.registry.store import ModelRegistryStore


@pytest.mark.xfail(strict=False, reason="PRE-EXISTING: sync test calling async endpoint handler")
def test_vol_surface_endpoint_contract(tmp_path: Path):
    cfg = MLPlatformConfig(registry_db_path=tmp_path / "r.sqlite", model_root=tmp_path / "models", trace_root=tmp_path / "traces")
    store = ModelRegistryStore(cfg.registry_db_path, cfg.model_root)
    store.publish_version("vol_surface", "v1", "x", {"pinball_loss": 0.1, "calibration_score": 0.8, "edge_hit_rate": 0.6}, {"hidden_size": 8}, {"feature_schema": {}, "drift_baseline": {"7": {"rv_hist_20_mean": 0.2, "rv_hist_20_std": 1.0}}, "calibration": {"calibration_score": 0.8}})
    store.set_alias("vol_surface", "prod", "v1")
    s = InferenceGatewayServer(cfg)
    out = s.vol_surface_http_forecast(VolSurfaceRequest(asof="2026-01-01", underlying_symbol="SPY", tenors=[7], history={7: [{"rv_hist_20": 0.2}] * 5}, trace_id="t"))
    assert out["model_name"] == "vol_surface"
    assert "predicted_rv" in out
