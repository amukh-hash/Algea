from pathlib import Path

from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.ml_platform.inference_gateway.client import InferenceGatewayClient
from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer
from backend.app.ml_platform.registry.store import ModelRegistryStore
from backend.app.strategies.vrp.vrp_sleeve import VRPSleeve


def test_vrp_sleeve_uses_forecast(tmp_path: Path):
    cfg = MLPlatformConfig(registry_db_path=tmp_path / "r.sqlite", model_root=tmp_path / "models", trace_root=tmp_path / "traces")
    store = ModelRegistryStore(cfg.registry_db_path, cfg.model_root)
    store.publish_version("vol_surface", "v1", "x", {"pinball_loss": 0.1, "calibration_score": 0.8, "edge_hit_rate": 0.6}, {"hidden_size": 8}, {"feature_schema": {}, "drift_baseline": {"7": {"rv_hist_20_mean": 0.2, "rv_hist_20_std": 1.0}, "14": {"rv_hist_20_mean": 0.2, "rv_hist_20_std": 1.0}, "30": {"rv_hist_20_mean": 0.2, "rv_hist_20_std": 1.0}, "60": {"rv_hist_20_mean": 0.2, "rv_hist_20_std": 1.0}}, "calibration": {"calibration_score": 0.8}})
    store.set_alias("vol_surface", "prod", "v1")

    sleeve = VRPSleeve(InferenceGatewayClient(InferenceGatewayServer(cfg), timeout_ms=1000))
    out = sleeve.generate_targets("2026-01-01", "SPY", {7: 0.25, 14: 0.26, 30: 0.27, 60: 0.28}, {7: {"rv_hist_20": 0.1}, 14: {"rv_hist_20": 0.1}, 30: {"rv_hist_20": 0.1}, 60: {"rv_hist_20": 0.1}}, "vrp-1")
    assert out["status"] == "ok"
