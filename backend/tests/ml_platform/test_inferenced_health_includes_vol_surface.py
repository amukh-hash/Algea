from pathlib import Path

from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer
from backend.app.ml_platform.registry.store import ModelRegistryStore


def test_health_includes_vol_surface(tmp_path: Path):
    cfg = MLPlatformConfig(registry_db_path=tmp_path / "r.sqlite", model_root=tmp_path / "models", trace_root=tmp_path / "traces")
    store = ModelRegistryStore(cfg.registry_db_path, cfg.model_root)
    store.publish_version("vol_surface", "v1", "x", {"pinball_loss": 0.1, "calibration_score": 0.8, "edge_hit_rate": 0.6}, {"hidden_size": 8}, {"feature_schema": {}, "drift_baseline": {"7": {"rv_hist_20_mean": 0.2, "rv_hist_20_std": 1.0}}, "calibration": {"calibration_score": 0.8}})
    store.set_alias("vol_surface", "prod", "v1")
    s = InferenceGatewayServer(cfg)
    h = s.get_health()
    assert "vol_surface:prod" in h["models"]
