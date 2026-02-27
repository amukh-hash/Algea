from pathlib import Path

from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.ml_platform.inference_gateway.client import InferenceGatewayClient
from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer
from backend.app.ml_platform.registry.store import ModelRegistryStore
from backend.app.strategies.statarb.sleeve import StatArbSleeve


def test_statarb_sleeve_integration(tmp_path: Path):
    cfg = MLPlatformConfig(registry_db_path=tmp_path / "r.sqlite", model_root=tmp_path / "models", trace_root=tmp_path / "traces")
    store = ModelRegistryStore(cfg.registry_db_path, cfg.model_root)
    store.publish_version("itransformer", "v1", "x", {"rank_ic": 0.7, "pair_stability": 0.8, "calibration_score": 0.8, "sharpe": 1.1, "max_drawdown": 0.1}, {"hidden_size": 16}, {"feature_schema": {}, "drift_baseline": {"score_mean": 0.0}, "calibration": {"calibration_score": 0.8}})
    store.set_alias("itransformer", "prod", "v1")
    sleeve = StatArbSleeve(InferenceGatewayClient(InferenceGatewayServer(cfg), timeout_ms=1000))
    out = sleeve.generate_targets("2026-01-01", ["XLF", "XLK", "XLE"], [[0.1, 0.2], [0.2, 0.1], [0.3, 0.0]], "st1")
    assert out["status"] == "ok"
    assert out["targets"]
