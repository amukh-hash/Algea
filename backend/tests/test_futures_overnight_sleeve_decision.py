from pathlib import Path

from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.ml_platform.inference_gateway.client import InferenceGatewayClient
from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer
from backend.app.ml_platform.registry.store import ModelRegistryStore
from backend.app.strategies.futures_overnight.sleeve import FuturesOvernightSleeve


def test_futures_overnight_decision_deterministic(tmp_path: Path) -> None:
    cfg = MLPlatformConfig(
        registry_db_path=tmp_path / "reg.sqlite",
        model_root=tmp_path / "models",
        trace_root=tmp_path / "traces",
    )
    store = ModelRegistryStore(cfg.registry_db_path, cfg.model_root)
    store.publish_version("chronos2", "v1", "abc", {"pinball_loss": 0.1}, {"x": 1}, {"feature_schema": {}, "drift_baseline": {"mean": 100, "std": 1}, "calibration": {"calibration_score": 0.8}})
    store.set_alias("chronos2", "prod", "v1")

    client = InferenceGatewayClient(InferenceGatewayServer(cfg), timeout_ms=500)
    sleeve = FuturesOvernightSleeve(client, enabled=True)
    d1 = sleeve.generate_targets("ES", [100.0, 101.0, 102.0, 103.0], "trace-x", "2026-01-01")
    d2 = sleeve.generate_targets("ES", [100.0, 101.0, 102.0, 103.0], "trace-y", "2026-01-01")
    assert d1["status"] == d2["status"]
    assert d1["targets"] == d2["targets"]
