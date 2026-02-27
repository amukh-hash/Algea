from pathlib import Path

from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.ml_platform.inference_gateway.client import InferenceGatewayClient
from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer
from backend.app.ml_platform.registry.store import ModelRegistryStore
from backend.app.strategies.selector.selector_sleeve import SelectorSleeve


def test_selector_determinism_same_inputs(tmp_path: Path):
    cfg = MLPlatformConfig(registry_db_path=tmp_path / "r.sqlite", model_root=tmp_path / "models", trace_root=tmp_path / "traces")
    store = ModelRegistryStore(cfg.registry_db_path, cfg.model_root)
    store.publish_version("selector_smoe", "v1", "x", {"rank_ic": 0.6, "calibration_score": 0.7}, {"n_experts": 4, "top_k": 1}, {"feature_schema": {}, "drift_baseline": {}, "calibration": {"calibration_score": 0.7}})
    store.set_alias("selector_smoe", "prod", "v1")
    sleeve = SelectorSleeve(InferenceGatewayClient(InferenceGatewayServer(cfg), timeout_ms=1000))
    a = sleeve.generate_targets("2026-01-01", ["A", "B"], [[0.1], [0.2]], "d1", {})
    b = sleeve.generate_targets("2026-01-01", ["A", "B"], [[0.1], [0.2]], "d2", {})
    assert a["targets"] == b["targets"]
