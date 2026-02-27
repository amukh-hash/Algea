from pathlib import Path

from backend.app.ml_platform.registry.promotion import promote_if_eligible
from backend.app.ml_platform.registry.store import ModelRegistryStore


def test_smoe_promotion_gates(tmp_path: Path):
    store = ModelRegistryStore(tmp_path / "r.sqlite", tmp_path / "models")
    store.publish_version("selector_smoe", "v1", "x", {"rank_ic": 0.7, "top_bottom_spread": 0.1, "load_balance_score": 0.2, "router_entropy_mean": 0.2, "calibration_score": 0.8, "sharpe": 1.1, "max_drawdown": 0.1}, {"n": 1}, {"feature_schema": {}, "drift_baseline": {}, "calibration": {"calibration_score": 0.8}})
    assert promote_if_eligible(store, "selector_smoe", "v1", {"rank_ic": 0.7, "top_bottom_spread": 0.1, "load_balance_score": 0.2, "router_entropy_mean": 0.2, "calibration_score": 0.8, "sharpe": 1.1, "max_drawdown": 0.1})
