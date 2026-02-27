from pathlib import Path

from backend.app.ml_platform.registry.promotion import promote_if_eligible
from backend.app.ml_platform.registry.store import ModelRegistryStore


def test_rl_promotion_gates_block_bad_metrics(tmp_path: Path):
    store = ModelRegistryStore(tmp_path / "r.sqlite", tmp_path / "models")
    store.publish_version("rl_policy", "v1", "x", {"constraint_violation_rate": 0.2, "seed_stability_score": 0.9, "max_drawdown": 0.1, "calibration_score": 0.8, "sharpe": 1.2}, {}, {"feature_schema": {}})
    assert not promote_if_eligible(store, "rl_policy", "v1", {"constraint_violation_rate": 0.2, "seed_stability_score": 0.9, "max_drawdown": 0.1, "calibration_score": 0.8, "sharpe": 1.2})
