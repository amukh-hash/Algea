from pathlib import Path

from backend.app.ml_platform.registry.promotion import promote_if_eligible
from backend.app.ml_platform.registry.store import ModelRegistryStore


def test_itransformer_promotion_gates(tmp_path: Path):
    store = ModelRegistryStore(tmp_path / "r.sqlite", tmp_path / "models")
    store.publish_version("itransformer", "v1", "x", {"rank_ic": 0.7, "pair_stability": 0.8, "calibration_score": 0.8, "sharpe": 1.1, "max_drawdown": 0.1}, {"hidden_size": 16}, {"feature_schema": {}, "drift_baseline": {"score_mean": 0.0}, "calibration": {"calibration_score": 0.8}})
    assert promote_if_eligible(store, "itransformer", "v1", {"rank_ic": 0.7, "pair_stability": 0.8, "calibration_score": 0.8, "sharpe": 1.1, "max_drawdown": 0.1})
