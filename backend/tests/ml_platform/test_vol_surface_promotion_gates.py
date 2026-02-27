from pathlib import Path

from backend.app.ml_platform.registry.promotion import promote_if_eligible
from backend.app.ml_platform.registry.store import ModelRegistryStore


def test_vol_surface_promotion_gate(tmp_path: Path):
    store = ModelRegistryStore(tmp_path / "r.sqlite", tmp_path / "models")
    store.publish_version("vol_surface", "v1", "x", {"pinball_loss": 0.1, "edge_hit_rate": 0.6, "calibration_score": 0.8, "sharpe": 1.1, "max_drawdown": 0.1}, {"hidden_size": 8}, {"feature_schema": {}, "drift_baseline": {"7": {}}, "calibration": {"calibration_score": 0.8}})
    assert promote_if_eligible(store, "vol_surface", "v1", {"pinball_loss": 0.1, "edge_hit_rate": 0.6, "calibration_score": 0.8, "sharpe": 1.1, "max_drawdown": 0.1})
