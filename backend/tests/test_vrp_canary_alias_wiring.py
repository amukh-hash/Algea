from pathlib import Path

from backend.app.ml_platform.registry.store import ModelRegistryStore


def test_vrp_canary_alias_wiring(tmp_path: Path):
    store = ModelRegistryStore(tmp_path / "r.sqlite", tmp_path / "models")
    store.publish_version("vol_surface", "v2", "x", {"pinball_loss": 0.1, "edge_hit_rate": 0.6, "calibration_score": 0.8}, {"hidden_size": 8}, {"feature_schema": {}, "drift_baseline": {"7": {}}, "calibration": {"calibration_score": 0.8}})
    store.set_alias("vol_surface", "canary", "v2")
    assert store.resolve_alias("vol_surface", "canary") == "v2"
