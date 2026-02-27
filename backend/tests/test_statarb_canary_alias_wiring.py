from pathlib import Path

from backend.app.ml_platform.registry.store import ModelRegistryStore


def test_statarb_canary_alias_wiring(tmp_path: Path):
    store = ModelRegistryStore(tmp_path / "r.sqlite", tmp_path / "models")
    store.publish_version("itransformer", "v2", "x", {"rank_ic": 0.7, "pair_stability": 0.8, "calibration_score": 0.8}, {"hidden_size": 16}, {"feature_schema": {}, "drift_baseline": {"score_mean": 0.0}, "calibration": {"calibration_score": 0.8}})
    store.set_alias("itransformer", "canary", "v2")
    assert store.resolve_alias("itransformer", "canary") == "v2"
