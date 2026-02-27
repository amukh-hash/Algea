from pathlib import Path

from backend.app.ml_platform.registry.store import ModelRegistryStore


def test_selector_canary_alias_wiring(tmp_path: Path):
    store = ModelRegistryStore(tmp_path / "r.sqlite", tmp_path / "models")
    store.publish_version("selector_smoe", "v2", "x", {"rank_ic": 0.7, "calibration_score": 0.8}, {"n_experts": 4, "top_k": 1}, {"feature_schema": {}, "drift_baseline": {}, "calibration": {"calibration_score": 0.8}})
    store.set_alias("selector_smoe", "canary", "v2")
    assert store.resolve_alias("selector_smoe", "canary") == "v2"
