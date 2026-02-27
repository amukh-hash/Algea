from pathlib import Path

from backend.app.ml_platform.registry.store import ModelRegistryStore


def test_canary_alias_resolution(tmp_path: Path) -> None:
    store = ModelRegistryStore(tmp_path / "reg.sqlite", tmp_path / "models")
    store.publish_version("chronos2", "v2", "x", {"pinball_loss": 0.1}, {"a": 1}, {"feature_schema": {}, "drift_baseline": {"mean": 0, "std": 1}, "calibration": {"calibration_score": 0.8}})
    store.set_alias("chronos2", "canary", "v2")
    assert store.resolve_alias("chronos2", "canary") == "v2"
