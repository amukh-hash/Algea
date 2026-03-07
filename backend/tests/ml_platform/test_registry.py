import shutil
from pathlib import Path

from backend.app.ml_platform.registry.promotion import promote_if_eligible
from backend.app.ml_platform.registry.store import ModelRegistryStore

DUMMY_WEIGHTS = Path(__file__).parent.parent / "fixtures" / "dummy_weights.safetensors"


def test_publish_promote_and_resolve(tmp_path: Path) -> None:
    store = ModelRegistryStore(tmp_path / "registry.sqlite", tmp_path / "models")

    # Pre-place real weights so publish_version validation passes
    version_dir = tmp_path / "models" / "chronos2" / "v1"
    version_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DUMMY_WEIGHTS, version_dir / "weights.safetensors")

    store.publish_version(
        model_name="chronos2",
        version="v1",
        sha256="deadbeef",
        metrics={"sharpe": 1.2, "max_drawdown": 0.1, "calibration_score": 0.7},
        config={"context_length": 64},
        data_lineage={"feature_schema": {"name": "TSFMSeriesSchema"}},
    )
    ok = promote_if_eligible(
        store,
        "chronos2",
        "v1",
        {"sharpe": 1.2, "max_drawdown": 0.1, "calibration_score": 0.7},
    )
    assert ok
    assert store.resolve_alias("chronos2", "prod") == "v1"
