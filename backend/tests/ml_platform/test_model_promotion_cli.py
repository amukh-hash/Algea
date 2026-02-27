from pathlib import Path
import subprocess
import sys

from backend.app.ml_platform.registry.store import ModelRegistryStore


def test_promotion_cli(tmp_path: Path) -> None:
    db = tmp_path / "reg.sqlite"
    root = tmp_path / "models"
    store = ModelRegistryStore(db, root)
    store.publish_version("chronos2", "v1", "x", {"pinball_loss": 0.1, "calibration_score": 0.9}, {"a": 1}, {"feature_schema": {}, "drift_baseline": {"mean": 0, "std": 1}, "calibration": {"calibration_score": 0.9}})

    cmd = [
        sys.executable,
        "backend/scripts/promote_model.py",
        "--model",
        "chronos2",
        "--version",
        "v1",
        "--to",
        "prod",
        "--metrics",
        '{"pinball_loss":0.1,"calibration_score":0.9,"sharpe":1.1,"max_drawdown":0.1}',
    ]
    env = {
        "MODEL_REGISTRY_DB": str(db),
        "MODEL_ARTIFACT_ROOT": str(root),
    }
    res = subprocess.run(cmd, check=False, capture_output=True, text=True, env={**env, **__import__("os").environ})
    assert res.returncode == 0
    assert store.resolve_alias("chronos2", "prod") == "v1"
