from pathlib import Path

from backend.app.ml_platform.models.selector_smoe.artifact import save_smoe_artifact
from backend.app.ml_platform.models.selector_smoe.model import SMoEConfig


def test_smoe_artifact_files(tmp_path: Path):
    import torch; torch.save({"_test": True}, tmp_path / "weights.safetensors")
    save_smoe_artifact(tmp_path, SMoEConfig(), {"rank_ic": 0.6}, {"x": 1})
    for f in ["weights.safetensors", "model_config.json", "feature_schema.json", "metrics.json", "calibration.json", "drift_baseline.json", "README_model_card.md"]:
        assert (tmp_path / f).exists()
