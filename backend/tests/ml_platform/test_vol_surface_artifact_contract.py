from pathlib import Path

from backend.app.ml_platform.models.vol_surface.artifact import save_vol_surface_artifact


def test_vol_surface_artifact_contract(tmp_path: Path):
    import torch; torch.save({"_test": True}, tmp_path / "weights.safetensors")
    save_vol_surface_artifact(tmp_path, {"hidden_size": 8}, {"pinball_loss": 0.1}, {"7": {"rv_hist_20_mean": 0.2, "rv_hist_20_std": 1.0}})
    for name in ["weights.safetensors", "model_config.json", "feature_schema.json", "metrics.json", "calibration.json", "drift_baseline.json", "README_model_card.md"]:
        assert (tmp_path / name).exists()
