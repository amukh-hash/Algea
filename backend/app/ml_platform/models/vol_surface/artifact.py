from __future__ import annotations

import json
from pathlib import Path


def save_vol_surface_artifact(path: Path, config: dict, metrics: dict, drift_baseline: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    # NOTE: weights.safetensors must be provided by the training pipeline
    if not (path / "weights.safetensors").exists():
        from backend.app.core.runtime_mode import ArtifactValidationError
        raise ArtifactValidationError("weights.safetensors not found — training pipeline must provide real weights")
    (path / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (path / "feature_schema.json").write_text(json.dumps({"name": "VolSurfaceSchema"}, indent=2), encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (path / "calibration.json").write_text(json.dumps({"calibration_score": metrics.get("calibration_score", 0.5)}, indent=2), encoding="utf-8")
    (path / "drift_baseline.json").write_text(json.dumps(drift_baseline, indent=2), encoding="utf-8")
    (path / "README_model_card.md").write_text("# vol_surface\n", encoding="utf-8")


def load_vol_surface_artifact(path: Path) -> dict:
    return {
        "config": json.loads((path / "model_config.json").read_text(encoding="utf-8")),
        "drift_baseline": json.loads((path / "drift_baseline.json").read_text(encoding="utf-8")),
        "calibration": json.loads((path / "calibration.json").read_text(encoding="utf-8")),
    }
