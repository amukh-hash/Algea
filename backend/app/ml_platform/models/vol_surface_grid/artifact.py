from __future__ import annotations

import json
from pathlib import Path


def save_vol_surface_grid_artifact(out_dir: Path, config: dict, metrics: dict, drift_baseline: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_config.json").write_text(json.dumps(config, sort_keys=True, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, sort_keys=True, indent=2), encoding="utf-8")
    (out_dir / "drift_baseline.json").write_text(json.dumps(drift_baseline, sort_keys=True, indent=2), encoding="utf-8")
    (out_dir / "feature_schema.json").write_text(json.dumps({"name": "VolSurfaceGridSchema"}, sort_keys=True, indent=2), encoding="utf-8")
    (out_dir / "calibration.json").write_text(json.dumps({"calibration_score": metrics.get("calibration_score", 0.0)}, sort_keys=True, indent=2), encoding="utf-8")
    (out_dir / "README_model_card.md").write_text("# vol_surface_grid\n", encoding="utf-8")
    (out_dir / "weights.safetensors").write_text("grid-stub", encoding="utf-8")
