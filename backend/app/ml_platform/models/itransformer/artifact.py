from __future__ import annotations

import json
from pathlib import Path


def save_itransformer_artifact(path: Path, config: dict, metrics: dict, drift_baseline: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "weights.safetensors").write_bytes(b"itransformer-weights")
    (path / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (path / "feature_schema.json").write_text(json.dumps({"name": "MultivariatePanelSchema"}, indent=2), encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (path / "calibration.json").write_text(json.dumps({"calibration_score": metrics.get("calibration_score", 0.7)}, indent=2), encoding="utf-8")
    (path / "drift_baseline.json").write_text(json.dumps(drift_baseline, indent=2), encoding="utf-8")
    (path / "README_model_card.md").write_text("# itransformer\n", encoding="utf-8")
