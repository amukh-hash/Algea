from __future__ import annotations

import json
from pathlib import Path

from .model import SMoEConfig


def save_smoe_artifact(path: Path, config: SMoEConfig, metrics: dict, drift_baseline: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "weights.safetensors").write_bytes(b"smoe-weights")
    (path / "model_config.json").write_text(json.dumps({"n_experts": config.n_experts, "top_k": config.top_k}, indent=2), encoding="utf-8")
    (path / "feature_schema.json").write_text(json.dumps({"name": "CrossSectionalSchema"}, indent=2), encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (path / "calibration.json").write_text(json.dumps({"calibration_score": 0.7}, indent=2), encoding="utf-8")
    (path / "drift_baseline.json").write_text(json.dumps(drift_baseline, indent=2), encoding="utf-8")
    (path / "README_model_card.md").write_text("# selector_smoe\n", encoding="utf-8")


def load_smoe_artifact(path: Path) -> dict:
    return {
        "config": json.loads((path / "model_config.json").read_text(encoding="utf-8")),
        "metrics": json.loads((path / "metrics.json").read_text(encoding="utf-8")),
        "drift_baseline": json.loads((path / "drift_baseline.json").read_text(encoding="utf-8")),
    }
