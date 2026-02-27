from __future__ import annotations

import json
from pathlib import Path


REQUIRED_FILES = {
    "weights.safetensors",
    "model_config.json",
    "feature_schema.json",
    "metrics.json",
    "calibration.json",
    "drift_baseline.json",
    "README_model_card.md",
}


def save_rl_policy_artifact(path: Path, config: dict, metrics: dict, drift_baseline: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "weights.safetensors").write_bytes(b"rl-policy-weights")
    (path / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (path / "feature_schema.json").write_text(json.dumps({"name": "RLSizingSchema"}, indent=2), encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (path / "calibration.json").write_text(json.dumps({"ece": metrics.get("ece", 0.0)}, indent=2), encoding="utf-8")
    (path / "drift_baseline.json").write_text(json.dumps(drift_baseline, indent=2), encoding="utf-8")
    (path / "README_model_card.md").write_text("# rl_policy\n", encoding="utf-8")


def load_rl_policy_artifact(path: Path) -> dict:
    missing = sorted(f for f in REQUIRED_FILES if not (path / f).exists())
    if missing:
        raise ValueError(f"rl_policy artifact missing: {missing}")
    return {
        "config": json.loads((path / "model_config.json").read_text(encoding="utf-8")),
        "drift_baseline": json.loads((path / "drift_baseline.json").read_text(encoding="utf-8")),
        "metrics": json.loads((path / "metrics.json").read_text(encoding="utf-8")),
    }
