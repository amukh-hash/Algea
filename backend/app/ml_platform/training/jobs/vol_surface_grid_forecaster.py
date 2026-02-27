from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ...models.vol_surface_grid.artifact import save_vol_surface_grid_artifact
from ...models.vol_surface_grid.model import VolSurfaceGridForecaster


@dataclass
class TrainVolSurfaceGridForecasterJob:
    job_type: str
    model_name: str
    version: str
    grid_history: list[dict]

    def run(self, out_dir: Path) -> dict:
        model = VolSurfaceGridForecaster()
        pred, unc, drift = model.forecast(self.grid_history)
        labels = self.grid_history[-1].get("target", {}) if self.grid_history else {}
        mae = sum(abs(pred.get(k, 0.0) - float(v)) for k, v in labels.items()) / max(len(labels), 1)
        metrics = {"mae": mae, "pinball_loss": mae, "calibration_score": max(0.0, 1.0 - unc), "stability": max(0.0, 1.0 - drift)}
        save_vol_surface_grid_artifact(out_dir, {"scale": 0.05}, metrics, {"drift": drift})
        sha = hashlib.sha256((out_dir / "weights.safetensors").read_bytes()).hexdigest()
        return {"artifact_dir": out_dir, "metrics": metrics, "config": {"scale": 0.05}, "sha256": sha, "lineage": {"drift_baseline": {"drift": drift}}}
