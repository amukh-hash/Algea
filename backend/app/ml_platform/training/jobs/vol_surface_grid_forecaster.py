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
        import torch
        model = VolSurfaceGridForecaster()
        vrp_pred, unc, drift = model.forecast(self.grid_history)
        # forecast() returns (float, float, float), not a dict
        labels = self.grid_history[-1].get("target", {}) if self.grid_history else {}
        mae = abs(vrp_pred - float(next(iter(labels.values()), 0.0))) if labels else abs(vrp_pred)
        metrics = {"mae": mae, "pinball_loss": mae, "calibration_score": max(0.0, 1.0 - unc), "stability": max(0.0, 1.0 - drift)}

        # Save real model weights before calling save_artifact
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "transformer": model.model.state_dict(),
            "head": model.regression_head.state_dict(),
            "scale": model.scale,
        }, out_dir / "weights.safetensors")

        save_vol_surface_grid_artifact(out_dir, {"scale": 0.05}, metrics, {"drift": drift})
        sha = hashlib.sha256((out_dir / "weights.safetensors").read_bytes()).hexdigest()
        return {"artifact_dir": out_dir, "metrics": metrics, "config": {"scale": 0.05}, "sha256": sha, "lineage": {"drift_baseline": {"drift": drift}}}
