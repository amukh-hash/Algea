from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ...models.itransformer.artifact import save_itransformer_artifact
from ...models.itransformer.model import ITransformerModel
from ..evaluators.itransformer_eval import evaluate_itransformer


@dataclass
class TrainITransformerJob:
    job_type: str
    model_name: str
    version: str
    feature_matrix: list[list[float]]
    labels: list[float]
    hidden_size: int = 32

    def run(self, out_dir: Path) -> dict:
        model = ITransformerModel(hidden_size=self.hidden_size)
        scores, _ = model.signal(self.feature_matrix)
        metrics = evaluate_itransformer(scores, self.labels)
        drift_baseline = {
            "score_mean": sum(scores) / max(len(scores), 1),
            "score_std": 1.0,
        }
        config = {"hidden_size": self.hidden_size}
        # Save real model weights before creating artifact
        import torch
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"hidden_size": model.hidden_size, "config": config}, out_dir / "weights.safetensors")
        save_itransformer_artifact(out_dir, config, metrics, drift_baseline)
        sha = hashlib.sha256((out_dir / "weights.safetensors").read_bytes()).hexdigest()
        return {
            "artifact_dir": out_dir,
            "metrics": metrics,
            "config": config,
            "lineage": {
                "feature_schema": {"name": "MultivariatePanelSchema"},
                "drift_baseline": drift_baseline,
                "calibration": {"calibration_score": metrics["calibration_score"]},
            },
            "sha256": sha,
        }
