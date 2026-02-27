from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from ...models.selector_master.context_encoder import encode_market_context
from ...models.selector_smoe.artifact import save_smoe_artifact
from ...models.selector_smoe.model import SMoEConfig, SMoERankerModel
from ..evaluators.ranking_eval import evaluate_ranking


@dataclass
class TrainSMoERankerJob:
    job_type: str
    model_name: str
    version: str
    feature_matrix: list[list[float]]
    labels: list[float]
    market_context: dict[str, float] = field(default_factory=dict)
    n_experts: int = 4
    top_k: int = 1

    def run(self, out_dir: Path) -> dict:
        cfg = SMoEConfig(n_experts=self.n_experts, top_k=self.top_k)
        model = SMoERankerModel(cfg)
        ctx = encode_market_context(self.market_context)

        scores: list[float] = []
        ent: list[float] = []
        util = {i: 0 for i in range(cfg.n_experts)}
        for row in self.feature_matrix:
            out = model.forward_row(row, ctx)
            scores.append(out["score"])
            ent.append(out["router_entropy"])
            util[out["expert_id"]] += 1

        metrics = evaluate_ranking(scores, self.labels, ent, util)
        drift_baseline = {"feature_mean": sum(sum(r) for r in self.feature_matrix) / max(len(self.feature_matrix), 1), "feature_std": 1.0}
        save_smoe_artifact(out_dir, cfg, metrics, drift_baseline)
        sha = hashlib.sha256((out_dir / "weights.safetensors").read_bytes()).hexdigest()
        return {
            "artifact_dir": out_dir,
            "metrics": metrics,
            "config": {"n_experts": self.n_experts, "top_k": self.top_k},
            "lineage": {"feature_schema": {"name": "CrossSectionalSchema"}, "drift_baseline": drift_baseline, "calibration": {"calibration_score": metrics["calibration_score"]}},
            "sha256": sha,
        }
