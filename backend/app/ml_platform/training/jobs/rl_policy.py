from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ...models.rl_policy.artifact import save_rl_policy_artifact
from ...rl.env_vrp import VRPSizingEnv
from ...rl.rollout_worker import run_rollout
from ..evaluators.rl_eval import evaluate_rl_policy


@dataclass
class TrainRLPolicyJob:
    job_type: str
    model_name: str
    version: str
    steps: int = 32
    hidden_size: int = 32
    seed: int = 7

    def run(self, out_dir: Path) -> dict:
        env = VRPSizingEnv(seed=self.seed, horizon=max(4, self.steps // 4))
        transitions = run_rollout(env, self.steps, seed=self.seed)
        returns = [float(t["reward"]) for t in transitions]
        violations = [1 if t["action"]["size_multiplier"] > 1 else 0 for t in transitions]
        metrics = evaluate_rl_policy(returns, violations)
        config = {"algo": "td3", "hidden_size": self.hidden_size, "seed": self.seed}
        drift_baseline = {"state_mean": 0.0}
        save_rl_policy_artifact(out_dir, config=config, metrics=metrics, drift_baseline=drift_baseline)
        sha = hashlib.sha256((out_dir / "weights.safetensors").read_bytes()).hexdigest()
        return {
            "artifact_dir": out_dir,
            "metrics": metrics,
            "config": config,
            "lineage": {
                "feature_schema": {"name": "RLSizingSchema"},
                "drift_baseline": drift_baseline,
                "calibration": {"ece": 0.05},
            },
            "sha256": sha,
        }
