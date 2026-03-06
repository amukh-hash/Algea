"""
SMoE Expert Ensemble — bridges RankTransformer into the Selector sleeve.

Replaces the deterministic ``(expert_id + 1) * 0.1`` stub with an ensemble
of real ``RankTransformer`` models from ``algae.models.ranker``.

Falls closed if trained weights are not available.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── Legacy stub (preserved for CI / noop mode) ──────────────────────────

def expert_score(features: list[float], expert_id: int) -> float:
    """Legacy deterministic expert stub.

    .. deprecated::
        Use ``SMoEExpertEnsemble.compute_scores()`` for production.
    """
    w = (expert_id + 1) * 0.1
    return sum((i + 1) * w * f for i, f in enumerate(features)) / max(len(features), 1)


# ── Production Expert Ensemble ──────────────────────────────────────────

class SMoEExpertEnsemble(nn.Module):
    """Ensemble of RankTransformers acting as distinct SMoE experts.

    Each expert is a full encoder-only Transformer with quantile,
    direction, and risk heads.  Weights are loaded from individual
    ``.pt`` checkpoint files.

    Parameters
    ----------
    num_experts : int
        Number of expert models in the ensemble.
    d_input : int
        Feature dimensionality per asset in the cross-section.
    weights_dir : str or None
        Directory containing ``expert_0.pt`` … ``expert_{N-1}.pt``.
        If None, all experts have random weights (fail-closed for production).
    device : str
        CUDA device string.
    """

    def __init__(
        self,
        num_experts: int = 4,
        d_input: int = 8,
        weights_dir: Optional[str] = None,
        device: str = "cuda:1",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.is_loaded = False

        try:
            from algae.models.ranker.rank_transformer import RankTransformer
        except ImportError:
            logger.error("Cannot import RankTransformer from algae.models.ranker")
            return

        self.experts = nn.ModuleDict()
        loaded_count = 0

        for i in range(num_experts):
            # Auto-detect architecture from JSON manifest (Deep SMoE Optuna output)
            cfg = {
                "d_input": d_input, "d_model": 128, "n_head": 4,
                "n_layers": 2, "dropout": 0.1, "max_len": 512,
            }
            if weights_dir:
                config_path = Path(weights_dir) / f"expert_{i}_config.json"
                if config_path.exists():
                    import json
                    with open(config_path) as f:
                        loaded_cfg = json.load(f)
                    cfg.update(loaded_cfg)
                    logger.info("Expert %d config loaded: d=%d L=%d H=%d",
                                i, cfg["d_model"], cfg["n_layers"], cfg["n_head"])

            model = RankTransformer(
                d_input=cfg["d_input"],
                d_model=cfg["d_model"],
                n_head=cfg["n_head"],
                n_layers=cfg["n_layers"],
                dropout=cfg["dropout"],
                max_len=cfg["max_len"],
            )
            if weights_dir:
                pt_path = Path(weights_dir) / f"expert_{i}.pt"
                try:
                    state = torch.load(pt_path, map_location=self.device, weights_only=True)
                    model.load_state_dict(state, strict=True)
                    loaded_count += 1
                    logger.info("Loaded SMoE expert %d from %s", i, pt_path)
                except FileNotFoundError:
                    logger.warning("Expert %d weights not found at %s", i, pt_path)
                except Exception as e:
                    logger.warning("Failed to load expert %d: %s", i, e)
            self.experts[f"expert_{i}"] = model

        if loaded_count == num_experts and num_experts > 0:
            self.is_loaded = True
        else:
            logger.warning(
                "SMoE Expert Ensemble: loaded %d/%d experts.  "
                "Fail-closed enforced for production inference.",
                loaded_count, num_experts,
            )

        self.to(self.device, dtype=torch.bfloat16)
        self.eval()

    @torch.inference_mode()
    def compute_scores(
        self, cross_sectional_features: list[list[float]]
    ) -> dict[str, list[float]]:
        """Process a daily equity cross-section through all experts.

        Parameters
        ----------
        cross_sectional_features : list[list[float]]
            Feature matrix of shape ``[num_equities, d_input]``.

        Returns
        -------
        dict[str, list[float]]
            ``{"expert_0": [score_per_equity...], ...}``

        Raises
        ------
        RuntimeError
            If not all experts have trained weights loaded.
        """
        if not self.is_loaded:
            raise RuntimeError(
                "SMoE Expert Ensemble fail-closed: not all experts have "
                "trained weights.  Set ENABLE_SMOE_SELECTOR=0."
            )

        if not cross_sectional_features:
            raise ValueError("Cross-sectional feature matrix cannot be empty.")

        # Shape: [Batch=1, Equities, Features]
        x = torch.tensor(
            cross_sectional_features,
            dtype=torch.bfloat16,
            device=self.device,
        ).unsqueeze(0)

        results: dict[str, list[float]] = {}
        for eid, expert in self.experts.items():
            outputs = expert(x)
            # RankTransformer returns dict with "score" key
            if isinstance(outputs, dict):
                scores_t = outputs.get("score", outputs.get("quantiles", None))
                if scores_t is not None:
                    # score shape: [B, N, 1] → flatten
                    scores = scores_t.squeeze().cpu().float().tolist()
                else:
                    scores = []
            else:
                scores = outputs.squeeze().cpu().float().tolist()

            if isinstance(scores, float):
                scores = [scores]
            results[eid] = scores

        return results
