"""
StatArb V3 — 5-Fold iTransformer Ensemble

Loads all 5 cross-validated iTransformer checkpoints, runs inference
in parallel, and averages predictions for consensus delta-Z forecasts.

Architecture: d=128, L=2, H=4 (from Optuna best trial)
Input:  [1, 60, 10]  (60 bars × 10 idiosyncratic pairs)
Output: [10]          (predicted delta-Z for each pair)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class StatArbV3Ensemble:
    """Load and ensemble 5-fold iTransformer for StatArb V3 inference."""

    def __init__(
        self,
        weights_dir: str = "backend/artifacts/model_weights",
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.models: list[torch.nn.Module] = []
        self.is_loaded = False

        w_dir = Path(weights_dir)
        config_path = w_dir / "itransformer_config.json"

        if not config_path.exists():
            logger.error("iTransformer config not found at %s", config_path)
            return

        try:
            from algae.models.tsfm.itransformer import iTransformer
        except ImportError:
            logger.error("algae.models.tsfm.itransformer not importable")
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            loaded_count = 0
            for i in range(1, 6):
                pt_path = w_dir / f"itransformer_f{i}.pt"
                if not pt_path.exists():
                    logger.warning("Missing fold %d: %s", i, pt_path)
                    continue

                model = iTransformer(
                    num_variates=cfg.get("num_variates", 10),
                    lookback_len=cfg.get("lookback_len", 60),
                    pred_len=cfg.get("pred_len", 1),
                    d_model=cfg.get("d_model", 128),
                    n_heads=cfg.get("n_heads", 4),
                    e_layers=cfg.get("e_layers", 2),
                    dropout=0.0,  # Inference: no dropout
                )
                state = torch.load(pt_path, map_location=self.device, weights_only=True)
                model.load_state_dict(state, strict=True)
                model.to(self.device)
                model.eval()
                self.models.append(model)
                loaded_count += 1

            if loaded_count > 0:
                logger.info(
                    "Armed %d/5 iTransformer folds for StatArb V3 on %s",
                    loaded_count, self.device,
                )
                self.is_loaded = True
        except Exception as e:
            logger.error("Failed to load StatArb V3 ensemble: %s", e)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> Optional[np.ndarray]:
        """Run consensus prediction across all loaded folds.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``[1, 60, 10]`` — the live feature tensor from builder.

        Returns
        -------
        np.ndarray or None
            Shape ``[10]`` — predicted delta-Z for each of the 10 pairs.
        """
        if not self.is_loaded:
            return None

        x = x.to(self.device).float()

        preds = []
        for model in self.models:
            p = model(x)  # [1, 1, 10] or [1, 10]
            if p.ndim == 3:
                p = p.squeeze(1)  # [1, 1, 10] -> [1, 10]
            preds.append(p)

        # Average across folds and drop batch dim -> [10]
        consensus = torch.stack(preds).mean(dim=0).squeeze(0)
        return consensus.cpu().numpy()
