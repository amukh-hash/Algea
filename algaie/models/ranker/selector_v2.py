"""
Two-Head Rank Selector V2 — shared encoder with rank and trade heads.

Ported from deprecated/backend_app_snapshot/models/selector_v2.py.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoHeadRankSelector(nn.Module):
    """
    Two-head model for cross-sectional ranking **and** trade-probability estimation.

    Inputs : ``[B, N, F]`` normalised feature vectors.
    Outputs: ``(score, p_trade)`` — both ``[B, N]``.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Rank head → real-valued score
        self.rank_head = nn.Linear(hidden_dim, 1)

        # Trade head → P(trade) ∈ [0, 1]
        self.trade_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``x``: ``[B, N, F]`` → ``(score [B, N], p_trade [B, N])``."""
        h = self.encoder(x)
        score = self.rank_head(h).squeeze(-1)
        p_trade = self.trade_head(h).squeeze(-1)
        return score, p_trade


class WeightedPairwiseLoss(nn.Module):
    """
    Composite loss: ``L = L_rank + λ · L_trade``

    * **L_rank** — Boolean-weighted pairwise logistic loss (top-vs-bottom pairs).
    * **L_trade** — Weighted BCE loss on the trade probability head.
    """

    def __init__(
        self,
        trade_lambda: float = 0.25,
        top_bottom_quantile: float = 0.20,
        max_pairs: int = 2000,
        seed: int = 42,
    ):
        super().__init__()
        self.trade_lambda = trade_lambda
        self.q = top_bottom_quantile
        self.max_pairs = max_pairs
        self.seed = seed
        self.bce = nn.BCELoss(reduction="none")

    def forward(
        self,
        scores: torch.Tensor,   # [B, N]
        p_trade: torch.Tensor,  # [B, N]
        y_rank: torch.Tensor,   # [B, N]
        y_trade: torch.Tensor,  # [B, N]
        weights: torch.Tensor,  # [B, N]
        mask: torch.Tensor,     # [B, N]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = scores.shape[0]
        mask_bool = mask.bool()

        # --- Vectorized trade loss (no per-batch loop needed) ---------------
        bce_all = self.bce(p_trade, y_trade)  # [B, N]
        masked_bce = bce_all * weights * mask.float()
        masked_w = weights * mask.float()
        per_batch_w_sum = masked_w.sum(dim=1)  # [B]
        valid_trade = per_batch_w_sum > 0
        per_batch_trade = masked_bce.sum(dim=1) / (per_batch_w_sum + 1e-8)  # [B]
        num_valid = valid_trade.sum().clamp(min=1)
        avg_trade_loss = per_batch_trade[valid_trade].sum() / num_valid

        # --- Pairwise rank loss (variable-length masks require per-batch) ---
        total_rank_loss = torch.tensor(0.0, device=scores.device)
        valid_rank_batches = 0

        rng = torch.Generator(device=scores.device)
        rng.manual_seed(self.seed)

        for b in range(batch_size):
            mask_b = mask_bool[b]
            if not mask_b.any():
                continue

            s = scores[b][mask_b]
            y = y_rank[b][mask_b]
            w = weights[b][mask_b]
            N = s.shape[0]
            k = max(1, int(N * self.q))

            sorted_idx = torch.argsort(y, descending=True)
            top_idx = sorted_idx[:k]
            bottom_idx = sorted_idx[-k:]

            grid_i, grid_j = torch.meshgrid(top_idx, bottom_idx, indexing="ij")
            grid_i = grid_i.flatten()
            grid_j = grid_j.flatten()

            num_pairs = grid_i.shape[0]
            if num_pairs > self.max_pairs:
                perm = torch.randperm(num_pairs, generator=rng)[: self.max_pairs]
                idx_i, idx_j = grid_i[perm], grid_j[perm]
            else:
                idx_i, idx_j = grid_i, grid_j

            if len(idx_i) > 0:
                score_diff = s[idx_i] - s[idx_j]
                w_pair = torch.sqrt(w[idx_i] * w[idx_j])
                loss_ij = F.softplus(-score_diff)
                l_rank = (loss_ij * w_pair).sum() / (w_pair.sum() + 1e-8)
                total_rank_loss = total_rank_loss + l_rank
                valid_rank_batches += 1

        avg_rank_loss = total_rank_loss / max(1, valid_rank_batches)
        total_loss = avg_rank_loss + self.trade_lambda * avg_trade_loss

        return total_loss, {
            "loss_rank": avg_rank_loss,
            "loss_trade": avg_trade_loss,
            "mean_p_trade": p_trade[mask_bool].mean() if mask_bool.any() else torch.tensor(0.0),
        }
