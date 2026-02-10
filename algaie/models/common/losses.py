"""
Loss functions for foundation and ranker models.

Includes:
  - huber_loss: smooth L1 for regression
  - listwise_softmax_loss: listwise ranking via KL-divergence
  - pairwise_margin_loss: pairwise ranking with margin
  - compute_pinball_loss: quantile (pinball) loss for probabilistic forecasting
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target, beta=delta)


# ---------------------------------------------------------------------------
# Ranking losses  (from deprecated/backend_app_snapshot/models/rank_losses.py)
# ---------------------------------------------------------------------------

def listwise_softmax_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Listwise ranking loss via softmax cross-entropy (KL divergence).

    Parameters
    ----------
    scores  : [N, 1] or [N] predicted ranking scores
    targets : [N, 1] or [N] true relevance (e.g. returns)
    temperature : softmax temperature
    """
    scores = scores.view(-1)
    targets = targets.view(-1)

    pred_dist = F.softmax(scores / temperature, dim=0)
    true_dist = F.softmax(targets / temperature, dim=0)

    return F.kl_div(pred_dist.log(), true_dist, reduction="batchmean")


def pairwise_margin_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """
    Pairwise ranking loss (margin-based).

    For all pairs ``(i, j)`` where ``target_i > target_j``, penalise when
    ``score_i - score_j < margin``.  Feasible for N ≤ ~500 assets.
    """
    scores = scores.view(-1)
    targets = targets.view(-1)
    N = scores.size(0)
    if N < 2:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    s_i = scores.unsqueeze(1)   # [N, 1]
    s_j = scores.unsqueeze(0)   # [1, N]
    t_i = targets.unsqueeze(1)
    t_j = targets.unsqueeze(0)

    y = torch.sign(t_i - t_j)
    mask = y != 0
    loss_mat = F.relu(-y * (s_i - s_j) + margin)

    return (loss_mat * mask).sum() / (mask.sum() + 1e-6)


# ---------------------------------------------------------------------------
# Quantile / pinball loss  (from deprecated/legacy_scripts/teacher/phase1_train_teacher_gold.py)
# ---------------------------------------------------------------------------

def compute_pinball_loss(
    y_true: torch.Tensor,
    quantile_preds: torch.Tensor,
    quantiles: List[float],
    quantile_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Pinball (quantile) loss.

    Parameters
    ----------
    y_true          : [B] or [B, H]
    quantile_preds  : [B, Q] or [B, Q, H']  (H' may exceed H)
    quantiles       : list of float quantile levels (length Q)
    quantile_weights: optional [Q] per-quantile importance weights
    """
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(1)
    if y_true.ndim == 2:
        y_true = y_true.unsqueeze(1)  # [B, 1, H]
    if quantile_preds.ndim == 2:
        quantile_preds = quantile_preds.unsqueeze(-1)

    horizon = y_true.shape[-1]
    if quantile_preds.shape[-1] > horizon:
        quantile_preds = quantile_preds[..., :horizon]

    device = y_true.device
    q_tensor = torch.tensor(quantiles, device=device).view(1, -1, 1)

    errors = y_true - quantile_preds
    loss_per_q = torch.max(errors * q_tensor, -errors * (1 - q_tensor))

    if quantile_weights is not None:
        weights = quantile_weights.view(1, -1, 1).to(device)
        loss_per_q = loss_per_q * weights

    return loss_per_q.mean()
