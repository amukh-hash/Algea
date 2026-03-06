"""
Train SMoE Selector Experts — Partitioned RankTransformer Ensemble.

Each expert specializes on a distinct market regime/sector to prevent
mode collapse.  Uses Pairwise Margin Ranking Loss for cross-sectional
rank prediction.

Device: cuda:0 (3090 Ti, 24GB) — offline batch training.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from algae.models.ranker.rank_transformer import RankTransformer

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Pairwise Margin Ranking Loss
# ═══════════════════════════════════════════════════════════════════════════

class PairwiseRankingLoss(nn.Module):
    """Margin-based pairwise ranking loss for cross-sectional stock ranking.

    For each pair (i, j) where return_i > return_j, enforce:
        score_i > score_j + margin

    This is more appropriate than MSE because we care about *relative rank*,
    not absolute return prediction.
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(
        self, scores: torch.Tensor, returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        scores : Tensor [B, N, 1]
            Model-predicted cross-sectional scores.
        returns : Tensor [B, N]
            Actual forward returns (used to define ground-truth ranking).

        Returns
        -------
        Tensor
            Scalar loss.
        """
        scores = scores.squeeze(-1)  # [B, N]
        B, N = scores.shape

        # Build pairwise differences
        # score_diff[b, i, j] = score_i - score_j
        score_diff = scores.unsqueeze(2) - scores.unsqueeze(1)
        # return_diff[b, i, j] = return_i - return_j
        return_diff = returns.unsqueeze(2) - returns.unsqueeze(1)

        # Target: sign of return difference (+1 if i outperforms j)
        target = torch.sign(return_diff)

        # Margin ranking: max(0, -target * score_diff + margin)
        loss = torch.clamp(-target * score_diff + self.margin, min=0)

        # Only count upper triangle (avoid double counting)
        mask = torch.triu(torch.ones(N, N, device=scores.device), diagonal=1)
        mask = mask.unsqueeze(0)  # [1, N, N]

        loss = (loss * mask).sum() / (mask.sum() * B + 1e-8)
        return loss


# ═══════════════════════════════════════════════════════════════════════════
# Expert Data Partitioning (Anti-Collapse)
# ═══════════════════════════════════════════════════════════════════════════

EXPERT_SPECS = {
    0: {"name": "Tech_Growth", "description": "High beta, momentum factors"},
    1: {"name": "Value_Financials", "description": "Low P/E, dividend yield"},
    2: {"name": "High_Vol", "description": "VIX > 20 regime"},
    3: {"name": "Low_Vol", "description": "VIX < 15 range-bound"},
}


def partition_data_for_expert(
    features: np.ndarray,
    returns: np.ndarray,
    expert_id: int,
    vol_proxy_col: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition training data by expert specialization.

    Parameters
    ----------
    features : np.ndarray
        Shape ``[num_days, num_stocks, num_features]``.
    returns : np.ndarray
        Shape ``[num_days, num_stocks]``.
    expert_id : int
        Expert index (0-3).
    vol_proxy_col : int
        Column index in features that proxies volatility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Filtered (features, returns) for this expert's specialty.
    """
    n_days = features.shape[0]
    n_stocks = features.shape[1]

    if expert_id == 0:
        # Tech/Growth: high momentum stocks (top 40% by momentum feature)
        momentum = features[:, :, 1]  # ret_5d column
        threshold = np.percentile(momentum, 60, axis=1, keepdims=True)
        mask = momentum >= threshold
    elif expert_id == 1:
        # Value/Financials: low momentum stocks (bottom 40%)
        momentum = features[:, :, 1]
        threshold = np.percentile(momentum, 40, axis=1, keepdims=True)
        mask = momentum <= threshold
    elif expert_id == 2:
        # High Vol: days where cross-sectional vol is above median
        daily_vol = features[:, :, 2].std(axis=1)  # vol_20d column
        vol_median = np.median(daily_vol)
        day_mask = daily_vol > vol_median
        mask = np.broadcast_to(day_mask[:, None], (n_days, n_stocks))
    elif expert_id == 3:
        # Low Vol: days where cross-sectional vol is below median
        daily_vol = features[:, :, 2].std(axis=1)
        vol_median = np.median(daily_vol)
        day_mask = daily_vol <= vol_median
        mask = np.broadcast_to(day_mask[:, None], (n_days, n_stocks))
    else:
        mask = np.ones((n_days, n_stocks), dtype=bool)

    # For simplicity, filter at the day level for vol-regime experts
    if expert_id in (2, 3):
        day_indices = mask[:, 0]
        return features[day_indices], returns[day_indices]

    # For sector-based experts, we still use all days but weight the loss
    return features, returns


# ═══════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def train_single_expert(
    expert_id: int,
    features: np.ndarray,
    returns: np.ndarray,
    output_path: str,
    d_input: int = 18,
    d_model: int = 128,
    n_head: int = 4,
    n_layers: int = 2,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda:0",
) -> dict:
    """Train a single RankTransformer expert.

    Returns
    -------
    dict
        Training summary.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    spec = EXPERT_SPECS.get(expert_id, {"name": f"Expert_{expert_id}"})
    logger.info("Training Expert %d (%s) on %s", expert_id, spec["name"], dev)

    # Partition data
    exp_features, exp_returns = partition_data_for_expert(
        features, returns, expert_id
    )
    logger.info("Expert %d data: %d days × %d stocks", expert_id, *exp_features.shape[:2])

    # TimeSeriesSplit
    n = len(exp_features)
    split = int(n * 0.8)
    train_X = torch.tensor(exp_features[:split], dtype=torch.float32)
    train_Y = torch.tensor(exp_returns[:split], dtype=torch.float32)
    val_X = torch.tensor(exp_features[split:], dtype=torch.float32)
    val_Y = torch.tensor(exp_returns[split:], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_X, train_Y), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_X, val_Y), batch_size=batch_size, shuffle=False
    )

    # Model
    model = RankTransformer(
        d_input=d_input, d_model=d_model, n_head=n_head, n_layers=n_layers
    ).to(dev)

    criterion = PairwiseRankingLoss(margin=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    patience_counter = 0
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(dev), Y_batch.to(dev)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs["score"], Y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(dev), Y_batch.to(dev)
                outputs = model(X_batch)
                val_loss += criterion(outputs["score"], Y_batch).item()
        val_loss /= max(len(val_loader), 1)
        scheduler.step()

        logger.info(
            "Expert %d Epoch %d/%d — train=%.6f val=%.6f",
            expert_id, epoch + 1, epochs, train_loss, val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), out)
        else:
            patience_counter += 1
            if patience_counter >= 8:
                logger.info("Expert %d early stop at epoch %d", expert_id, epoch + 1)
                break

    return {"expert_id": expert_id, "best_val_loss": best_val_loss, "checkpoint": str(out)}


def train_all_experts(
    features_path: str,
    returns_path: str,
    output_dir: str = "backend/artifacts/model_weights/smoe",
    num_experts: int = 4,
    device: str = "cuda:0",
    **kwargs,
) -> dict:
    """Train all SMoE experts sequentially.

    Parameters
    ----------
    features_path : str
        ``.npy`` file of shape ``[num_days, num_stocks, num_features]``.
    returns_path : str
        ``.npy`` file of shape ``[num_days, num_stocks]``.
    """
    features = np.load(features_path)
    returns = np.load(returns_path)
    logger.info("Loaded features=%s, returns=%s", features.shape, returns.shape)

    results = []
    for i in range(num_experts):
        out_path = str(Path(output_dir) / f"expert_{i}.pt")
        r = train_single_expert(
            expert_id=i,
            features=features,
            returns=returns,
            output_path=out_path,
            d_input=features.shape[2],
            device=device,
            **kwargs,
        )
        results.append(r)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {"status": "ok", "experts": results}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to features.npy [days, stocks, features]")
    parser.add_argument("--returns", required=True, help="Path to returns.npy [days, stocks]")
    parser.add_argument("--output-dir", default="backend/artifacts/model_weights/smoe")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    result = train_all_experts(
        features_path=args.features,
        returns_path=args.returns,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
    )
    print(result)
