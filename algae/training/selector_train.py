"""
Selector (TwoHeadRankSelector) training pipeline.

Ported from deprecated/backend_app_snapshot/training/trainer.py
and the selector-specific training patterns.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def train_selector(
    features_df: pd.DataFrame,
    destination: Path,
    *,
    input_dim: int = 5,
    hidden_dim: int = 64,
    dropout: float = 0.1,
    trade_lambda: float = 0.25,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 5e-4,
    device: Optional[torch.device] = None,
) -> None:
    """
    Train a ``TwoHeadRankSelector`` with ``WeightedPairwiseLoss``.

    Parameters
    ----------
    features_df : selector feature frame (``SelectorFeatureBuilder.build()`` output)
    destination : path for the saved state dict
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if device is None:
        from algae.core.device import get_device
        device = get_device()

    from algae.models.ranker.selector_v2 import TwoHeadRankSelector, WeightedPairwiseLoss
    from algae.training.selector_dataset import SelectorDataset, selector_collate_fn

    ds = SelectorDataset(features_df=features_df)
    if len(ds) == 0:
        logger.warning("No training dates — aborting.")
        return

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=selector_collate_fn)

    model = TwoHeadRankSelector(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    criterion = WeightedPairwiseLoss(trade_lambda=trade_lambda).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in loader:
            X = batch["X"].to(device)
            y_rank = batch["y_rank"].to(device)
            y_trade = batch["y_trade"].to(device)
            w = batch["w"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            scores, p_trade = model(X)
            loss, metrics = criterion(scores, p_trade, y_rank, y_trade, w, mask)

            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            steps += 1

        avg = total_loss / max(1, steps)
        logger.info(f"Epoch {epoch + 1}/{epochs} — loss={avg:.6f}")

    torch.save(model.state_dict(), destination)
    logger.info(f"Selector model saved → {destination}")
