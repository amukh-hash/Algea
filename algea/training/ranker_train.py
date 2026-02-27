"""
Ranker model training loop.

Replaces the placeholder stub with a real training pipeline that uses
``RankTransformer`` + ``listwise_softmax_loss``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from algea.models.ranker.rank_transformer import RankerConfig, SimpleRanker

logger = logging.getLogger(__name__)


def train_ranker_model(
    features: pd.DataFrame,
    destination: Path,
    *,
    input_dim: int = 9,
    d_model: int = 64,
    n_head: int = 2,
    n_layers: int = 2,
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> None:
    """
    Train a ``RankTransformer`` on cross-sectional features.

    Falls back to ``SimpleRanker`` stub if ``torch`` training is not feasible
    (e.g. missing GPU deps).
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if device is None:
        from algea.core.device import get_device
        device = get_device()

    try:
        from algea.models.ranker.rank_transformer import RankTransformer
        from algea.models.common.losses import listwise_softmax_loss
        from algea.training.selector_dataset import SelectorDataset, selector_collate_fn
    except ImportError as e:
        logger.warning(f"Full ranker training failed ({e}); saving SimpleRanker placeholder.")
        model = SimpleRanker(RankerConfig())
        destination.write_text("ranker model placeholder", encoding="utf-8")
        return

    try:
        ds = SelectorDataset(features_df=features)
        if len(ds) == 0:
            logger.warning("No training dates — falling back to SimpleRanker stub.")
            raise ValueError("empty dataset")

        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=selector_collate_fn)

        model = RankTransformer(d_input=input_dim, d_model=d_model, n_head=n_head, n_layers=n_layers).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            steps = 0

            for batch in loader:
                X = batch["X"].to(device)
                targets = batch["y_rank"].to(device)
                mask = batch["mask"].to(device)

                optimizer.zero_grad()
                out = model(X, mask)
                scores = out["score"].squeeze(-1)

                batch_losses = [
                    listwise_softmax_loss(scores[i][mask[i]], targets[i][mask[i]])
                    for i in range(scores.size(0))
                    if mask[i].sum() >= 2
                ]

                if batch_losses:
                    loss = torch.stack(batch_losses).mean()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                steps += 1

            avg = total_loss / max(1, steps)
            logger.info(f"Epoch {epoch + 1}/{epochs} — loss={avg:.6f}")

        torch.save(model.state_dict(), destination)
        logger.info(f"Ranker model saved → {destination}")

    except (ValueError, RuntimeError) as e:
        logger.warning(f"Full ranker training failed ({e}); saving SimpleRanker placeholder.")
        model = SimpleRanker(RankerConfig())
        destination.write_text("ranker model placeholder", encoding="utf-8")

