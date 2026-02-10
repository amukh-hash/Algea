"""
Data-loading utilities — iterators, collate functions, and train/test splitters.

The module also re-exports collate functions from:
  - ``algaie.training.chronos_dataset.chronos_collate_fn``
  - ``algaie.training.selector_dataset.selector_collate_fn``
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Basic helpers (original stubs)
# ---------------------------------------------------------------------------

def iter_batches(df: pd.DataFrame, batch_size: int) -> Iterator[pd.DataFrame]:
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size]


def split_train_test(df: pd.DataFrame, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_ts = pd.Timestamp(split_date)
    train = df[df["date"] <= split_ts].copy()
    test = df[df["date"] > split_ts].copy()
    return train, test


# ---------------------------------------------------------------------------
# Gold dataset collate
# ---------------------------------------------------------------------------

def _stack_field(batch: List[Dict[str, np.ndarray]], key: str) -> torch.Tensor:
    """Stack a single field from a batch of dicts into a float32 tensor."""
    return torch.tensor(np.stack([b[key] for b in batch]), dtype=torch.float32)


def gold_collate_fn(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """Stack per-sample dicts from ``GoldFuturesWindowDataset`` into batched tensors."""
    out: Dict[str, torch.Tensor] = {
        "x_float": _stack_field(batch, "x_float"),
        "y_float": _stack_field(batch, "y_float"),
    }
    for key in ("future_target_1d", "target_10d"):
        if batch[0].get(key) is not None:
            out[key] = _stack_field(batch, key)
    return out


# ---------------------------------------------------------------------------
# Re-exports for convenience
# ---------------------------------------------------------------------------

from algaie.training.chronos_dataset import chronos_collate_fn  # noqa: E402,F401
from algaie.training.selector_dataset import selector_collate_fn  # noqa: E402,F401

__all__ = [
    "iter_batches",
    "split_train_test",
    "gold_collate_fn",
    "chronos_collate_fn",
    "selector_collate_fn",
]
