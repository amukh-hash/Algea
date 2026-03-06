"""
SelectorDataset — cross-sectional day-level dataset for the rank selector.

One sample = one trading day.  Each item contains feature vectors for all
tickers tradable on that date, plus regression targets.

Supports:
    - Priors-based features from ``selector_schema.MODEL_FEATURE_COLS``
    - Legacy feature columns (backward compatible)
    - Time-based splits via ``make_time_split``
    - Padding / masking via ``selector_collate_fn``
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SelectorDataset(Dataset):
    """
    One sample = one trading day.

    Each item is a dict containing::

        {X: [N, F], y_ret: [N], y_vol: [N], w: [N], mask, date, symbols}

    Parameters
    ----------
    features_df : DataFrame from ``build_priors_frame`` or loaded from parquet
    date_range : optional ``(start, end)`` date filter
    feature_cols : feature column names (default: MODEL_FEATURE_COLS)
    """

    # Default: use the priors-based z-scored + agreement features
    DEFAULT_FEATURE_COLS: List[str] = []  # set at import from schema

    def __init__(
        self,
        features_df: pd.DataFrame,
        date_range: Optional[Tuple[str, str]] = None,
        feature_cols: Optional[List[str]] = None,
        decision_frequency: int = 1,
    ) -> None:
        # Lazy import to avoid circular deps
        if not self.DEFAULT_FEATURE_COLS:
            from algae.data.priors.selector_schema import MODEL_FEATURE_COLS
            SelectorDataset.DEFAULT_FEATURE_COLS = list(MODEL_FEATURE_COLS)

        df = features_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

        if date_range:
            start, end = date_range
            df = df[(df["date"] >= start) & (df["date"] <= end)].copy()

        # Ensure weight column
        if "weight" not in df.columns:
            df["weight"] = 1.0
        df = df[df["weight"] > 0].copy()

        # Normalise symbol column
        if "ticker" in df.columns and "symbol" not in df.columns:
            df = df.rename(columns={"ticker": "symbol"})

        # Ensure target columns exist
        if "y_ret" not in df.columns:
            # Backward compat: try y_rank → y_ret
            if "y_rank" in df.columns:
                df["y_ret"] = df["y_rank"]
            else:
                df["y_ret"] = 0.0
        if "y_vol" not in df.columns:
            df["y_vol"] = 0.0

        # Legacy support
        if "y_rank" not in df.columns:
            df["y_rank"] = df["y_ret"]
        if "y_trade" not in df.columns:
            df["y_trade"] = (df["y_ret"] > 0).astype(float)
        if "tier" not in df.columns:
            df["tier"] = 0

        self.features_df = df
        self.dates = sorted(df["date"].unique())

        # Part 2A: decision-date sampling — only train on every Nth date
        if decision_frequency > 1:
            self.dates = self.dates[::decision_frequency]

        self.feature_cols = feature_cols or list(self.DEFAULT_FEATURE_COLS)

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        date = self.dates[idx]
        day_df = self.features_df[self.features_df["date"] == date]

        X = day_df[self.feature_cols].values.astype(np.float32)
        y_ret = day_df["y_ret"].fillna(0.0).values.astype(np.float32)
        y_vol = day_df["y_vol"].fillna(0.0).values.astype(np.float32)
        w = day_df["weight"].values.astype(np.float32)

        # Backward compat
        y_rank = day_df["y_rank"].fillna(0.0).values.astype(np.float32)
        y_trade = day_df["y_trade"].fillna(0).values.astype(np.float32)
        tiers = day_df["tier"].fillna(-1).values.astype(np.int64)

        return {
            "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
            "symbols": day_df["symbol"].values,
            "X": torch.tensor(X),
            "y_ret": torch.tensor(y_ret),
            "y_vol": torch.tensor(y_vol),
            "y_rank": torch.tensor(y_rank),
            "y_trade": torch.tensor(y_trade),
            "w": torch.tensor(w),
            "tiers": torch.tensor(tiers),
        }


# ---------------------------------------------------------------------------
# Collate — pads to max sequence length in batch
# ---------------------------------------------------------------------------

def _pad_tensor(batch: List[Dict], key: str, B: int, max_len: int,
                fill_value: float = 0.0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a padded tensor for a given key across the batch."""
    if batch[0][key].dim() == 2:
        F = batch[0][key].shape[1]
        out = torch.full((B, max_len, F), fill_value, dtype=dtype)
    else:
        out = torch.full((B, max_len), fill_value, dtype=dtype)
    for i, b in enumerate(batch):
        L = b[key].shape[0]
        out[i, :L] = b[key]
    return out


def selector_collate_fn(batch: List[Dict]) -> Dict[str, object]:
    """Pad cross-sectional batches to ``N_max`` and produce a boolean mask."""
    max_len = max(b["X"].shape[0] for b in batch)
    B = len(batch)

    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, b in enumerate(batch):
        mask[i, : b["X"].shape[0]] = True

    result = {
        "X": _pad_tensor(batch, "X", B, max_len),
        "y_ret": _pad_tensor(batch, "y_ret", B, max_len),
        "y_vol": _pad_tensor(batch, "y_vol", B, max_len),
        "y_rank": _pad_tensor(batch, "y_rank", B, max_len),
        "y_trade": _pad_tensor(batch, "y_trade", B, max_len),
        "w": _pad_tensor(batch, "w", B, max_len),
        "tiers": _pad_tensor(batch, "tiers", B, max_len, fill_value=-1, dtype=torch.long),
        "mask": mask,
        "dates": [b["date"] for b in batch],
        "symbols": [b["symbols"] for b in batch],
    }
    return result


# ---------------------------------------------------------------------------
# Time-based split helper
# ---------------------------------------------------------------------------

def make_time_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a priors-frame DataFrame by date boundaries.

    Parameters
    ----------
    df : DataFrame with a ``date`` column
    train_end : str
        Last date (inclusive) of the training period, e.g. "2021-12-31"
    val_end : str
        Last date (inclusive) of the validation period, e.g. "2023-12-31"

    Returns
    -------
    (train_df, val_df, test_df) — disjoint by date
    """
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)

    train_df = df[df["date"] <= train_end_dt].copy()
    val_df = df[(df["date"] > train_end_dt) & (df["date"] <= val_end_dt)].copy()
    test_df = df[df["date"] > val_end_dt].copy()

    return train_df, val_df, test_df
