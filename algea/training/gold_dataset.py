"""
GoldFuturesWindowDataset — sliding-window dataset over per-ticker parquet files.

Ported from deprecated/backend_app_snapshot/training/gold_dataset.py.
Decoupled from ``backend.app`` — universe mask passed as a dict or omitted.
"""
from __future__ import annotations

import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, Subset


class GoldFuturesWindowDataset(Dataset):
    """
    Sliding-window dataset for Chronos / foundation model training.

    Each item is a ``[Context, F]`` context window and ``[Pred, F]`` target
    window, constructed from per-ticker parquet files. Windows are filtered via
    an *observable universe mask* (if supplied), and indexed in temporal order
    for deterministic train/val splitting.

    Parameters
    ----------
    files : sorted list of per-ticker ``.parquet`` files
    required_cols : columns to read from each file (order matters for ``col_map``)
    context, pred : sliding-window sizes
    stride_rows : step size between consecutive windows
    max_windows : max windows per file (excess subsampled)
    seed, cache_size : RNG seed and LRU cache capacity
    target_col : column used for log-return transformation
    obs_lookup : optional ``{symbol: set_of_observable_dates}``
    """

    def __init__(
        self,
        files: List[Path],
        required_cols: Tuple[str, ...],
        context: int,
        pred: int,
        stride_rows: int = 5,
        max_windows: int = 500,
        seed: int = 42,
        cache_size: int = 128,
        target_col: str = "close_adj",
        obs_lookup: Optional[Dict[str, Set]] = None,
    ) -> None:
        self.files = files
        self.required_cols = required_cols
        self.target_col = target_col
        self.col_map = {name: i for i, name in enumerate(required_cols)}
        self.context = context
        self.pred = pred
        self.stride_rows = stride_rows
        self.max_windows = max_windows
        self.rng = np.random.RandomState(seed)
        self._load_file_array = lru_cache(maxsize=cache_size)(self._load_file_array_impl)
        self.obs_lookup = obs_lookup

        self.index: List[Tuple[int, int, int]] = []
        self._build_index()

    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        for fi, fp in enumerate(self.files):
            try:
                ticker = fp.stem
                if self.obs_lookup is not None and ticker not in self.obs_lookup:
                    continue

                df_dates = pl.scan_parquet(fp).select(["date"]).collect()
                n = df_dates.height
                dates = df_dates["date"].to_list()

                max_start = n - (self.context + self.pred)
                if max_start <= 0:
                    continue

                starts = list(range(0, max_start, self.stride_rows))

                valid_starts: List[int] = []
                if self.obs_lookup is not None:
                    valid_dates = self.obs_lookup.get(ticker, set())
                    for s in starts:
                        anchor_idx = s + self.context - 1
                        if anchor_idx < n and dates[anchor_idx] in valid_dates:
                            valid_starts.append(s)
                else:
                    valid_starts = starts

                if len(valid_starts) > self.max_windows:
                    valid_starts = self.rng.choice(valid_starts, size=self.max_windows, replace=False).tolist()

                for s in valid_starts:
                    anchor_ts = 0
                    anchor_idx = s + self.context - 1
                    if 0 <= anchor_idx < len(dates):
                        d = dates[anchor_idx]
                        if hasattr(d, "timestamp"):
                            anchor_ts = int(d.timestamp())
                        else:
                            anchor_ts = int(datetime.datetime(d.year, d.month, d.day).timestamp())
                    self.index.append((fi, int(s), anchor_ts))
            except (OSError, pl.exceptions.ComputeError):
                continue

        try:
            self.index.sort(key=lambda x: x[2])
        except TypeError:
            self.rng.shuffle(self.index)

    # ------------------------------------------------------------------
    def split_validation(self, split_pct: float = 0.1) -> Tuple[Subset, Subset]:
        total = len(self.index)
        val_size = int(total * split_pct)
        train_size = total - val_size
        return Subset(self, list(range(train_size))), Subset(self, list(range(train_size, total)))

    def __len__(self) -> int:
        return len(self.index)

    def _load_file_array_impl(self, fi: int) -> np.ndarray:
        """Load and cache per-file numpy array (wrapped by lru_cache)."""
        fp = self.files[fi]
        schema = pl.scan_parquet(fp).schema
        actual_cols = [c for c in self.required_cols if c in schema]
        df = pl.scan_parquet(fp).select(actual_cols).collect()
        return df.to_numpy().astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        fi, s, _ = self.index[idx]

        try:
            arr = self._load_file_array(fi)
        except (OSError, pl.exceptions.ComputeError):
            dummy = np.zeros((self.context, len(self.required_cols)), dtype=np.float32)
            return {"x_float": dummy, "y_float": np.zeros((self.pred, len(self.required_cols)), dtype=np.float32)}

        # Locate columns to exclude (date / timestamp)
        ts_idx = self.col_map.get("timestamp")
        date_idx = self.col_map.get("date")

        end_row = min(s + self.context + self.pred, len(arr))
        subset = arr[s:end_row]

        exclude = set()
        if ts_idx is not None:
            exclude.add(ts_idx)
        if date_idx is not None:
            exclude.add(date_idx)

        if exclude:
            feat_indices = [i for i in range(arr.shape[1]) if i not in exclude]
            feats = subset[:, feat_indices].astype(np.float32)
            filtered_cols = [self.required_cols[i] for i in feat_indices]
        else:
            feats = subset.astype(np.float32)
            filtered_cols = list(self.required_cols)

        try:
            target_feat_idx = filtered_cols.index(self.target_col)
        except ValueError:
            target_feat_idx = -1

        # Pad if short
        expected = self.context + self.pred
        if len(feats) < expected:
            pad_len = expected - len(feats)
            feats = np.pad(feats, ((0, pad_len), (0, 0)))

        # Log-return transform relative to end-of-context reference price
        if target_feat_idx >= 0:
            ref_val = feats[self.context - 1, target_feat_idx]
            if ref_val > 1e-6:
                feats[:, target_feat_idx] = np.log(feats[:, target_feat_idx] / ref_val)
            else:
                feats[:, target_feat_idx] = 0.0

        future_target_1d = None
        target_10d = None
        if target_feat_idx >= 0:
            future_target_1d = feats[self.context : self.context + self.pred, target_feat_idx].astype(np.float32)
            if self.target_col.startswith("ret") and len(future_target_1d) >= 10:
                if not (np.any(future_target_1d[:10] < -0.5) or np.any(future_target_1d[:10] > 0.5)):
                    if np.all(1 + future_target_1d[:10] > 0):
                        target_10d = float(np.prod(1 + future_target_1d[:10]) - 1)

        return {
            "x_float": feats[: self.context],
            "y_float": feats[self.context :],
            "future_target_1d": future_target_1d,
            "target_10d": target_10d,
        }
