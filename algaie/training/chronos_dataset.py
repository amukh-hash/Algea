"""
ChronosDataset — sliding-window dataset for Chronos-2 foundation model training.

Supports:
  * Sliding-window and random-anchor sampling modes
  * Optional past covariates (market-level features aligned to context window)
  * Universe gating via configurable mask column (is_tradable / is_observable / auto)
  * Optional tier-based sampling balance
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ChronosDataset(Dataset):
    """
    Dataset for Chronos-2 training / fine-tuning.

    Parameters
    ----------
    files : per-ticker parquet files with ``date`` and ``target_col``
    context_len, prediction_len : window sizes
    stride : step between consecutive windows (used in sliding mode)
    obs_lookup : ``{symbol: set_of_dates}`` eligible anchor mask (optional)
    target_col : price column name
    max_samples_per_file : cap per ticker
    seed : RNG seed
    sampling_mode : ``"sliding"`` (strided windows) or ``"random_anchors"``
    covariates_df : pandas DataFrame indexed by date with covariate columns (optional)
    tier_lookup : ``{symbol: int}`` tier assignment for balancing (optional)
    tier_weights : target proportion per tier, e.g. ``{1: 0.4, 2: 0.4, 3: 0.2}``
    """

    def __init__(
        self,
        files: List[Path],
        context_len: int,
        prediction_len: int,
        stride: int = 10,
        obs_lookup: Optional[Dict[str, Set]] = None,
        target_col: str = "close",
        max_samples_per_file: Optional[int] = None,
        seed: int = 42,
        sampling_mode: str = "sliding",
        covariates_df: Optional[pd.DataFrame] = None,
        tier_lookup: Optional[Dict[str, int]] = None,
        tier_weights: Optional[Dict[int, float]] = None,
    ) -> None:
        self.files = files
        self.context_len = context_len
        self.prediction_len = prediction_len
        self.stride = stride
        self.target_col = target_col
        self.obs_lookup = obs_lookup
        self.sampling_mode = sampling_mode
        self.rng = np.random.RandomState(seed)

        # Covariates: build date-indexed lookup
        self.covariates_df = covariates_df
        self._cov_dates: Optional[np.ndarray] = None
        self._cov_values: Optional[np.ndarray] = None
        self._cov_ncols: int = 0
        if covariates_df is not None and not covariates_df.empty:
            cov = covariates_df.copy()
            cov["date"] = pd.to_datetime(cov["date"]).dt.date
            cov = cov.sort_values("date").set_index("date")
            # Forward-fill levels, NaN→0 for returns
            cov = cov.ffill().fillna(0.0)
            self._cov_dates = np.array(cov.index.tolist())
            self._cov_values = cov.values.astype(np.float32)
            self._cov_ncols = self._cov_values.shape[1]

        # Tier balancing
        self.tier_lookup = tier_lookup
        self.tier_weights = tier_weights or {1: 0.4, 2: 0.4, 3: 0.2}

        self.stats: Counter = Counter(n_files_total=len(files))

        # Pre-cache file dates for __getitem__ covariate alignment
        self._file_dates_cache: Dict[int, List] = {}

        effective_max = self._compute_per_file_caps(max_samples_per_file)
        self.index: List[Tuple[int, int]] = self._build_index(effective_max)

    def _compute_per_file_caps(
        self, max_samples: Optional[int]
    ) -> Dict[int, int]:
        """Compute per-file sample caps, optionally adjusted for tier balance."""
        base: Dict[int, int] = {}
        if max_samples is None:
            return base

        if self.tier_lookup is None:
            # uniform cap
            for fi in range(len(self.files)):
                base[fi] = max_samples
            return base

        # Group files by tier
        tier_files: Dict[int, List[int]] = {}
        for fi, fp in enumerate(self.files):
            ticker = fp.stem
            tier = self.tier_lookup.get(ticker, 3)
            tier_files.setdefault(tier, []).append(fi)

        total_budget = max_samples * len(self.files)
        for tier, fis in tier_files.items():
            w = self.tier_weights.get(tier, 0.2)
            tier_budget = total_budget * w
            per_file = max(1, int(tier_budget / max(1, len(fis))))
            for fi in fis:
                base[fi] = per_file

        return base

    # ------------------------------------------------------------------
    def _build_index(
        self, max_samples: Dict[int, int]
    ) -> List[Tuple[int, int]]:
        index: List[Tuple[int, int]] = []

        for fi, fp in enumerate(self.files):
            try:
                ticker = fp.stem
                valid_obs_dates = (
                    self.obs_lookup.get(ticker) if self.obs_lookup else None
                )
                if self.obs_lookup is not None and valid_obs_dates is None:
                    continue

                df = pl.read_parquet(fp, columns=["date", self.target_col])
                if "date" in df.columns:
                    df = df.with_columns(
                        pl.col("date").dt.date().alias("date")
                    )

                dates = df["date"].to_list()
                values = df[self.target_col].to_numpy()
                n = len(df)
                self.stats["n_rows_total"] += n

                max_start = n - (self.context_len + self.prediction_len)
                if max_start <= 0:
                    continue

                # Determine candidate starts based on sampling mode
                if self.sampling_mode == "random_anchors":
                    candidate_starts = list(range(0, max_start + 1))
                else:
                    candidate_starts = list(range(0, max_start, self.stride))

                self.stats["n_windows_potential"] += len(candidate_starts)

                valid_file_starts: List[int] = []
                for s in candidate_starts:
                    anchor_date = dates[s + self.context_len - 1]
                    if (
                        valid_obs_dates is not None
                        and anchor_date not in valid_obs_dates
                    ):
                        self.stats["n_dropped_observable"] += 1
                        continue

                    window_vals = values[
                        s : s + self.context_len + self.prediction_len
                    ]
                    if np.isnan(window_vals).any():
                        self.stats["n_dropped_nan"] += 1
                        continue
                    if (window_vals <= 1e-8).any():
                        self.stats["n_dropped_invalid_price"] += 1
                        continue

                    valid_file_starts.append(s)

                file_cap = max_samples.get(fi, len(valid_file_starts))
                if file_cap and len(valid_file_starts) > file_cap:
                    chosen = self.rng.choice(
                        valid_file_starts, size=file_cap, replace=False
                    )
                    valid_file_starts = chosen.tolist()

                # Cache dates for covariate alignment
                if self._cov_values is not None and valid_file_starts:
                    self._file_dates_cache[fi] = dates

                for s in valid_file_starts:
                    index.append((fi, int(s)))
            except Exception:
                continue

        self.stats["n_final_samples"] = len(index)
        logger.info(
            f"ChronosDataset({self.sampling_mode}): "
            f"{len(index)} samples. Stats: {self.stats}"
        )
        return index

    def __len__(self) -> int:
        return len(self.index)

    def _get_covariates(
        self, fi: int, start_row: int
    ) -> Optional[Dict[str, np.ndarray]]:
        """Retrieve aligned past and future covariates."""
        if self._cov_values is None:
            return None

        # Get dates for this file
        dates = self._file_dates_cache.get(fi)
        if dates is None:
            fp = self.files[fi]
            df = pl.read_parquet(fp, columns=["date"])
            df = df.with_columns(pl.col("date").dt.date().alias("date"))
            dates = df["date"].to_list()
            self._file_dates_cache[fi] = dates

        # Indices
        ctx_end = start_row + self.context_len
        pred_end = ctx_end + self.prediction_len

        # Slices
        past_dates = dates[start_row : ctx_end]
        future_dates = dates[ctx_end : pred_end]

        def _lookup(d_list, n_rows):
            out = np.zeros((n_rows, self._cov_ncols), dtype=np.float32)
            for i, d in enumerate(d_list):
                idx = np.searchsorted(self._cov_dates, d)
                if idx < len(self._cov_dates) and self._cov_dates[idx] == d:
                    out[i] = self._cov_values[idx]
                elif idx > 0:
                    out[i] = self._cov_values[idx - 1]
            return out

        return {
            "past": _lookup(past_dates, self.context_len),
            "future": _lookup(future_dates, self.prediction_len),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fi, start_row = self.index[idx]
        fp = self.files[fi]

        df = pl.read_parquet(fp, columns=[self.target_col])
        window_vals = (
            df[self.target_col]
            .slice(start_row, self.context_len + self.prediction_len)
            .to_numpy()
            .astype(np.float32)
        )

        if len(window_vals) < (self.context_len + self.prediction_len):
            raise ValueError("Window too short")

        # Relative log price (return-space modelling)
        ref_val = window_vals[self.context_len - 1]
        x_trans = np.log(window_vals / ref_val)

        context = x_trans[: self.context_len]
        target = x_trans[self.context_len :]

        result = {
            "past_target": torch.tensor(
                context, dtype=torch.float32
            ).unsqueeze(-1),
            "future_target": torch.tensor(
                target, dtype=torch.float32
            ).unsqueeze(-1),
            "scale": torch.tensor([ref_val], dtype=torch.float32),
        }

        # Past covariates (causality: only context period)
        # Covariates
        covs = self._get_covariates(fi, start_row)
        if covs is not None:
            # Past
            p = np.nan_to_num(covs["past"], nan=0.0, posinf=0.0, neginf=0.0)
            result["past_covariates"] = torch.tensor(p, dtype=torch.float32)
            
            # Future
            f = np.nan_to_num(covs["future"], nan=0.0, posinf=0.0, neginf=0.0)
            result["future_covariates"] = torch.tensor(f, dtype=torch.float32)

        return result


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def chronos_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Stack per-sample tensors into batched tensors for Chronos training."""
    result = {
        "past_target": torch.stack([b["past_target"] for b in batch]),
        "future_target": torch.stack([b["future_target"] for b in batch]),
        "scale": torch.stack([b["scale"] for b in batch]),
    }
    # Stack covariates if present
    for key in ("past_covariates", "future_covariates"):
        if key in batch[0]:
            result[key] = torch.stack([b[key] for b in batch])
    return result
