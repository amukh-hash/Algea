"""Panel dataset builder for CS-Transformer training.

Converts a flat ``(trading_day, instrument)`` dataset into daily panel
batches suitable for torch DataLoader:

    X : [B, N, F]  — features per instrument per day
    y : [B, N]     — reversal scores per instrument per day
    mask : [B, N]  — True for instruments present that day
    y_risk : [B, N] — risk target (log(eps + |r_oc|)) when two-head
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    Dataset = object  # type: ignore[misc,assignment]


class PanelDataset(Dataset):
    """Torch Dataset that yields daily cross-sectional panels.

    Parameters
    ----------
    dataset : flat DataFrame indexed by (trading_day, instrument) or with columns.
    features : ordered feature column names.
    label_col : label column name (default ``"y"``).
    min_instruments_per_day : skip days with fewer instruments.
    max_instruments : pad panels to this size (None = infer from data).
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        features: Sequence[str],
        label_col: str = "y",
        min_instruments_per_day: int = 2,
        max_instruments: Optional[int] = None,
        extra_cols: Sequence[str] = (),
        risk_target_transform: str = "log_abs",
        risk_target_eps: float = 1e-6,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("torch is required for PanelDataset")

        self.features = list(features)
        self.label_col = label_col
        self.n_features = len(features)
        self.extra_cols = [c for c in extra_cols if c in dataset.columns]
        self.risk_target_transform = risk_target_transform
        self.risk_target_eps = risk_target_eps

        # Detect r_oc column for risk target
        self._r_oc_col: Optional[str] = None
        for candidate in ("r_oc", "ret_oc"):
            if candidate in self.extra_cols:
                self._r_oc_col = candidate
                break

        # Ensure we have flat columns
        df = dataset.reset_index(drop=False) if dataset.index.names[0] is not None else dataset.copy()
        if "instrument" not in df.columns and "root" in df.columns:
            df["instrument"] = df["root"]

        # Sort deterministically
        df = df.sort_values(["trading_day", "instrument"]).reset_index(drop=True)

        # Group by trading_day, filter
        self._panels: list[tuple] = []
        self._instruments_universe = sorted(df["instrument"].unique())  # global instrument ordering
        self._max_n = max_instruments or len(self._instruments_universe)

        for day, grp in df.groupby("trading_day"):
            if len(grp) < min_instruments_per_day:
                continue
            grp = grp.sort_values("instrument").reset_index(drop=True)
            instruments = grp["instrument"].tolist()

            X = grp[self.features].values.astype(np.float64)
            y = grp[self.label_col].values.astype(np.float64) if self.label_col in grp.columns else np.zeros(len(grp))
            extras = {c: grp[c].values.astype(np.float64) for c in self.extra_cols}

            # Compute y_risk from r_oc (deterministic, leakage-safe)
            y_risk: Optional[np.ndarray] = None
            if self._r_oc_col is not None and self._r_oc_col in extras:
                raw = extras[self._r_oc_col]
                if self.risk_target_transform == "log_abs":
                    y_risk = np.log(self.risk_target_eps + np.abs(raw))
                elif self.risk_target_transform == "raw_abs":
                    y_risk = np.abs(raw)
                else:
                    y_risk = np.log(self.risk_target_eps + np.abs(raw))

            self._panels.append((day, instruments, X, y, extras, y_risk))

    def __len__(self) -> int:
        return len(self._panels)

    def __getitem__(self, idx: int) -> dict:
        day, instruments, X_arr, y_arr, extras, y_risk_arr = self._panels[idx]
        n = len(instruments)
        N = self._max_n

        # Pad to max_instruments
        X_padded = np.zeros((N, self.n_features), dtype=np.float64)
        y_padded = np.zeros(N, dtype=np.float64)
        mask = np.zeros(N, dtype=bool)

        X_padded[:n] = X_arr[:N]
        y_padded[:n] = y_arr[:N]
        mask[:n] = True

        item = {
            "X": torch.tensor(X_padded, dtype=torch.float32),
            "y": torch.tensor(y_padded, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "trading_day": str(day),
            "n_instruments": n,
        }
        # Expose transformed risk target
        if y_risk_arr is not None:
            yr_padded = np.zeros(N, dtype=np.float64)
            yr_padded[:n] = y_risk_arr[:N]
            item["y_risk"] = torch.tensor(yr_padded, dtype=torch.float32)
        # Legacy extra_cols passthrough
        for col_name, col_arr in extras.items():
            padded = np.zeros(N, dtype=np.float64)
            padded[:n] = col_arr[:N]
            item[col_name] = torch.tensor(padded, dtype=torch.float32)
        return item


def panel_collate_fn(batch: list[dict]) -> dict:
    """Stack panel dicts into batched tensors.

    Returns
    -------
    dict with:
        X : [B, N, F]
        y : [B, N]
        mask : [B, N]
        trading_days : list[str]
        n_instruments : [B]
        + any extra_col tensors
    """
    result = {
        "X": torch.stack([b["X"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "trading_days": [b["trading_day"] for b in batch],
        "n_instruments": torch.tensor([b["n_instruments"] for b in batch]),
    }
    # Stack any extra columns (e.g. r_oc for risk target)
    extra_keys = [k for k in batch[0] if k not in result and k not in ("trading_day", "n_instruments")]
    for k in extra_keys:
        if isinstance(batch[0][k], torch.Tensor):
            result[k] = torch.stack([b[k] for b in batch])
    return result
