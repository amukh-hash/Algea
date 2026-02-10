"""
Live inference contract for the selector model.

``score_universe()`` guarantees train/live feature parity by reusing the
exact same feature pipeline (``feature_utils.build_features``) used
during training.  Supports cache-first priors loading.

Usage
-----
::

    from algaie.models.ranker.selector_inference import score_universe

    df = score_universe(
        date=date(2025, 1, 15),
        universe=["AAPL", "MSFT", "GOOG"],
        teacher10=teacher_10d_wrapper,
        teacher30=teacher_30d_wrapper,
        selector_model=selector,
        ticker_data_dir=Path("backend/data/canonical/per_ticker"),
    )
    # → DataFrame with columns: symbol, score, risk
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _load_ticker_closes(
    symbol: str,
    target_date: datetime.date,
    context_len: int,
    ticker_data_dir: Path,
) -> Optional[torch.Tensor]:
    """Load last ``context_len`` closes up to ``target_date``."""
    fpath = ticker_data_dir / f"{symbol}.parquet"
    if not fpath.exists():
        return None
    df = pd.read_parquet(fpath, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["date"] <= target_date].sort_values("date")
    if len(df) < context_len:
        return None
    closes = df["close"].iloc[-context_len:].values.astype(np.float32)
    if np.any(~np.isfinite(closes)) or np.any(closes <= 0):
        return None
    return torch.tensor(closes, dtype=torch.float32)


def _load_priors_from_cache(
    target_date: datetime.date,
    horizon_tag: str,
    priors_cache_root: Path,
) -> Optional[pd.DataFrame]:
    """Attempt to load cached priors for a date and teacher horizon."""
    date_str = target_date.isoformat()
    part_path = priors_cache_root / f"teacher={horizon_tag}" / f"date={date_str}" / "part.parquet"
    if part_path.exists():
        return pd.read_parquet(part_path)
    return None


def _compute_priors_for_universe(
    teacher: nn.Module,
    symbols: List[str],
    target_date: datetime.date,
    horizon_days: int,
    context_len: int,
    ticker_data_dir: Path,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Compute priors using the canonical function, returning a DataFrame."""
    from algaie.data.priors.chronos_priors_compute import (
        compute_teacher_priors,
        priors_to_row,
    )

    h_suffix = str(horizon_days)
    contexts: Dict[str, torch.Tensor] = {}

    for sym in symbols:
        ctx = _load_ticker_closes(sym, target_date, context_len, ticker_data_dir)
        if ctx is not None:
            contexts[sym] = ctx

    if not contexts:
        return pd.DataFrame()

    sym_list = sorted(contexts.keys())
    rows: List[Dict] = []

    for i in range(0, len(sym_list), batch_size):
        batch_syms = sym_list[i : i + batch_size]
        batch_tensor = torch.stack(
            [contexts[s].unsqueeze(-1) for s in batch_syms]
        )  # [B, T, 1]

        try:
            priors_list = compute_teacher_priors(
                teacher=teacher,
                input_tensor=batch_tensor,
                horizon_days=horizon_days,
                strict=False,
                mode="native_nll",
            )
            for sym, p in zip(batch_syms, priors_list):
                row = priors_to_row(p, h_suffix)
                row["date"] = target_date
                row["symbol"] = sym
                rows.append(row)
        except Exception as e:
            logger.warning(f"Priors failed for batch starting {batch_syms[0]}: {e}")

    return pd.DataFrame(rows)


@torch.no_grad()
def score_universe(
    date: datetime.date,
    universe: List[str],
    teacher10: nn.Module,
    teacher30: nn.Module,
    selector_model: nn.Module,
    ticker_data_dir: Path,
    context_len: int = 252,
    priors_cache_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Score tickers for a given date using teacher priors and the selector.

    Guarantees train/live feature parity by reusing
    ``feature_utils.build_features()``.

    Parameters
    ----------
    date : datetime.date
        The scoring date.
    universe : list[str]
        Tradable symbols on this date.
    teacher10, teacher30 : nn.Module
        Loaded Chronos2NativeWrapper teachers.
    selector_model : nn.Module
        Loaded RankTransformer selector.
    ticker_data_dir : Path
        Directory with per-ticker parquet files.
    context_len : int
        Number of historical trading days per context.
    priors_cache_root : Path | None
        If provided, attempt to load priors from cache first.

    Returns
    -------
    pd.DataFrame
        Columns: ``symbol``, ``score``, ``risk`` (if risk head present).
        Sorted by score descending.
    """
    from algaie.data.priors.feature_utils import build_features

    symbols = sorted(universe)

    # --- Load or compute priors -------------------------------------------
    df_10 = None
    df_30 = None

    if priors_cache_root is not None:
        df_10 = _load_priors_from_cache(date, "10d", priors_cache_root)
        df_30 = _load_priors_from_cache(date, "30d", priors_cache_root)

    if df_10 is None:
        logger.info(f"Computing 10d priors for {len(symbols)} symbols")
        df_10 = _compute_priors_for_universe(
            teacher10, symbols, date, 10, context_len, ticker_data_dir)

    if df_30 is None:
        logger.info(f"Computing 30d priors for {len(symbols)} symbols")
        df_30 = _compute_priors_for_universe(
            teacher30, symbols, date, 30, context_len, ticker_data_dir)

    if df_10.empty or df_30.empty:
        logger.warning("No priors available")
        return pd.DataFrame(columns=["symbol", "score"])

    # --- Join and build features ------------------------------------------
    join_cols = ["date", "symbol"]
    # Drop non-feature metadata from 30d to avoid merge conflicts
    meta_drop = [c for c in df_30.columns
                 if c not in join_cols and not c.endswith("_30")]
    df = df_10.merge(
        df_30.drop(columns=meta_drop, errors="ignore"),
        on=join_cols, how="inner",
    )

    if df.empty:
        return pd.DataFrame(columns=["symbol", "score"])

    df = build_features(df)

    # --- Model inference --------------------------------------------------
    from algaie.data.priors.selector_schema import MODEL_FEATURE_COLS

    feature_cols = [c for c in MODEL_FEATURE_COLS if c in df.columns]
    if not feature_cols:
        logger.error("No feature columns found after build_features")
        return pd.DataFrame(columns=["symbol", "score"])

    X = torch.tensor(
        df[feature_cols].values.astype(np.float32)
    ).unsqueeze(0)  # [1, N, F]

    device = next(selector_model.parameters()).device
    X = X.to(device)

    selector_model.eval()
    out = selector_model(X)
    scores = out["score"].squeeze(-1).squeeze(0).cpu().numpy()  # [N]

    result = pd.DataFrame({
        "symbol": df["symbol"].values,
        "score": scores,
    })

    # Optional risk head
    if "risk" in out:
        risk = out["risk"].squeeze(-1).squeeze(0).cpu().numpy()
        result["risk"] = risk

    return result.sort_values("score", ascending=False).reset_index(drop=True)
