"""
Live inference contract for the selector model.

``score_universe()`` guarantees train/live feature parity by reusing the
exact same feature pipeline (``feature_utils.build_features``) used
during training.  Supports both MLPSelector and RankTransformer models
via the same dict-based forward API.

Usage
-----
::

    from algea.models.ranker.selector_inference import score_universe

    df = score_universe(
        date=date(2025, 1, 15),
        universe=["AAPL", "MSFT", "GOOG"],
        teacher10=teacher_10d_wrapper,
        teacher30=teacher_30d_wrapper,
        selector_model=selector,  # MLPSelector or RankTransformer
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
    from algea.data.priors.chronos_priors_compute import (
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
    # Blend / gate params (overridden by manifest if provided)
    blend_mode: str = "sigmoid",
    gate_g0: float = 0.0,
    gate_g1: float = 1.0,
    gate_threshold: float = 0.0,
    gate_k: float = 1.0,
    gate_gamma_cs: float = 1.0,
    gate_cs_sign: float = -1.0,
    # Baseline params
    baseline_a: float = 1.0,
    baseline_lambda: float = 0.5,
    baseline_mu: float = 0.5,
    # Manifest for production parity
    blend_manifest_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Score tickers for a given date using teacher priors and the selector.

    Produces three scoring variants:
    - ``score_model``: raw MLP/Transformer output
    - ``score_baseline``: priors-only linear combiner
    - ``score_final``: regime-aware blend of model and baseline

    Guarantees train/live feature parity by reusing
    ``feature_utils.build_features()`` and ``baseline_scorer`` functions.

    If ``blend_manifest_path`` is provided, loads gate parameters and
    the temporal scaler from the manifest for production parity.

    Parameters
    ----------
    date : datetime.date
        The scoring date.
    universe : list[str]
        Tradable symbols on this date.
    teacher10, teacher30 : nn.Module
        Loaded Chronos2NativeWrapper teachers.
    selector_model : nn.Module
        Loaded MLPSelector or RankTransformer selector.
    ticker_data_dir : Path
        Directory with per-ticker parquet files.
    context_len : int
        Number of historical trading days per context.
    priors_cache_root : Path | None
        If provided, attempt to load priors from cache first.
    blend_manifest_path : Path | None
        Path to ``blend_manifest.json`` from training. Loads gate params
        and ``cs_scaler`` for production-parity temporal normalization.
    blend_mode : str
        "sigmoid" or "piecewise" gate strategy.
    gate_g0, gate_g1 : float
        Sigmoid gate bias and slope.
    gate_threshold, gate_k : float
        Piecewise gate threshold and slope.
    baseline_a, baseline_lambda, baseline_mu : float
        Baseline scorer weights.

    Returns
    -------
    pd.DataFrame
        Columns: ``symbol``, ``score_model``, ``score_baseline``,
        ``score_final``, ``risk`` (if risk head present).
        Sorted by ``score_final`` descending.
    """
    import json as _json

    from algea.data.priors.feature_utils import (
        add_date_regime_features,
        build_features,
        compute_date_cross_sectional_stats,
        apply_time_zscore_scaler,
    )
    from algea.models.ranker.baseline_scorer import (
        blend_scores,
        compute_baseline_score,
    )

    # --- Load manifest if provided ----------------------------------------
    cs_scaler = None
    gate_gamma = 0.0  # uncertainty gating weight
    if blend_manifest_path is not None:
        try:
            with open(blend_manifest_path, "r") as f:
                manifest = _json.load(f)
            blend_mode = manifest.get("blend_mode", blend_mode)
            gate_g0 = manifest.get("gate_g0", gate_g0)
            gate_g1 = manifest.get("gate_g1", gate_g1)
            gate_gamma_cs = manifest.get("gate_gamma_cs", gate_gamma_cs)
            gate_cs_sign = manifest.get("gate_cs_sign", gate_cs_sign)
            gate_gamma = manifest.get("gate_gamma_unc", 0.0)
            baseline_a = manifest.get("baseline_a", baseline_a)
            baseline_lambda = manifest.get("baseline_lambda", baseline_lambda)
            baseline_mu = manifest.get("baseline_mu", baseline_mu)
            cs_scaler = manifest.get("cs_scaler", None)
            logger.info(
                f"Loaded blend manifest: g0={gate_g0}, g1={gate_g1}, "
                f"gamma_cs={gate_gamma_cs}, cs_sign={gate_cs_sign}, "
                f"gamma_unc={gate_gamma}"
            )
            cs_spec = manifest.get("cs_feature_spec")
            if cs_spec:
                logger.info(
                    f"  cs_feature={cs_spec.get('z_col')}, "
                    f"meaning={cs_spec.get('meaning')}, "
                    f"gate_cs_sign={gate_cs_sign}, "
                    f"gamma_cs={gate_gamma_cs}"
                )
        except Exception as e:
            logger.warning(f"Failed to load blend manifest: {e}")

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
        return pd.DataFrame(columns=["symbol", "score_final"])

    # --- Join and build features ------------------------------------------
    join_cols = ["date", "symbol"]
    meta_drop = [c for c in df_30.columns
                 if c not in join_cols and not c.endswith("_30")]
    df = df_10.merge(
        df_30.drop(columns=meta_drop, errors="ignore"),
        on=join_cols, how="inner",
    )

    if df.empty:
        return pd.DataFrame(columns=["symbol", "score_final"])

    df = build_features(df)

    # Apply date-level regime features using saved scaler (production parity)
    if cs_scaler is not None:
        df, _ = add_date_regime_features(df, scaler=cs_scaler)
    else:
        # No scaler available — log warning, set z_cs to 0
        logger.warning(
            "No cs_scaler available for temporal normalization. "
            "z_cs_tail_30_std will be 0. Pass blend_manifest_path for "
            "production parity."
        )
        df["z_cs_tail_30_std"] = 0.0
        df["z_cs_tail_30_mean"] = 0.0

    # --- Model inference --------------------------------------------------
    from algea.data.priors.selector_schema import MODEL_FEATURE_COLS

    feature_cols = [c for c in MODEL_FEATURE_COLS if c in df.columns]
    if not feature_cols:
        logger.error("No feature columns found after build_features")
        return pd.DataFrame(columns=["symbol", "score_final"])

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
        "score_model": scores,
    })

    # Copy z-features needed for blending
    z_cols_needed = ["z_drift_10", "z_regime_risk", "z_iqr_30", "z_tail_risk_30",
                     "z_cs_tail_30_std", "z_regime_uncertainty"]
    for c in z_cols_needed:
        if c in df.columns:
            result[c] = df[c].values

    result["score_baseline"] = compute_baseline_score(
        result, a=baseline_a, lam=baseline_lambda, mu=baseline_mu,
    )

    # Blended score
    result["score_final"] = blend_scores(
        result, "score_model", "score_baseline",
        blend_mode=blend_mode,
        g0=gate_g0, g1=gate_g1,
        gate_threshold=gate_threshold, gate_k=gate_k,
        gate_gamma=gate_gamma,
        gate_gamma_cs=gate_gamma_cs,
        gate_cs_sign=gate_cs_sign,
    )

    # Optional risk head
    if "risk" in out:
        risk = out["risk"].squeeze(-1).squeeze(0).cpu().numpy()
        result["risk"] = risk

    # Clean up z-cols from output
    result = result.drop(columns=[c for c in z_cols_needed if c in result.columns],
                         errors="ignore")

    return result.sort_values("score_final", ascending=False).reset_index(drop=True)


