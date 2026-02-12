"""
Shared feature-engineering utilities for the selector pipeline.

Used by both ``build_priors_frame.py`` (offline training) and
``selector_inference.py`` (live).  By defining derivation and
z-scoring here exactly once, we guarantee feature parity.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from algaie.data.priors.selector_schema import (
    AGREEMENT_COLS,
    DERIVED_COLS,
    ZSCORE_SOURCE_COLS,
    Z_FEATURE_COLS,
)

logger = logging.getLogger(__name__)

_DEGENERACY_EPS = 1e-12


# ═══════════════════════════════════════════════════════════════════════════
# CS feature specification — single source of truth for sign semantics
# ═══════════════════════════════════════════════════════════════════════════

# cs_tail_30_std is the cross-sectional standard deviation of tail_risk_30
# across all tickers on a given date.  Higher values indicate greater
# dispersion / market stress.  After time z-scoring:
#   positive z_cs → stress above historical norm
#   negative z_cs → calmer than historical norm
#
# With meaning="stress_up" and recommended_cs_sign=+1:
#   gate_input += gamma_cs * (+1) * z_cs
#   → stress↑ → gate_input↑ → w↓ → more baseline  (correct)

CS_FEATURE_SPEC: dict = {
    "name": "cs_tail_30_std",
    "z_col": "z_cs_tail_30_std",
    "meaning": "stress_up",          # higher z = higher market stress
    "recommended_cs_sign": 1.0,      # +1 so stress↑ → gate_input↑ → w↓
}


def make_cs_feature_spec() -> dict:
    """Return a copy of CS_FEATURE_SPEC for manifest serialization."""
    return dict(CS_FEATURE_SPEC)


# ═══════════════════════════════════════════════════════════════════════════
# Derived features
# ═══════════════════════════════════════════════════════════════════════════

def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns (iqr, upside, downside, skew) for each horizon.

    Mutates ``df`` in-place and returns it.

    Required input columns (per horizon h ∈ {10, 30}):
        q10_{h}, q50_{h}, q90_{h}
    """
    for h in ("10", "30"):
        q10 = df[f"q10_{h}"]
        q50 = df[f"q50_{h}"]
        q90 = df[f"q90_{h}"]
        df[f"iqr_{h}"] = q90 - q10
        df[f"upside_{h}"] = q90 - q50
        df[f"downside_{h}"] = q50 - q10
        df[f"skew_{h}"] = q90 + q10 - 2 * q50
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Cross-sectional z-scoring
# ═══════════════════════════════════════════════════════════════════════════

def zscore_cross_sectional(
    df: pd.DataFrame,
    cols: List[str] | None = None,
    clamp: float = 5.0,
    track_degeneracy: bool = True,
) -> pd.DataFrame:
    """z-score selected columns cross-sectionally (within same date).

    Parameters
    ----------
    df : DataFrame
        Must contain a ``date`` column and the columns listed in *cols*.
    cols : list[str] | None
        Columns to z-score.  Defaults to ``ZSCORE_SOURCE_COLS``.
    clamp : float
        Winsorise z-scores to ``[-clamp, +clamp]``.
    track_degeneracy : bool
        If True, add ``deg_{col}`` int8 columns flagging dates where
        the cross-sectional std was ≤ eps (i.e. z-score is meaningless).

    Returns
    -------
    DataFrame with ``z_{col}`` columns appended (and optionally ``deg_{col}``).
    """
    if cols is None:
        cols = list(ZSCORE_SOURCE_COLS)

    for col in cols:
        z_col = f"z_{col}"
        mu = df.groupby("date")[col].transform("mean")
        sigma = df.groupby("date")[col].transform("std")

        degenerate = (sigma <= _DEGENERACY_EPS) | sigma.isna()

        sigma = sigma.replace(0, np.nan)
        z = (df[col] - mu) / sigma
        z = z.clip(-clamp, clamp).fillna(0.0)
        df[z_col] = z

        if track_degeneracy:
            df[f"deg_{col}"] = degenerate.astype(np.int8)

    return df


def compute_agreement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-horizon agreement features from z-scored values.

    Robust ``z_regime_risk`` definition:
    - Uses ``z_iqr_30`` when it is non-degenerate (has cross-sectional variance).
    - Falls back to ``z_tail_risk_30`` when ``z_iqr_30`` is degenerate.

    Also adds ``z_regime_uncertainty``:
    - ``|z_drift_gap| + |z_tail_gap|`` — disagreement proxy across horizons.

    Required z-columns: ``z_drift_10``, ``z_drift_30``,
    ``z_tail_risk_10``, ``z_tail_risk_30``, ``z_iqr_30``.
    """
    df["z_drift_gap"] = df["z_drift_10"] - df["z_drift_30"]
    df["z_tail_gap"] = df["z_tail_risk_10"] - df["z_tail_risk_30"]

    # Robust regime risk: fallback from z_iqr_30 to z_tail_risk_30
    if "deg_iqr_30" in df.columns:
        degenerate = df["deg_iqr_30"].astype(bool)
    else:
        # Compute degeneracy inline if flags not present
        sigma = df.groupby("date")["z_iqr_30"].transform("std")
        degenerate = (sigma <= _DEGENERACY_EPS) | sigma.isna()

    df["z_regime_risk"] = np.where(
        degenerate,
        df["z_tail_risk_30"].values,   # fallback
        df["z_iqr_30"].values,          # primary
    )

    # Regime uncertainty: cross-horizon disagreement magnitude
    df["z_regime_uncertainty"] = (
        df["z_drift_gap"].abs() + df["z_tail_gap"].abs()
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Full feature pipeline
# ═══════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline: derived → z-score → agreement.

    Input ``df`` must already contain the 16 raw teacher priors columns
    (suffixed ``_10`` and ``_30``) and a ``date`` column.
    """
    df = compute_derived_features(df)
    df = zscore_cross_sectional(df)
    df = compute_agreement_features(df)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Compute-time fixup for pre-built priors frames
# ═══════════════════════════════════════════════════════════════════════════

def recompute_regime_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute z_regime_risk + z_regime_uncertainty on an already-built frame.

    Use this at training load time to apply the robust fallback logic
    without rebuilding the full priors_frame from scratch.

    Works with frames that already have z_iqr_30, z_tail_risk_30,
    z_drift_10, z_drift_30, z_tail_risk_10 columns.
    """
    # Compute degeneracy for iqr_30 per date
    if "iqr_30" in df.columns:
        sigma_raw = df.groupby("date")["iqr_30"].transform("std")
        df["deg_iqr_30"] = ((sigma_raw <= _DEGENERACY_EPS) | sigma_raw.isna()).astype(np.int8)
    elif "z_iqr_30" in df.columns:
        sigma_z = df.groupby("date")["z_iqr_30"].transform("std")
        df["deg_iqr_30"] = ((sigma_z <= _DEGENERACY_EPS) | sigma_z.isna()).astype(np.int8)
    else:
        df["deg_iqr_30"] = np.int8(1)  # assume degenerate if column missing

    degenerate = df["deg_iqr_30"].astype(bool)

    # Robust regime risk
    z_iqr = df["z_iqr_30"].values if "z_iqr_30" in df.columns else np.zeros(len(df))
    z_tail = df["z_tail_risk_30"].values if "z_tail_risk_30" in df.columns else np.zeros(len(df))
    df["z_regime_risk"] = np.where(degenerate, z_tail, z_iqr)

    # Regime uncertainty
    if "z_drift_gap" not in df.columns:
        if "z_drift_10" in df.columns and "z_drift_30" in df.columns:
            df["z_drift_gap"] = df["z_drift_10"] - df["z_drift_30"]
    if "z_tail_gap" not in df.columns:
        if "z_tail_risk_10" in df.columns and "z_tail_risk_30" in df.columns:
            df["z_tail_gap"] = df["z_tail_risk_10"] - df["z_tail_risk_30"]

    if "z_drift_gap" in df.columns and "z_tail_gap" in df.columns:
        df["z_regime_uncertainty"] = df["z_drift_gap"].abs() + df["z_tail_gap"].abs()
    else:
        df["z_regime_uncertainty"] = 0.0

    n_deg_dates = df.groupby("date")["deg_iqr_30"].first().sum()
    total_dates = df["date"].nunique()
    logger.info(
        f"Regime risk recomputed: {n_deg_dates}/{total_dates} dates degenerate "
        f"(iqr_30 fallback → z_tail_risk_30). "
        f"z_regime_risk global std={df['z_regime_risk'].std():.4f}"
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Date-level regime features (temporal z-scored cross-sectional stats)
# ═══════════════════════════════════════════════════════════════════════════

def compute_date_cross_sectional_stats(
    df: pd.DataFrame,
    risk_col: str = "tail_risk_30",
) -> pd.DataFrame:
    """Compute per-date cross-sectional std and mean of *risk_col*.

    Returns a DataFrame indexed by ``date`` with columns:
    - ``cs_tail_30_std``  (cross-sectional std across tickers)
    - ``cs_tail_30_mean`` (cross-sectional mean across tickers)
    """
    if risk_col not in df.columns:
        logger.warning(f"compute_date_cross_sectional_stats: '{risk_col}' missing")
        dates = df["date"].drop_duplicates().reset_index(drop=True)
        return pd.DataFrame({
            "date": dates,
            "cs_tail_30_std": 0.0,
            "cs_tail_30_mean": 0.0,
        })
    agg = df.groupby("date")[risk_col].agg(["std", "mean"]).reset_index()
    short = risk_col.replace("tail_risk_", "tail_").replace("dispersion_", "disp_")
    agg.columns = ["date", f"cs_{short}_std", f"cs_{short}_mean"]
    return agg


def fit_time_zscore_scaler(
    date_df: pd.DataFrame,
    value_col: str,
    clamp: float = 5.0,
) -> dict:
    """Fit a temporal z-score scaler from per-date scalar values.

    Parameters
    ----------
    date_df : DataFrame with one row per date, containing *value_col*.
    value_col : column to compute mean/std over.
    clamp : max absolute z-value.

    Returns
    -------
    dict with keys: col, mu, sigma, clamp.
    """
    vals = date_df[value_col].dropna()
    mu = float(vals.mean()) if len(vals) > 0 else 0.0
    sigma = float(vals.std()) if len(vals) > 1 else 0.0
    if sigma <= _DEGENERACY_EPS:
        logger.warning(
            f"fit_time_zscore_scaler: sigma≈0 for '{value_col}' "
            f"(n={len(vals)}). Setting sigma=1."
        )
        sigma = 1.0
    return {"col": value_col, "mu": mu, "sigma": sigma, "clamp": clamp}


def apply_time_zscore_scaler(
    date_df: pd.DataFrame,
    value_col: str,
    scaler: dict,
    out_col: str | None = None,
) -> pd.DataFrame:
    """Apply a pre-fitted temporal z-score scaler.

    Parameters
    ----------
    date_df : DataFrame with one row per date, containing *value_col*.
    scaler : dict from ``fit_time_zscore_scaler``.
    out_col : output column name (default: ``z_{value_col}``).

    Returns date_df with *out_col* appended.
    """
    if out_col is None:
        out_col = f"z_{value_col}"
    mu = scaler["mu"]
    sigma = scaler["sigma"]
    clamp = scaler.get("clamp", 5.0)
    if value_col in date_df.columns:
        vals = date_df[value_col]
        date_df[out_col] = ((vals - mu) / sigma).clip(-clamp, clamp).fillna(0.0)
    else:
        date_df[out_col] = 0.0
    return date_df


def add_date_regime_features(
    df: pd.DataFrame,
    risk_col: str = "tail_risk_30",
    clamp: float = 5.0,
    scaler: dict | None = None,
) -> tuple:
    """Add per-date market regime scalars, z-scored across time.

    Computes cross-sectional stats per date, then applies temporal
    z-scoring using a provided scaler (no leakage) or fits one from
    the dates present (legacy/fallback).

    New columns added (broadcast to all tickers on the date):
    - ``z_cs_tail_30_std``  — temporal z-score of cross-sectional std
    - ``z_cs_tail_30_mean`` — temporal z-score of cross-sectional mean

    Parameters
    ----------
    scaler : dict | None
        If provided, uses this scaler (fitted on train dates) for
        temporal normalization. If None, fits from all dates in df
        (leaky — only use when explicitly acceptable).

    Returns
    -------
    (df, scaler_used) : tuple
        The modified DataFrame and the scaler that was used/fitted.
    """
    # Step 1: per-date cross-sectional stats
    date_df = compute_date_cross_sectional_stats(df, risk_col)
    std_col = "cs_tail_30_std"

    # Step 2: fit or reuse scaler
    if scaler is None:
        scaler_used = fit_time_zscore_scaler(date_df, std_col, clamp)
    else:
        scaler_used = scaler

    # Step 3: apply temporal z-score
    date_df = apply_time_zscore_scaler(
        date_df, std_col, scaler_used, out_col="z_cs_tail_30_std",
    )
    # Also z-score the mean (using its own simple scaler, not persisted)
    mean_col = "cs_tail_30_mean"
    if mean_col in date_df.columns:
        mean_scaler = fit_time_zscore_scaler(date_df, mean_col, clamp)
        date_df = apply_time_zscore_scaler(
            date_df, mean_col, mean_scaler, out_col="z_cs_tail_30_mean",
        )

    # Step 4: merge z_cs columns onto per-ticker rows
    z_cols = [c for c in date_df.columns if c.startswith("z_cs_")]
    merge_cols = ["date"] + z_cols
    for c in z_cols:
        if c in df.columns:
            df = df.drop(columns=[c])
    df = df.merge(date_df[merge_cols], on="date", how="left")
    for c in z_cols:
        df[c] = df[c].fillna(0.0)

    # Ensure expected columns exist
    for c in ["z_cs_tail_30_std", "z_cs_tail_30_mean"]:
        if c not in df.columns:
            df[c] = 0.0

    logger.info(
        f"Date regime features: z_cs_tail_30_std std={df['z_cs_tail_30_std'].std():.4f}, "
        f"range=[{df['z_cs_tail_30_std'].min():.2f}, {df['z_cs_tail_30_std'].max():.2f}], "
        f"scaler mu={scaler_used['mu']:.6f} sigma={scaler_used['sigma']:.6f}"
    )

    return df, scaler_used


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def diagnose_cross_sectional_variance(
    df: pd.DataFrame,
    date,
    cols: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute cross-sectional std for specified columns on a single date.

    Parameters
    ----------
    df : DataFrame with 'date' column
    date : target date (str or datetime.date)
    cols : columns to check (defaults to quantile + risk columns)

    Returns
    -------
    Dict mapping col name → cross-sectional std on that date.
    """
    date = pd.Timestamp(date)
    ddf = df[df["date"] == date]

    if len(ddf) == 0:
        return {"error": f"No rows for date {date}"}

    if cols is None:
        cols = [
            "q10_30", "q50_30", "q90_30", "iqr_30",
            "tail_risk_30", "drift_30",
            "z_iqr_30", "z_tail_risk_30", "z_regime_risk",
        ]

    result = {"date": str(date.date()), "n_symbols": len(ddf)}
    for c in cols:
        if c in ddf.columns:
            result[f"{c}_std"] = round(float(ddf[c].std()), 6)
        else:
            result[f"{c}_std"] = None
    return result
