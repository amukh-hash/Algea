"""
Shared feature-engineering utilities for the selector pipeline.

Used by both ``build_priors_frame.py`` (offline training) and
``selector_inference.py`` (live).  By defining derivation and
z-scoring here exactly once, we guarantee feature parity.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from algaie.data.priors.selector_schema import (
    AGREEMENT_COLS,
    DERIVED_COLS,
    ZSCORE_SOURCE_COLS,
    Z_FEATURE_COLS,
)


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

    Returns
    -------
    DataFrame with ``z_{col}`` columns appended.
    """
    if cols is None:
        cols = list(ZSCORE_SOURCE_COLS)

    for col in cols:
        z_col = f"z_{col}"
        mu = df.groupby("date")[col].transform("mean")
        sigma = df.groupby("date")[col].transform("std")
        sigma = sigma.replace(0, np.nan)
        z = (df[col] - mu) / sigma
        z = z.clip(-clamp, clamp).fillna(0.0)
        df[z_col] = z

    return df


def compute_agreement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-horizon agreement features from z-scored values.

    Required z-columns: ``z_drift_10``, ``z_drift_30``,
    ``z_tail_risk_10``, ``z_tail_risk_30``, ``z_iqr_30``.
    """
    df["z_drift_gap"] = df["z_drift_10"] - df["z_drift_30"]
    df["z_tail_gap"] = df["z_tail_risk_10"] - df["z_tail_risk_30"]
    df["z_regime_risk"] = df["z_iqr_30"]
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
