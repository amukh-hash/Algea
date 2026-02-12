"""
Selector evaluation utilities: bucketed metrics, priors-only baseline,
risk-adjusted scoring, and target alignment diagnostics.

All functions operate on DataFrames with columns:
    date, symbol, score_col, target_col, and optionally z-feature columns.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Core per-date metrics (pure numpy, no torch dependency)
# ═══════════════════════════════════════════════════════════════════════════

def _spearman_ic(scores: np.ndarray, targets: np.ndarray) -> float:
    finite = np.isfinite(scores) & np.isfinite(targets)
    if finite.sum() < 3:
        return np.nan
    s, t = scores[finite], targets[finite]
    if np.std(s) < 1e-12 or np.std(t) < 1e-12:
        return np.nan
    ic, _ = spearmanr(s, t)
    return float(ic) if np.isfinite(ic) else np.nan


def _ndcg_at_k(scores: np.ndarray, targets: np.ndarray, k: int = 50) -> float:
    n = len(scores)
    if n < 2:
        return np.nan
    k = min(k, n)
    rel = targets - targets.min() + 1e-8
    pred_order = np.argsort(-scores)
    dcg = sum(rel[pred_order[i]] / np.log2(i + 2) for i in range(k))
    ideal_order = np.argsort(-rel)
    idcg = sum(rel[ideal_order[i]] / np.log2(i + 2) for i in range(k))
    return float(dcg / max(idcg, 1e-12))


def _decile_spread(scores: np.ndarray, targets: np.ndarray) -> float:
    n = len(scores)
    if n < 10:
        return np.nan
    k = max(1, n // 10)
    order = np.argsort(-scores)
    return float(targets[order[:k]].mean() - targets[order[-k:]].mean())


def _topk_return(scores: np.ndarray, targets: np.ndarray, k: int = 50) -> float:
    k = min(k, len(scores))
    if k < 1:
        return np.nan
    order = np.argsort(-scores)
    return float(targets[order[:k]].mean())


def per_date_metrics(
    df: pd.DataFrame,
    score_col: str = "score",
    target_col: str = "y_ret",
    k: int = 50,
) -> pd.DataFrame:
    """Compute IC, NDCG@K, spread, topK return per date.

    Returns DataFrame with columns: date, ic, ndcg, spread, topk_return, n_tickers.
    """
    rows = []
    for dt, grp in df.groupby("date"):
        s = grp[score_col].values.astype(float)
        t = grp[target_col].values.astype(float)
        rows.append({
            "date": dt,
            "ic": _spearman_ic(s, t),
            "ndcg": _ndcg_at_k(s, t, k),
            "spread": _decile_spread(s, t),
            "topk_return": _topk_return(s, t, k),
            "n_tickers": len(grp),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# A) Bucketed metrics (monthly / weekly)
# ═══════════════════════════════════════════════════════════════════════════

def compute_bucketed_metrics(
    df: pd.DataFrame,
    score_col: str = "score",
    target_col: str = "y_ret",
    bucket: str = "month",
    k: int = 50,
) -> pd.DataFrame:
    """Compute metrics aggregated by time bucket.

    Parameters
    ----------
    df : DataFrame with date, symbol, score_col, target_col
    bucket : "month" -> YYYY-MM, "week" -> YYYY-Www
    k : top-K for NDCG and topk_return

    Returns
    -------
    DataFrame with columns:
        bucket, ic_pooled, ic_by_date, ndcg, spread, topk_return, n_dates, n_rows
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if bucket == "month":
        df["bucket"] = df["date"].dt.to_period("M").astype(str)
    elif bucket == "week":
        df["bucket"] = df["date"].dt.isocalendar().apply(
            lambda r: f"{r['year']}-W{r['week']:02d}", axis=1
        )
    else:
        raise ValueError(f"Unknown bucket type: {bucket}")

    # Per-date metrics first
    date_metrics = per_date_metrics(df, score_col, target_col, k)
    date_metrics["date"] = pd.to_datetime(date_metrics["date"])
    if bucket == "month":
        date_metrics["bucket"] = date_metrics["date"].dt.to_period("M").astype(str)
    else:
        date_metrics["bucket"] = date_metrics["date"].dt.isocalendar().apply(
            lambda r: f"{r['year']}-W{r['week']:02d}", axis=1
        )

    rows = []
    for bkt, bgrp in date_metrics.groupby("bucket"):
        # Pooled IC: compute over all rows in this bucket
        bucket_rows = df[df["bucket"] == bkt]
        s_all = bucket_rows[score_col].values.astype(float)
        t_all = bucket_rows[target_col].values.astype(float)

        rows.append({
            "bucket": bkt,
            "ic_pooled": _spearman_ic(s_all, t_all),
            "ic_by_date": bgrp["ic"].mean(),
            "ndcg": bgrp["ndcg"].mean(),
            "spread": bgrp["spread"].mean(),
            "topk_return": bgrp["topk_return"].mean(),
            "n_dates": len(bgrp),
            "n_rows": len(bucket_rows),
        })

    return pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# C) Priors-only baseline score
# ═══════════════════════════════════════════════════════════════════════════

def compute_priors_baseline_score(
    df: pd.DataFrame,
    drift_col: str = "z_drift_10",
    tail_col: str = "z_tail_risk_30",
    regime_col: str = "z_regime_risk",
) -> pd.Series:
    """Priors-only baseline: score = z_drift_10 - 0.5*z_tail_risk_30 - 0.5*z_regime_risk.

    Falls back to z_iqr_30 if z_regime_risk is missing.
    """
    d = df[drift_col].fillna(0)
    t = df[tail_col].fillna(0) if tail_col in df.columns else 0
    # z_regime_risk is an alias for z_iqr_30
    if regime_col in df.columns:
        r = df[regime_col].fillna(0)
    elif "z_iqr_30" in df.columns:
        r = df["z_iqr_30"].fillna(0)
    else:
        r = 0
    return d - 0.5 * t - 0.5 * r


# ═══════════════════════════════════════════════════════════════════════════
# D) Risk-adjusted score (post-processing)
# ═══════════════════════════════════════════════════════════════════════════

def apply_risk_adjustment(
    df: pd.DataFrame,
    score_col: str = "score",
    lambda_risk: float = 0.5,
    mu_tail: float = 0.5,
    regime_col: str = "z_regime_risk",
    tail_col: str = "z_tail_risk_30",
) -> pd.Series:
    """Post-hoc risk adjustment: score_adj = score - λ·z_iqr_30 - μ·z_tail_risk_30.

    Uses z_regime_risk (alias for z_iqr_30) and z_tail_risk_30.
    """
    score = df[score_col].values.astype(float)

    # Get regime risk (z_iqr_30 or z_regime_risk)
    if regime_col in df.columns:
        risk = df[regime_col].fillna(0).values.astype(float)
    elif "z_iqr_30" in df.columns:
        risk = df["z_iqr_30"].fillna(0).values.astype(float)
    else:
        risk = np.zeros_like(score)

    if tail_col in df.columns:
        tail = df[tail_col].fillna(0).values.astype(float)
    else:
        tail = np.zeros_like(score)

    return pd.Series(score - lambda_risk * risk - mu_tail * tail, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════
# E) Target alignment diagnostic
# ═══════════════════════════════════════════════════════════════════════════

def diagnose_target_alignment(
    df: pd.DataFrame,
    ticker_data_dir,
    target_col: str = "y_ret",
    horizon: int = 5,
    n_samples: int = 20,
    tol: float = 0.01,
) -> Dict[str, object]:
    """Spot-check target alignment by recomputing returns from per-ticker data.

    Verifies y_ret matches log(close[t+H]/close[t]) for a random sample of
    (date, symbol) pairs.

    Returns dict with keys: n_checked, n_mismatches, max_abs_error, mismatches.
    """
    from pathlib import Path
    ticker_data_dir = Path(ticker_data_dir)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Sample random rows
    n_samples = min(n_samples, len(df))
    sample = df.sample(n=n_samples, random_state=42)

    mismatches = []
    n_checked = 0

    for _, row in sample.iterrows():
        sym = row["symbol"]
        dt = row["date"]
        expected = row[target_col]

        fpath = ticker_data_dir / f"{sym}.parquet"
        if not fpath.exists():
            continue

        tdf = pd.read_parquet(fpath, columns=["date", "close"])
        tdf["date"] = pd.to_datetime(tdf["date"])
        tdf = tdf.sort_values("date").reset_index(drop=True)

        # Find index of target date
        match_idx = tdf.index[tdf["date"] == dt]
        if len(match_idx) == 0:
            continue
        idx = match_idx[0]

        if idx + horizon >= len(tdf):
            continue

        close_t = tdf.loc[idx, "close"]
        close_th = tdf.loc[idx + horizon, "close"]
        if close_t <= 0 or close_th <= 0:
            continue

        recomputed = np.log(close_th / close_t)
        n_checked += 1
        err = abs(recomputed - expected)

        if err > tol:
            mismatches.append({
                "symbol": sym,
                "date": str(dt.date()),
                "expected": round(expected, 6),
                "recomputed": round(recomputed, 6),
                "abs_error": round(err, 6),
            })

    max_err = max((m["abs_error"] for m in mismatches), default=0.0)
    return {
        "n_checked": n_checked,
        "n_mismatches": len(mismatches),
        "max_abs_error": max_err,
        "mismatches": mismatches,
    }


def diagnose_zscore_universe(
    df: pd.DataFrame,
    n_dates: int = 5,
) -> Dict[str, object]:
    """Check z-score universe consistency: z-scores should have mean~0, std~1
    per date over the joined universe.

    Returns dict with per-date stats.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    z_cols = [c for c in df.columns if c.startswith("z_")]
    if not z_cols:
        return {"error": "No z-score columns found"}

    dates = sorted(df["date"].unique())
    sample_dates = [dates[i] for i in np.linspace(0, len(dates) - 1, n_dates, dtype=int)]

    results = []
    for dt in sample_dates:
        ddf = df[df["date"] == dt]
        row_result = {"date": str(dt)[:10], "n_symbols": len(ddf)}
        for zc in z_cols[:5]:  # spot-check first 5
            vals = ddf[zc].dropna()
            row_result[f"{zc}_mean"] = round(vals.mean(), 4)
            row_result[f"{zc}_std"] = round(vals.std(), 4)
        results.append(row_result)

    # Warn if any mean > 0.1 or std deviates from 1.0 by > 0.3
    warnings = []
    for r in results:
        for k, v in r.items():
            if k.endswith("_mean") and abs(v) > 0.1:
                warnings.append(f"{r['date']} {k}={v}")
            if k.endswith("_std") and abs(v - 1.0) > 0.3:
                warnings.append(f"{r['date']} {k}={v}")

    return {
        "n_dates_checked": len(results),
        "stats": results,
        "warnings": warnings,
    }
