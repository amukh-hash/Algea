"""
Preflight auditor for the selector training pipeline.

Run BEFORE training to fail fast on data integrity, feature health,
target alignment, split hygiene, and inference parity issues.

Usage::

    python backend/scripts/audit_selector_pipeline.py \\
        --priors-frame backend/data/selector/priors_frame \\
        --train-end 2023-12-31 --val-end 2025-06-30

Exit code 0 = all checks pass.  Non-zero = at least one failure.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -- Project imports ----------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algaie.data.priors.selector_schema import (
    TEACHER_PRIORS_COLS,
    Z_FEATURE_COLS,
    DEAD_ZSCORE_COLS,
    MODEL_FEATURE_COLS,
    TARGET_COLS,
)
from algaie.data.priors.feature_utils import (
    recompute_regime_risk,
    compute_date_cross_sectional_stats,
    fit_time_zscore_scaler,
    add_date_regime_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("audit")

# =============================================================================
# Utilities
# =============================================================================

def _load_priors_frame(path: Path) -> pd.DataFrame:
    """Load partitioned or single-file priors frame."""
    if path.is_dir():
        parts = sorted(path.glob("date=*/part.parquet"))
        if not parts:
            raise FileNotFoundError(f"No partitions found in {path}")
        dfs = [pd.read_parquet(p) for p in parts]
        for p, df in zip(parts, dfs):
            date_str = p.parent.name.replace("date=", "")
            df["date"] = pd.to_datetime(date_str)
        return pd.concat(dfs, ignore_index=True)
    return pd.read_parquet(path)


def make_time_split(df, train_end, val_end):
    """Deterministic time split."""
    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)
    train = df[df["date"] <= train_end].copy()
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    test = df[df["date"] > val_end].copy()
    return train, val, test


# =============================================================================
# Check 1: Data Integrity
# =============================================================================

def check_data_integrity(df: pd.DataFrame) -> list[str]:
    """Verify schema, NaN/inf, row counts, missing dates."""
    failures = []

    # 1a: Required columns
    for col in TEACHER_PRIORS_COLS:
        if col not in df.columns:
            failures.append(f"MISSING_COLUMN: {col}")

    # 1b: Critical columns NaN/inf check
    critical_raw = [c for c in ["q10_10", "q50_10", "q90_10", "drift_10",
                                "q10_30", "q50_30", "q90_30", "drift_30",
                                "vol_forecast_10", "tail_risk_10", "prob_up_10",
                                "vol_forecast_30", "tail_risk_30", "prob_up_30",
                                "dispersion_10", "dispersion_30"] if c in df.columns]
    for col in critical_raw:
        n_nan = df[col].isna().sum()
        n_inf = np.isinf(df[col].astype(float)).sum() if df[col].dtype.kind in "fc" else 0
        if n_nan > 0:
            pct = 100 * n_nan / len(df)
            failures.append(f"NAN: {col} has {n_nan} NaN ({pct:.1f}%)")
        if n_inf > 0:
            failures.append(f"INF: {col} has {n_inf} infinite values")

    # 1c: Per-date row counts
    date_counts = df.groupby("date").size()
    median_count = date_counts.median()
    min_count = date_counts.min()
    max_count = date_counts.max()
    logger.info(f"  Per-date rows: min={min_count}, median={median_count:.0f}, max={max_count}")

    # Detect suspicious drops (< 50% of median)
    if median_count > 0:
        low_dates = date_counts[date_counts < 0.5 * median_count]
        if len(low_dates) > 0:
            failures.append(
                f"LOW_ROW_COUNT: {len(low_dates)} dates have <50% of median rows "
                f"(median={median_count:.0f}). First 5: {list(low_dates.index[:5])}"
            )

    # 1d: Missing dates (gaps > 5 trading days)
    dates_sorted = sorted(df["date"].unique())
    if len(dates_sorted) > 1:
        deltas = pd.Series(dates_sorted).diff().dt.days.dropna()
        big_gaps = deltas[deltas > 7]  # > 1 calendar week
        if len(big_gaps) > 0:
            gap_info = [(str(dates_sorted[i]), int(d))
                        for i, d in big_gaps.items()][:5]
            logger.warning(f"  Date gaps >7 days: {gap_info}")

    return failures


# =============================================================================
# Check 2: Feature Health
# =============================================================================

def check_feature_health(df: pd.DataFrame) -> list[str]:
    """Check z-score properties and dead features."""
    failures = []

    # Exclude known dead features from schema (they have documented fallbacks)
    known_dead_z = {f"z_{c}" for c in DEAD_ZSCORE_COLS}
    z_cols_present = [c for c in Z_FEATURE_COLS
                      if c in df.columns and c not in known_dead_z]
    if not z_cols_present:
        failures.append("NO_Z_FEATURES: no live z-score columns found")
        return failures

    # 2a: Dead features (global std < 1e-6) -- check FIRST
    dead_global = []
    for zc in z_cols_present:
        if df[zc].std() < 1e-6:
            dead_global.append(zc)
    if dead_global:
        failures.append(f"DEAD_FEATURES_GLOBAL: {dead_global}")

    # 2b: Z-score properties (sample 10 dates, skip globally dead)
    live_z_cols = [c for c in z_cols_present if c not in dead_global]
    rng = np.random.RandomState(42)
    dates = sorted(df["date"].unique())
    sample_dates = rng.choice(dates, size=min(10, len(dates)), replace=False)

    for dt in sample_dates:
        ddf = df[df["date"] == dt]
        for zc in live_z_cols:
            vals = ddf[zc].dropna()
            if len(vals) < 5:
                continue
            m, s = vals.mean(), vals.std()
            if abs(m) > 0.5:
                failures.append(f"Z_MEAN: {zc} on {dt}: mean={m:.3f} (expected ~0)")
            if s < 0.05 or s > 3.0:
                failures.append(f"Z_STD: {zc} on {dt}: std={s:.3f} (expected ~1)")

    # 2c: Per-date degeneracy (warning only -- fallback is active)
    if "deg_iqr_30" in df.columns:
        n_degen = df.groupby("date")["deg_iqr_30"].first().sum()
        pct_degen = 100 * n_degen / df["date"].nunique()
        logger.info(f"  iqr_30 degenerate dates: {int(n_degen)} ({pct_degen:.1f}%)")
        if pct_degen > 80:
            logger.warning(
                f"  iqr_30 degenerate on {pct_degen:.0f}% of dates "
                f"(fallback to z_tail_risk_30 is active)"
            )

    # 2d: z_cs_tail_30_std variation (if present)
    if "z_cs_tail_30_std" in df.columns:
        z_cs = df.groupby("date")["z_cs_tail_30_std"].first()
        if z_cs.std() < 0.01:
            failures.append(
                f"Z_CS_FLAT: z_cs_tail_30_std is near-constant (std={z_cs.std():.4f})"
            )
        else:
            logger.info(
                f"  z_cs_tail_30_std: range=[{z_cs.min():.3f}, {z_cs.max():.3f}], "
                f"std={z_cs.std():.3f}"
            )

    return failures


# =============================================================================
# Check 3: Target Alignment
# =============================================================================

def check_target_alignment(df: pd.DataFrame) -> list[str]:
    """Verify y_ret target column exists and has reasonable values."""
    failures = []

    if "y_ret" not in df.columns:
        failures.append("NO_TARGET: y_ret column missing")
        return failures

    # Basic stats
    y = df["y_ret"].dropna()
    logger.info(
        f"  y_ret: n={len(y)}, mean={y.mean():.6f}, std={y.std():.4f}, "
        f"range=[{y.min():.4f}, {y.max():.4f}]"
    )

    # Check for extreme values (>100% daily moves)
    n_extreme = (y.abs() > 1.0).sum()
    if n_extreme > 0:
        pct = 100 * n_extreme / len(y)
        if pct > 1.0:
            failures.append(
                f"TARGET_EXTREME: {n_extreme} rows ({pct:.2f}%) have |y_ret| > 100%"
            )
        else:
            logger.info(f"  y_ret extreme (|y|>1): {n_extreme} rows ({pct:.3f}%)")

    # NaN ratio
    n_nan = df["y_ret"].isna().sum()
    if n_nan > 0:
        pct = 100 * n_nan / len(df)
        logger.info(f"  y_ret NaN: {n_nan} ({pct:.1f}%)")
        if pct > 20:
            failures.append(f"TARGET_NAN: y_ret has {pct:.0f}% NaN")

    return failures


# =============================================================================
# Check 3b: Horizon Consistency
# =============================================================================

def check_horizon_consistency(df: pd.DataFrame) -> list[str]:
    """Verify horizon_ret column is uniform (not mixed horizons)."""
    failures = []

    if "horizon_ret" not in df.columns:
        logger.info("  horizon_ret column not present (legacy frame)")
        return failures

    horizons = df["horizon_ret"].dropna().unique()
    if len(horizons) == 0:
        failures.append("HORIZON_EMPTY: horizon_ret column is all NaN")
    elif len(horizons) == 1:
        logger.info(f"  horizon_ret: {int(horizons[0])}d (uniform ✓)")
    else:
        failures.append(
            f"HORIZON_MIXED: found multiple horizons: {sorted(horizons)}"
        )

    return failures


# =============================================================================
# Check 4: Split Hygiene + Scaler Leakage
# =============================================================================

def check_split_hygiene(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> list[str]:
    """Verify split boundaries and scaler fitted on train only."""
    failures = []
    train_df, val_df, test_df = make_time_split(df, train_end, val_end)

    logger.info(
        f"  Split sizes: train={len(train_df)} ({train_df['date'].nunique()}d), "
        f"val={len(val_df)} ({val_df['date'].nunique()}d), "
        f"test={len(test_df)} ({test_df['date'].nunique()}d)"
    )

    # Check no overlap
    train_dates = set(train_df["date"].unique())
    val_dates = set(val_df["date"].unique())
    test_dates = set(test_df["date"].unique())

    if train_dates & val_dates:
        failures.append("SPLIT_OVERLAP: train and val share dates")
    if train_dates & test_dates:
        failures.append("SPLIT_OVERLAP: train and test share dates")
    if val_dates & test_dates:
        failures.append("SPLIT_OVERLAP: val and test share dates")

    # Verify chronological ordering
    if len(val_df) > 0 and train_df["date"].max() >= val_df["date"].min():
        failures.append("SPLIT_ORDER: train max >= val min")
    if len(test_df) > 0 and len(val_df) > 0 and val_df["date"].max() >= test_df["date"].min():
        failures.append("SPLIT_ORDER: val max >= test min")

    # Scaler leakage test: fit on train vs fit on all
    train_stats = compute_date_cross_sectional_stats(train_df)
    all_stats = compute_date_cross_sectional_stats(df)

    scaler_train = fit_time_zscore_scaler(train_stats, "cs_tail_30_std")
    scaler_all = fit_time_zscore_scaler(all_stats, "cs_tail_30_std")

    mu_diff = abs(scaler_train["mu"] - scaler_all["mu"])
    sigma_diff = abs(scaler_train["sigma"] - scaler_all["sigma"])
    logger.info(
        f"  Scaler (train-only): mu={scaler_train['mu']:.6f}, sigma={scaler_train['sigma']:.6f}"
    )
    logger.info(
        f"  Scaler (all dates):  mu={scaler_all['mu']:.6f}, sigma={scaler_all['sigma']:.6f}"
    )
    logger.info(f"  Difference: dmu={mu_diff:.6f}, dsigma={sigma_diff:.6f}")

    if mu_diff < 1e-10 and sigma_diff < 1e-10 and len(val_df) > 0:
        failures.append(
            "LEAKAGE_SUSPECT: train-only and all-data scalers are identical "
            "despite having val/test data -- possible leakage"
        )

    return failures


# =============================================================================
# Check 5: Inference Parity
# =============================================================================

def check_inference_parity(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> list[str]:
    """Verify z_cs computed offline matches re-computation with saved scaler."""
    failures = []

    train_df, val_df, test_df = make_time_split(df, train_end, val_end)

    # Fit scaler on train
    train_stats = compute_date_cross_sectional_stats(train_df)
    scaler = fit_time_zscore_scaler(train_stats, "cs_tail_30_std")

    # Apply to full df (simulate training pipeline)
    df_train_applied, _ = add_date_regime_features(train_df.copy(), scaler=scaler)
    df_val_applied, _ = add_date_regime_features(val_df.copy(), scaler=scaler)

    # Pick random val dates and recompute (simulate inference)
    if len(val_df) == 0:
        logger.warning("  No val data for inference parity check")
        return []

    rng = np.random.RandomState(42)
    check_dates = rng.choice(val_df["date"].unique(),
                              size=min(3, val_df["date"].nunique()),
                              replace=False)

    for dt in check_dates:
        single = val_df[val_df["date"] == dt].copy()
        single_out, _ = add_date_regime_features(single, scaler=scaler)

        # Compare to batch-applied value
        batch_val = df_val_applied[df_val_applied["date"] == dt]["z_cs_tail_30_std"].iloc[0]
        single_val = single_out["z_cs_tail_30_std"].iloc[0]

        delta = abs(batch_val - single_val)
        if delta > 1e-6:
            failures.append(
                f"PARITY_MISMATCH: date={dt}, batch z_cs={batch_val:.6f}, "
                f"single z_cs={single_val:.6f}, delta={delta:.8f}"
            )

    if not failures:
        logger.info(f"  Inference parity: OK ({len(check_dates)} dates checked)")

    return failures


# =============================================================================
# Main
# =============================================================================

def run_audit(args) -> int:
    """Run all preflight checks. Returns number of failures."""
    logger.info("=" * 60)
    logger.info("Selector Pipeline Preflight Audit")
    logger.info("=" * 60)

    # Load data
    logger.info(f"\nLoading priors frame: {args.priors_frame}")
    try:
        df = _load_priors_frame(Path(args.priors_frame))
    except Exception as e:
        logger.error(f"Failed to load priors frame: {e}")
        return 1

    logger.info(f"  Total rows: {len(df)}, dates: {df['date'].nunique()}, "
                f"symbols: {df['symbol'].nunique()}")

    # Recompute regime risk (same as training)
    df = recompute_regime_risk(df)

    all_failures = []

    # Check 1: Data Integrity
    logger.info("\n-- Check 1: Data Integrity --")
    f = check_data_integrity(df)
    all_failures.extend(f)
    logger.info(f"  Result: {'PASS' if not f else f'FAIL ({len(f)} issues)'}")

    # Check 2: Feature Health
    logger.info("\n-- Check 2: Feature Health --")
    f = check_feature_health(df)
    all_failures.extend(f)
    logger.info(f"  Result: {'PASS' if not f else f'FAIL ({len(f)} issues)'}")

    # Check 3: Target Alignment
    logger.info("\n-- Check 3: Target Alignment --")
    f = check_target_alignment(df)
    all_failures.extend(f)
    logger.info(f"  Result: {'PASS' if not f else f'FAIL ({len(f)} issues)'}")

    # Check 3b: Horizon Consistency
    logger.info("\n-- Check 3b: Horizon Consistency --")
    f = check_horizon_consistency(df)
    all_failures.extend(f)
    logger.info(f"  Result: {'PASS' if not f else f'FAIL ({len(f)} issues)'}")

    # Check 4: Split Hygiene
    logger.info("\n-- Check 4: Split Hygiene + Scaler --")
    f = check_split_hygiene(df, args.train_end, args.val_end)
    all_failures.extend(f)
    logger.info(f"  Result: {'PASS' if not f else f'FAIL ({len(f)} issues)'}")

    # Check 5: Inference Parity
    logger.info("\n-- Check 5: Inference Parity --")
    f = check_inference_parity(df, args.train_end, args.val_end)
    all_failures.extend(f)
    logger.info(f"  Result: {'PASS' if not f else f'FAIL ({len(f)} issues)'}")

    # Summary
    logger.info("\n" + "=" * 60)
    if all_failures:
        logger.error(f"AUDIT FAILED: {len(all_failures)} issue(s)")
        for i, fail in enumerate(all_failures, 1):
            logger.error(f"  [{i}] {fail}")
        return len(all_failures)
    else:
        logger.info("ALL CHECKS PASSED -- safe to train")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Preflight audit for selector training pipeline"
    )
    parser.add_argument(
        "--priors-frame", required=True,
        help="Path to priors_frame dir or parquet"
    )
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--val-end", default="2025-06-30")
    args = parser.parse_args()

    n_failures = run_audit(args)
    sys.exit(1 if n_failures > 0 else 0)


if __name__ == "__main__":
    main()
