"""Risk-head calibration report for promotion gating.

Evaluates whether the risk head's predictions (proxy for ``|r_oc|``) are
calibrated—i.e., higher predicted risk corresponds to higher realized
tail magnitude.

Two checks
----------
1. **Spearman correlation** between ``risk_pred`` and ``log(eps + |r_oc|)``
   must be > 0.
2. **Bucket monotonicity**: 5 quantiles of ``risk_pred`` should exhibit
   monotonically increasing mean ``|r_oc|`` (with configurable tolerance).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskCalibrationReport:
    """Outcome of risk-head calibration check."""
    spearman_corr: float
    spearman_pvalue: float
    bucket_means: Dict[int, float]       # quantile index → mean |r_oc|
    bucket_pred_means: Dict[int, float]   # quantile index -> mean risk_pred
    monotonic: bool                       # True if bucket means mostly increase
    gate_passed: bool
    # F2 additions
    saturation_fraction_floor: float = 0.0  # pct sigma <= floor+eps
    saturation_fraction_cap: float = 0.0    # pct sigma > cap (if cap used)
    mean_daily_sigma_dispersion: float = 0.0  # mean var(sigma) across days

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def compute_risk_calibration(
    risk_pred: np.ndarray,
    r_oc_realized: np.ndarray,
    *,
    n_buckets: int = 5,
    eps: float = 1e-6,
    min_corr: float = 0.0,
    monotonic_tolerance: int = 1,
) -> RiskCalibrationReport:
    """Compute calibration metrics for the risk head.

    Parameters
    ----------
    risk_pred
        Model's risk predictions (higher = riskier).
    r_oc_realized
        Realized OC returns (raw, not absolute).
    n_buckets
        Number of quantile buckets for monotonicity check.
    eps
        Epsilon added before log transform.
    min_corr
        Minimum acceptable Spearman correlation.
    monotonic_tolerance
        Number of non-monotonic bucket transitions allowed.

    Returns
    -------
    RiskCalibrationReport
    """
    mask = np.isfinite(risk_pred) & np.isfinite(r_oc_realized)
    rp = risk_pred[mask]
    ro = np.abs(r_oc_realized[mask])
    log_ro = np.log(eps + ro)

    # Spearman correlation
    corr, pvalue = scipy_stats.spearmanr(rp, log_ro)

    # Bucket monotonicity
    quantiles = pd.qcut(rp, q=n_buckets, labels=False, duplicates="drop")
    df = pd.DataFrame({"q": quantiles, "abs_roc": ro, "risk_pred": rp})
    bucket_agg = df.groupby("q").agg({"abs_roc": "mean", "risk_pred": "mean"}).sort_index()

    bucket_means = {int(k): float(v) for k, v in bucket_agg["abs_roc"].items()}
    bucket_pred_means = {int(k): float(v) for k, v in bucket_agg["risk_pred"].items()}

    # Check monotonicity: count inversions
    vals = list(bucket_agg["abs_roc"].values)
    inversions = sum(1 for i in range(1, len(vals)) if vals[i] < vals[i - 1])
    is_monotonic = inversions <= monotonic_tolerance

    gate_passed = (corr > min_corr) and is_monotonic

    return RiskCalibrationReport(
        spearman_corr=float(corr),
        spearman_pvalue=float(pvalue),
        bucket_means=bucket_means,
        bucket_pred_means=bucket_pred_means,
        monotonic=is_monotonic,
        gate_passed=gate_passed,
    )


def compute_risk_calibration_extended(
    risk_pred: np.ndarray,
    r_oc_realized: np.ndarray,
    *,
    n_buckets: int = 5,
    eps: float = 1e-6,
    min_corr: float = 0.0,
    monotonic_tolerance: int = 1,
    sigma_floor: float = 1e-4,
    sigma_cap: float | None = None,
    trading_days: np.ndarray | None = None,
) -> RiskCalibrationReport:
    """Extended calibration with F2 anti-saturation diagnostics."""
    base = compute_risk_calibration(
        risk_pred, r_oc_realized,
        n_buckets=n_buckets, eps=eps,
        min_corr=min_corr, monotonic_tolerance=monotonic_tolerance,
    )

    mask = np.isfinite(risk_pred) & np.isfinite(r_oc_realized)
    rp = risk_pred[mask]

    # Saturation fractions
    floor_frac = float(np.mean(rp <= sigma_floor + 1e-8))
    cap_frac = 0.0
    if sigma_cap is not None:
        cap_frac = float(np.mean(rp > sigma_cap * 0.99))

    # Per-day sigma dispersion
    mean_disp = 0.0
    if trading_days is not None:
        days_masked = trading_days[mask] if len(trading_days) == len(risk_pred) else None
        if days_masked is not None:
            df = pd.DataFrame({"rp": rp, "day": days_masked})
            day_vars = df.groupby("day")["rp"].var().dropna()
            mean_disp = float(day_vars.mean()) if len(day_vars) > 0 else 0.0

    return RiskCalibrationReport(
        spearman_corr=base.spearman_corr,
        spearman_pvalue=base.spearman_pvalue,
        bucket_means=base.bucket_means,
        bucket_pred_means=base.bucket_pred_means,
        monotonic=base.monotonic,
        gate_passed=base.gate_passed,
        saturation_fraction_floor=floor_frac,
        saturation_fraction_cap=cap_frac,
        mean_daily_sigma_dispersion=mean_disp,
    )


def run_risk_calibration(
    dataset: pd.DataFrame,
    predictions: np.ndarray,
    *,
    r_oc_col: str = "r_oc",
) -> dict:
    """CLI-facing wrapper: run risk calibration and return a dict.

    Parameters
    ----------
    dataset
        Must contain ``r_oc`` column (realized OC returns).
    predictions
        Model predictions (used as risk_pred proxy).

    Returns
    -------
    dict
        Keys: ``spearman_correlation``, ``monotonicity_ok``, ``gate_passed``,
        plus full ``bucket_means`` and ``bucket_pred_means``.
    """
    if r_oc_col not in dataset.columns:
        logger.warning("Column '%s' missing — skipping risk calibration", r_oc_col)
        return {
            "spearman_correlation": 0.0,
            "monotonicity_ok": False,
            "gate_passed": False,
        }

    r_oc = dataset[r_oc_col].values
    # Use absolute predictions as risk proxy
    risk_pred = np.abs(predictions).flatten()

    # Align lengths
    n = min(len(risk_pred), len(r_oc))
    report = compute_risk_calibration(risk_pred[:n], r_oc[:n])

    return {
        "spearman_correlation": report.spearman_corr,
        "spearman_pvalue": report.spearman_pvalue,
        "monotonicity_ok": report.monotonic,
        "gate_passed": report.gate_passed,
        "bucket_means": report.bucket_means,
        "bucket_pred_means": report.bucket_pred_means,
    }


def save_report(report: RiskCalibrationReport, path: Path) -> None:
    """Write calibration report to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Risk calibration report → %s (passed=%s)", path, report.gate_passed)
