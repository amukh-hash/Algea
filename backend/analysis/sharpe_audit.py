"""
Sharpe Integrity Audit — 8-phase diagnostic for suspicious Sharpe ratios.

Detects:
  - Autocorrelation-inflated Sharpe (Newey-West adjustment)
  - Under-deployment Sharpe inflation
  - Cost-sensitive Sharpe
  - Concentrated alpha (top-N day dependence)
  - Regime-dependent Sharpe
  - Unstable rolling Sharpe

Entry point: ``run_sharpe_audit()``
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.analysis.sharpe_report import (
    TRADING_DAYS_PER_YEAR,
    MIN_OBS,
    compute_sharpe,
    _ann_return,
    _ann_vol,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Autocorrelation Audit
# ═══════════════════════════════════════════════════════════════════════════

def compute_autocorrelations(returns: pd.Series, max_lag: int = 5) -> List[float]:
    """Compute lag-1 through lag-max_lag autocorrelations."""
    n = len(returns)
    result: List[float] = []
    for lag in range(1, max_lag + 1):
        if n <= lag:
            result.append(0.0)
        else:
            result.append(round(float(returns.autocorr(lag=lag)), 8))
    return result


def ljung_box_pvalues(returns: pd.Series, max_lag: int = 5) -> List[float]:
    """Compute Ljung-Box test p-values for lag-1 through lag-max_lag.

    Manual implementation (no statsmodels dependency):
    Q(k) = n*(n+2) * sum_{j=1}^{k} rho_j^2 / (n-j)
    Compared against chi-squared(k) distribution.
    """
    n = len(returns)
    if n < max_lag + 1:
        return [1.0] * max_lag

    acfs = compute_autocorrelations(returns, max_lag)
    pvalues: List[float] = []

    for k in range(1, max_lag + 1):
        Q = 0.0
        for j in range(k):
            lag = j + 1
            rho = acfs[j]
            Q += (rho ** 2) / (n - lag)
        Q *= n * (n + 2)

        # Chi-squared survival function (manual via incomplete gamma)
        pval = _chi2_sf(Q, k)
        pvalues.append(round(pval, 8))

    return pvalues


def _chi2_sf(x: float, df: int) -> float:
    """Survival function of chi-squared distribution (1 - CDF).

    Uses the regularized incomplete gamma function via series expansion.
    """
    if x <= 0:
        return 1.0
    return 1.0 - _regularized_gamma_p(df / 2.0, x / 2.0)


def _regularized_gamma_p(a: float, x: float, max_iter: int = 200) -> float:
    """Lower regularized incomplete gamma function P(a, x).

    Uses series expansion: P(a,x) = e^{-x} * x^a * sum_{n=0}^{inf} x^n / Gamma(a+n+1)
    """
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0

    # Use series expansion
    ap = a
    s = 1.0 / a
    delta = s
    for n in range(1, max_iter):
        ap += 1.0
        delta *= x / ap
        s += delta
        if abs(delta) < abs(s) * 1e-12:
            break

    log_gamma_a = _log_gamma(a)
    return s * np.exp(-x + a * np.log(x) - log_gamma_a)


def _log_gamma(x: float) -> float:
    """Log-gamma via Stirling's approximation + Lanczos coefficients."""
    if x <= 0:
        return 0.0
    # Use numpy for reliability
    return float(math.lgamma(x))


def newey_west_vol(returns: pd.Series, max_lag: int = 5) -> float:
    """Newey-West adjusted annualized volatility.

    HAC estimator:
    sigma^2_NW = gamma_0 + 2 * sum_{j=1}^{L} (1 - j/(L+1)) * gamma_j

    where gamma_j = Cov(r_t, r_{t-j}).
    """
    n = len(returns)
    mu = float(returns.mean())
    demean = returns.values - mu

    # gamma_0
    gamma_0 = float(np.dot(demean, demean)) / n

    nw_var = gamma_0
    for j in range(1, max_lag + 1):
        if j >= n:
            break
        gamma_j = float(np.dot(demean[j:], demean[:-j])) / n
        weight = 1.0 - j / (max_lag + 1)
        nw_var += 2 * weight * gamma_j

    # Ensure non-negative
    nw_var = max(nw_var, 0.0)
    daily_vol = np.sqrt(nw_var)
    return float(daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR))


def newey_west_sharpe(returns: pd.Series, rf_annual: float = 0.0,
                      max_lag: int = 5) -> float:
    """Sharpe ratio using Newey-West adjusted volatility."""
    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
    excess = returns - rf_daily
    mu_ann = float(excess.mean()) * TRADING_DAYS_PER_YEAR
    vol_nw = newey_west_vol(excess, max_lag)
    if vol_nw < 1e-12:
        return 0.0
    return float(mu_ann / vol_nw)


def audit_autocorrelation(
    df: pd.DataFrame,
    rf_annual: float = 0.0,
) -> Dict[str, Any]:
    """Full autocorrelation audit for all sleeves."""
    result: Dict[str, Any] = {}

    for label, col in [("core", "core_ret"), ("vrp", "vrp_ret"), ("combined", "port_ret")]:
        returns = df[col]
        acfs = compute_autocorrelations(returns)
        pvals = ljung_box_pvalues(returns)

        significant = any(p < 0.05 for p in pvals)

        entry: Dict[str, Any] = {
            "autocorrelations": {f"lag_{i+1}": acfs[i] for i in range(len(acfs))},
            "ljung_box_pvalues": {f"lag_{i+1}": pvals[i] for i in range(len(pvals))},
            "significant_autocorrelation": significant,
            "naive_sharpe": round(compute_sharpe(returns, rf_annual), 6)
                if len(returns) >= MIN_OBS else None,
        }

        if significant:
            entry["nw_sharpe_lag5"] = round(newey_west_sharpe(returns, rf_annual, 5), 6)
            entry["nw_sharpe_lag10"] = round(newey_west_sharpe(returns, rf_annual, 10), 6)
            entry["nw_vol_lag5"] = round(newey_west_vol(returns, 5), 6)
            entry["nw_vol_lag10"] = round(newey_west_vol(returns, 10), 6)
        else:
            entry["nw_sharpe_lag5"] = entry["naive_sharpe"]
            entry["nw_sharpe_lag10"] = entry["naive_sharpe"]

        result[label] = entry

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — Capital Utilization Audit
# ═══════════════════════════════════════════════════════════════════════════

def audit_capital_utilization(
    df: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
) -> Dict[str, Any]:
    """Assess whether low deployment inflates Sharpe.

    Uses w_vrp as a proxy for capital utilization:
    - core_util = (1 - w_vrp) deployed to core
    - vrp_util  = w_vrp deployed to VRP
    - total_util = effective deployment fraction

    Since the strategy invests in core + VRP with weights summing to 1,
    if gross exposure is low, it deflates vol and inflates Sharpe.
    """
    w_vrp = df["w_vrp"]

    # Effective gross utilization: using w_vrp directly.
    # Full deployment = gross exposure ≈ 1.0.
    # If the backtest is synthetic, we track w_vrp as the VRP fraction.
    gross_util = pd.Series(np.ones(len(df)), index=df.index)  # weights sum to 1

    mean_w_vrp = float(w_vrp.mean())

    # Build a utilization proxy: if vol is unusually low relative to
    # what a fully-invested portfolio would produce, flag it.
    core_ret = df["core_ret"]
    port_ret = df["port_ret"]

    ann_vol = float(port_ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

    # Utilization ratio heuristic:
    # If annual vol < 5%, strategy is likely under-deployed
    under_deployed = ann_vol < 0.05

    corr_util_ret = float(w_vrp.corr(port_ret.abs())) if len(df) > 2 else 0.0

    return {
        "mean_w_vrp": round(mean_w_vrp, 6),
        "ann_vol_portfolio": round(ann_vol, 6),
        "under_deployed_flag": under_deployed,
        "vol_threshold": 0.05,
        "correlation_weight_abs_return": round(corr_util_ret, 6),
        "diagnosis": (
            "UNDER_DEPLOYMENT: Portfolio vol < 5% suggests capital is under-deployed, "
            "mechanically inflating Sharpe ratio."
            if under_deployed else
            "Capital deployment appears normal."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — Turnover & Cost Sensitivity
# ═══════════════════════════════════════════════════════════════════════════

def compute_turnover(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute daily and annualized turnover from weight changes."""
    w_vrp = df["w_vrp"]
    daily_turnover = w_vrp.diff().abs()
    daily_turnover.iloc[0] = abs(w_vrp.iloc[0])  # initial deployment

    mean_daily = float(daily_turnover.mean())
    ann_turnover = mean_daily * TRADING_DAYS_PER_YEAR

    # Average holding period (inverse of one-way turnover)
    avg_holding = 1.0 / max(mean_daily, 1e-12)

    return {
        "mean_daily_turnover": round(mean_daily, 8),
        "annualized_turnover": round(ann_turnover, 6),
        "avg_holding_period_days": round(avg_holding, 2),
    }


def cost_sensitivity_analysis(
    df: pd.DataFrame,
    rf_annual: float = 0.0,
    cost_bps_scenarios: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Recompute Sharpe under additional cost shocks.

    Applies per-trade cost to each weight change (turnover).
    """
    if cost_bps_scenarios is None:
        cost_bps_scenarios = [5.0, 10.0, 20.0]

    turnover_info = compute_turnover(df)
    daily_turnover = df["w_vrp"].diff().abs()
    daily_turnover.iloc[0] = abs(df["w_vrp"].iloc[0])

    port_ret = df["port_ret"]
    baseline_sharpe = compute_sharpe(port_ret, rf_annual) if len(port_ret) >= MIN_OBS else None

    scenarios: Dict[str, Any] = {}
    for bps in cost_bps_scenarios:
        cost_fraction = bps / 10_000
        daily_cost = daily_turnover * cost_fraction
        adjusted_ret = port_ret - daily_cost

        if len(adjusted_ret) >= MIN_OBS:
            adj_sharpe = round(compute_sharpe(adjusted_ret, rf_annual), 6)
        else:
            adj_sharpe = None

        label = f"+{int(bps)}bps"
        scenarios[label] = {
            "sharpe": adj_sharpe,
            "total_cost_drag_bps": round(float(daily_cost.sum() * 10_000), 2),
        }

    return {
        **turnover_info,
        "baseline_sharpe": round(baseline_sharpe, 6) if baseline_sharpe is not None else None,
        "cost_scenarios": scenarios,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4 — Bootstrap Confidence Interval
# ═══════════════════════════════════════════════════════════════════════════

def block_bootstrap_sharpe(
    returns: pd.Series,
    rf_annual: float = 0.0,
    block_length: int = 5,
    n_samples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Block bootstrap confidence interval for Sharpe ratio.

    Uses non-overlapping blocks of length `block_length`.
    Deterministic via fixed seed.
    """
    rng = np.random.RandomState(seed)
    r = returns.values
    n = len(r)
    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR

    # Number of blocks needed to fill the sample
    n_blocks = int(np.ceil(n / block_length))

    sharpes: List[float] = []
    for _ in range(n_samples):
        # Sample block start indices
        starts = rng.randint(0, n - block_length + 1, size=n_blocks)
        # Build bootstrap sample
        sample = np.concatenate([r[s:s + block_length] for s in starts])[:n]
        excess = sample - rf_daily
        mu = float(excess.mean())
        sigma = float(np.std(excess, ddof=1))
        if sigma < 1e-12:
            sharpes.append(0.0)
        else:
            sharpes.append(float(np.sqrt(TRADING_DAYS_PER_YEAR) * mu / sigma))

    sharpes_arr = np.array(sharpes)
    return {
        "n_samples": n_samples,
        "block_length": block_length,
        "mean_sharpe": round(float(sharpes_arr.mean()), 6),
        "std_sharpe": round(float(sharpes_arr.std(ddof=1)), 6),
        "ci_5": round(float(np.percentile(sharpes_arr, 5)), 6),
        "ci_95": round(float(np.percentile(sharpes_arr, 95)), 6),
        "ci_2_5": round(float(np.percentile(sharpes_arr, 2.5)), 6),
        "ci_97_5": round(float(np.percentile(sharpes_arr, 97.5)), 6),
    }


def audit_bootstrap(
    df: pd.DataFrame,
    rf_annual: float = 0.0,
) -> Dict[str, Any]:
    """Bootstrap CI for all sleeves."""
    result: Dict[str, Any] = {}
    for label, col in [("core", "core_ret"), ("vrp", "vrp_ret"), ("combined", "port_ret")]:
        result[label] = block_bootstrap_sharpe(df[col], rf_annual)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5 — Rolling Stability
# ═══════════════════════════════════════════════════════════════════════════

def audit_rolling_stability(
    df: pd.DataFrame,
    rf_annual: float = 0.0,
) -> Dict[str, Any]:
    """Rolling Sharpe stability for 63-day and 126-day windows."""
    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
    result: Dict[str, Any] = {}

    for window in [63, 126]:
        window_result: Dict[str, Any] = {}
        for label, col in [("core", "core_ret"), ("vrp", "vrp_ret"), ("combined", "port_ret")]:
            excess = df[col] - rf_daily
            roll_mean = excess.rolling(window, min_periods=window).mean()
            roll_std = excess.rolling(window, min_periods=window).std(ddof=1)
            roll_std = roll_std.replace(0, np.nan)
            rolling_sharpe = np.sqrt(TRADING_DAYS_PER_YEAR) * roll_mean / roll_std
            valid = rolling_sharpe.dropna()

            if len(valid) == 0:
                window_result[label] = {
                    "mean": None, "std": None, "min": None, "max": None,
                    "pct_negative": None, "unstable": False,
                }
            else:
                std_val = float(valid.std(ddof=1))
                mean_val = float(valid.mean())
                pct_negative = float((valid < 0).sum() / len(valid) * 100)

                # Unstable if std of rolling Sharpe > 1.0 or >50% of mean
                unstable = (std_val > 1.0) or (
                    abs(mean_val) > 0.1 and std_val / abs(mean_val) > 0.5
                )

                window_result[label] = {
                    "mean": round(mean_val, 6),
                    "std": round(std_val, 6),
                    "min": round(float(valid.min()), 6),
                    "max": round(float(valid.max()), 6),
                    "pct_negative": round(pct_negative, 2),
                    "unstable": unstable,
                }

        result[f"{window}d"] = window_result

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6 — PnL Concentration
# ═══════════════════════════════════════════════════════════════════════════

def audit_pnl_concentration(df: pd.DataFrame) -> Dict[str, Any]:
    """Measure dependence of total PnL on top N days."""
    result: Dict[str, Any] = {}

    for label, col in [("core", "core_pnl"), ("vrp", "vrp_pnl")]:
        if col not in df.columns:
            continue

        pnl = df[col]
        total_pnl = float(pnl.sum())

        if abs(total_pnl) < 1e-10:
            result[label] = {
                "top5_pct": 0.0, "top10_pct": 0.0,
                "concentrated": False,
            }
            continue

        sorted_pnl = pnl.sort_values(ascending=False)
        top5_pnl = float(sorted_pnl.head(5).sum())
        top10_pnl = float(sorted_pnl.head(10).sum())

        top5_pct = abs(top5_pnl / total_pnl) * 100
        top10_pct = abs(top10_pnl / total_pnl) * 100

        concentrated = top5_pct > 30 or top10_pct > 50

        result[label] = {
            "total_pnl": round(total_pnl, 2),
            "top5_pnl": round(top5_pnl, 2),
            "top10_pnl": round(top10_pnl, 2),
            "top5_pct": round(top5_pct, 2),
            "top10_pct": round(top10_pct, 2),
            "concentrated": concentrated,
        }

    # Combined (portfolio-level)
    port_ret = df["port_ret"]
    total_ret = float(port_ret.sum())
    if abs(total_ret) > 1e-10:
        sorted_ret = port_ret.sort_values(ascending=False)
        t5 = float(sorted_ret.head(5).sum())
        t10 = float(sorted_ret.head(10).sum())
        t5_pct = abs(t5 / total_ret) * 100
        t10_pct = abs(t10 / total_ret) * 100
        result["combined"] = {
            "total_return_sum": round(total_ret, 8),
            "top5_pct": round(t5_pct, 2),
            "top10_pct": round(t10_pct, 2),
            "concentrated": t5_pct > 30 or t10_pct > 50,
        }
    else:
        result["combined"] = {"top5_pct": 0.0, "top10_pct": 0.0, "concentrated": False}

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7 — Out-of-Sample Segmentation
# ═══════════════════════════════════════════════════════════════════════════

def audit_oos_segmentation(
    df: pd.DataFrame,
    rf_annual: float = 0.0,
) -> Dict[str, Any]:
    """Split into first half / second half and compare Sharpe."""
    n = len(df)
    mid = n // 2
    first_half = df.iloc[:mid]
    second_half = df.iloc[mid:]

    result: Dict[str, Any] = {
        "n_first": len(first_half),
        "n_second": len(second_half),
    }

    for label, col in [("core", "core_ret"), ("vrp", "vrp_ret"), ("combined", "port_ret")]:
        r1 = first_half[col]
        r2 = second_half[col]

        s1 = compute_sharpe(r1, rf_annual) if len(r1) >= MIN_OBS else None
        s2 = compute_sharpe(r2, rf_annual) if len(r2) >= MIN_OBS else None

        if s1 is not None and s2 is not None:
            ratio = s2 / s1 if abs(s1) > 1e-8 else None
            regime_dependent = ratio is not None and ratio < 0.5
        else:
            ratio = None
            regime_dependent = False

        result[label] = {
            "sharpe_first_half": round(s1, 6) if s1 is not None else None,
            "sharpe_second_half": round(s2, 6) if s2 is not None else None,
            "second_to_first_ratio": round(ratio, 4) if ratio is not None else None,
            "regime_dependent": regime_dependent,
        }

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Phase 8 — Final Diagnostic Summary
# ═══════════════════════════════════════════════════════════════════════════

def classify_sharpe_integrity(
    autocorrelation: Dict[str, Any],
    utilization: Dict[str, Any],
    cost: Dict[str, Any],
    bootstrap: Dict[str, Any],
    rolling: Dict[str, Any],
    concentration: Dict[str, Any],
    oos: Dict[str, Any],
) -> Dict[str, Any]:
    """Produce a final integrity classification."""
    flags: List[str] = []
    details: Dict[str, str] = {}

    # ── Autocorrelation ──────────────────────────────────────────────
    combined_ac = autocorrelation.get("combined", {})
    if combined_ac.get("significant_autocorrelation"):
        naive = combined_ac.get("naive_sharpe", 0) or 0
        nw5 = combined_ac.get("nw_sharpe_lag5", 0) or 0
        if abs(naive) > 1e-8:
            pct_drop = (1 - nw5 / naive) * 100
            if pct_drop > 10:
                flags.append("VOLATILITY_SMOOTHING")
                details["volatility_smoothing"] = (
                    f"Newey-West Sharpe ({nw5:.2f}) is {pct_drop:.1f}% lower than "
                    f"naive Sharpe ({naive:.2f}). Autocorrelation inflates the ratio."
                )

    # ── Under-deployment ─────────────────────────────────────────────
    if utilization.get("under_deployed_flag"):
        flags.append("UNDER_DEPLOYMENT")
        details["under_deployment"] = utilization.get("diagnosis", "")

    # ── Cost sensitivity ─────────────────────────────────────────────
    baseline = cost.get("baseline_sharpe", 0) or 0
    cost_10 = cost.get("cost_scenarios", {}).get("+10bps", {}).get("sharpe")
    if baseline > 0 and cost_10 is not None:
        pct_drop = (1 - cost_10 / baseline) * 100 if baseline > 1e-8 else 0
        if pct_drop > 20:
            flags.append("COST_SENSITIVE")
            details["cost_sensitive"] = (
                f"Sharpe drops {pct_drop:.1f}% under +10bps cost. "
                f"Baseline={baseline:.2f}, +10bps={cost_10:.2f}."
            )

    # ── Concentration ────────────────────────────────────────────────
    combined_conc = concentration.get("combined", {})
    if combined_conc.get("concentrated"):
        flags.append("CONCENTRATED_ALPHA")
        details["concentrated_alpha"] = (
            f"Top 5 days = {combined_conc.get('top5_pct', 0):.1f}% of total return. "
            f"Alpha is narrowly sourced."
        )

    # ── Rolling instability ──────────────────────────────────────────
    r63 = rolling.get("63d", {}).get("combined", {})
    if r63.get("unstable"):
        flags.append("ROLLING_UNSTABLE")
        details["rolling_unstable"] = (
            f"63-day rolling Sharpe std={r63.get('std', 'N/A')}, "
            f"range=[{r63.get('min', 'N/A')}, {r63.get('max', 'N/A')}]."
        )

    # ── OOS regime dependence ────────────────────────────────────────
    oos_combined = oos.get("combined", {})
    if oos_combined.get("regime_dependent"):
        flags.append("REGIME_DEPENDENT")
        details["regime_dependent"] = (
            f"First-half Sharpe={oos_combined.get('sharpe_first_half')}, "
            f"Second-half={oos_combined.get('sharpe_second_half')}. "
            f"Strategy may be overfit to early regime."
        )

    # ── Final classification ─────────────────────────────────────────
    if not flags:
        classification = "ROBUST_HIGH_SHARPE"
    elif "VOLATILITY_SMOOTHING" in flags:
        classification = "LIKELY_VOLATILITY_SMOOTHING_ARTIFACT"
    elif "UNDER_DEPLOYMENT" in flags and "COST_SENSITIVE" not in flags:
        classification = "UNDER_DEPLOYMENT_DRIVEN_SHARPE"
    elif "COST_SENSITIVE" in flags:
        classification = "COST_SENSITIVE_SHARPE"
    elif "CONCENTRATED_ALPHA" in flags:
        classification = "CONCENTRATED_ALPHA"
    elif "REGIME_DEPENDENT" in flags:
        classification = "REGIME_DEPENDENT_SHARPE"
    elif "ROLLING_UNSTABLE" in flags:
        classification = "UNSTABLE_SHARPE"
    else:
        classification = "INCONCLUSIVE"

    return {
        "classification": classification,
        "flags": flags,
        "details": details,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Top-level API
# ═══════════════════════════════════════════════════════════════════════════

def run_sharpe_audit(
    run_id: str,
    df: pd.DataFrame,
    rf_annual: float = 0.0,
    initial_capital: float = 1_000_000.0,
) -> Dict[str, Any]:
    """Run full 8-phase Sharpe Integrity Audit.

    Parameters
    ----------
    run_id : str
        Run identifier.
    df : pd.DataFrame
        Must already have: core_ret, vrp_ret, port_ret, w_vrp, core_pnl, vrp_pnl.
        (Output of validate_and_load or load_daily_data.)
    rf_annual : float
        Annual risk-free rate.
    initial_capital : float
        Starting capital.

    Returns
    -------
    dict with all 8 phases.
    """
    logger.info(f"Starting Sharpe Integrity Audit for run '{run_id}' "
                f"({len(df)} observations).")

    autocorrelation = audit_autocorrelation(df, rf_annual)
    utilization = audit_capital_utilization(df, initial_capital)
    cost = cost_sensitivity_analysis(df, rf_annual)
    bootstrap = audit_bootstrap(df, rf_annual)
    rolling = audit_rolling_stability(df, rf_annual)
    concentration = audit_pnl_concentration(df)
    oos = audit_oos_segmentation(df, rf_annual)
    integrity = classify_sharpe_integrity(
        autocorrelation, utilization, cost, bootstrap, rolling, concentration, oos
    )

    report = {
        "run_id": run_id,
        "n_observations": len(df),
        "autocorrelation_audit": autocorrelation,
        "capital_utilization": utilization,
        "turnover_cost_sensitivity": cost,
        "bootstrap_confidence": bootstrap,
        "rolling_stability": rolling,
        "pnl_concentration": concentration,
        "oos_segmentation": oos,
        "integrity_diagnosis": integrity,
    }

    return report


# ═══════════════════════════════════════════════════════════════════════════
# Console Printer
# ═══════════════════════════════════════════════════════════════════════════

def print_audit_report(report: Dict[str, Any]) -> None:
    """Pretty-print the Sharpe Integrity Audit report."""
    sep = "=" * 74
    dash = "-" * 74

    print(sep)
    print(f"  SHARPE INTEGRITY AUDIT — run_id: {report.get('run_id', 'N/A')}")
    print(f"  Observations: {report.get('n_observations', 'N/A')}")
    print(sep)

    # ── Phase 1: Autocorrelation ─────────────────────────────────────
    ac = report.get("autocorrelation_audit", {})
    print(f"\n  PHASE 1: AUTOCORRELATION AUDIT")
    print(dash)
    for label in ["core", "vrp", "combined"]:
        entry = ac.get(label, {})
        sig = "⚠ SIGNIFICANT" if entry.get("significant_autocorrelation") else "  ok"
        naive = entry.get("naive_sharpe", "N/A")
        nw5 = entry.get("nw_sharpe_lag5", "N/A")
        acfs = entry.get("autocorrelations", {})
        ac_str = "  ".join(f"L{k[-1]}={v:+.3f}" for k, v in acfs.items())
        print(f"    {label:>10s}: [{sig}]  naive={naive}  NW(5)={nw5}")
        print(f"               {ac_str}")

    # ── Phase 2: Capital Utilization ─────────────────────────────────
    util = report.get("capital_utilization", {})
    print(f"\n  PHASE 2: CAPITAL UTILIZATION")
    print(dash)
    flag = "⚠ UNDER-DEPLOYED" if util.get("under_deployed_flag") else "  ok"
    print(f"    [{flag}]  Portfolio vol = {util.get('ann_vol_portfolio', 'N/A')}")
    print(f"    Mean w_vrp = {util.get('mean_w_vrp', 'N/A')}")

    # ── Phase 3: Turnover & Costs ────────────────────────────────────
    cost = report.get("turnover_cost_sensitivity", {})
    print(f"\n  PHASE 3: TURNOVER & COST SENSITIVITY")
    print(dash)
    print(f"    Ann turnover:     {cost.get('annualized_turnover', 'N/A')}")
    print(f"    Avg holding:      {cost.get('avg_holding_period_days', 'N/A')} days")
    print(f"    Baseline Sharpe:  {cost.get('baseline_sharpe', 'N/A')}")
    for label, sc in cost.get("cost_scenarios", {}).items():
        print(f"    {label:>8s} Sharpe:  {sc.get('sharpe', 'N/A')}")

    # ── Phase 4: Bootstrap CI ────────────────────────────────────────
    boot = report.get("bootstrap_confidence", {})
    print(f"\n  PHASE 4: BOOTSTRAP CONFIDENCE INTERVAL (block=5d, n=1000)")
    print(dash)
    for label in ["core", "vrp", "combined"]:
        b = boot.get(label, {})
        print(f"    {label:>10s}: Sharpe={b.get('mean_sharpe', 'N/A')}  "
              f"[{b.get('ci_5', 'N/A')}, {b.get('ci_95', 'N/A')}] (90%)")

    # ── Phase 5: Rolling Stability ───────────────────────────────────
    roll = report.get("rolling_stability", {})
    print(f"\n  PHASE 5: ROLLING STABILITY")
    print(dash)
    for wk, wd in roll.items():
        print(f"    Window: {wk}")
        for label, stats in wd.items():
            flag = " ⚠" if stats.get("unstable") else ""
            if stats.get("mean") is not None:
                print(f"      {label:>10s}: mean={stats['mean']:>7.3f}  "
                      f"std={stats['std']:>6.3f}  "
                      f"[{stats['min']:>7.3f}, {stats['max']:>7.3f}]"
                      f"  neg={stats['pct_negative']:>4.1f}%{flag}")
            else:
                print(f"      {label:>10s}: insufficient data")

    # ── Phase 6: PnL Concentration ───────────────────────────────────
    conc = report.get("pnl_concentration", {})
    print(f"\n  PHASE 6: PnL CONCENTRATION")
    print(dash)
    for label in ["core", "vrp", "combined"]:
        c = conc.get(label, {})
        flag = " ⚠ CONCENTRATED" if c.get("concentrated") else ""
        print(f"    {label:>10s}: top5={c.get('top5_pct', 'N/A')}%  "
              f"top10={c.get('top10_pct', 'N/A')}%{flag}")

    # ── Phase 7: OOS Segmentation ────────────────────────────────────
    oos = report.get("oos_segmentation", {})
    print(f"\n  PHASE 7: OUT-OF-SAMPLE SEGMENTATION "
          f"(n1={oos.get('n_first', '?')}, n2={oos.get('n_second', '?')})")
    print(dash)
    for label in ["core", "vrp", "combined"]:
        o = oos.get(label, {})
        flag = " ⚠ REGIME-DEPENDENT" if o.get("regime_dependent") else ""
        print(f"    {label:>10s}: H1={o.get('sharpe_first_half', 'N/A')}  "
              f"H2={o.get('sharpe_second_half', 'N/A')}  "
              f"ratio={o.get('second_to_first_ratio', 'N/A')}{flag}")

    # ── Phase 8: Final Diagnosis ─────────────────────────────────────
    diag = report.get("integrity_diagnosis", {})
    print(f"\n  PHASE 8: FINAL DIAGNOSIS")
    print(sep)
    classification = diag.get("classification", "UNKNOWN")
    print(f"    ╔══════════════════════════════════════════════╗")
    print(f"    ║  CLASSIFICATION: {classification:<29s}║")
    print(f"    ╚══════════════════════════════════════════════╝")
    flags = diag.get("flags", [])
    if flags:
        print(f"    Flags: {', '.join(flags)}")
    for k, v in diag.get("details", {}).items():
        print(f"    • {v}")

    print(sep)
