"""
Validated backtest integration for Sharpe analysis.

Extends sharpe_report.py with strict data validation, weight timing
checks, portfolio reconstruction verification, ES95, and economic
interpretation diagnostics.

Use ``generate_validated_report()`` as the entry-point.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.analysis.sharpe_report import (
    REGIME_LABELS,
    TRADING_DAYS_PER_YEAR,
    MIN_OBS,
    _ann_return,
    _ann_vol,
    _max_drawdown,
    _summarize_rolling,
    compute_diagnostics,
    compute_regime_sharpe,
    compute_rolling_sharpe,
    compute_sharpe,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Input Data Validation
# ═══════════════════════════════════════════════════════════════════════════

class DataValidationError(Exception):
    """Raised when input data fails validation checks."""


def validate_and_load(
    daily_data: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """Strict validation, NAV reconstruction, and return computation.

    Checks:
      1) date sorted ascending
      2) no duplicate dates
      3) no NaN in core_pnl, vrp_pnl, w_vrp
      4) NAV identity: nav[t] == nav[t-1] + pnl[t]

    Returns enriched DataFrame with core_ret, vrp_ret, port_ret.
    """
    required = {"date", "core_pnl", "vrp_pnl", "w_vrp"}
    missing = required - set(daily_data.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    df = daily_data.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ── 1. Sort check ────────────────────────────────────────────────
    if not df["date"].is_monotonic_increasing:
        df = df.sort_values("date").reset_index(drop=True)
        logger.warning("Input dates were not sorted — sorted ascending.")

    # ── 2. Duplicate date check ──────────────────────────────────────
    dups = df["date"].duplicated()
    if dups.any():
        n_dups = int(dups.sum())
        raise DataValidationError(
            f"Found {n_dups} duplicate date(s). Dates must be unique."
        )

    df = df.reset_index(drop=True)

    # ── 3. NaN check on required columns ─────────────────────────────
    for col in ["core_pnl", "vrp_pnl", "w_vrp"]:
        n_nan = int(df[col].isna().sum())
        if n_nan > 0:
            raise DataValidationError(
                f"Column '{col}' has {n_nan} NaN values. "
                f"All PnL and weight values must be present."
            )

    # ── 4. NAV reconstruction / verification ─────────────────────────
    nav_was_provided = {"core_nav": "core_nav" in df.columns,
                        "vrp_nav": "vrp_nav" in df.columns}

    if not nav_was_provided["core_nav"]:
        df["core_nav"] = initial_capital + df["core_pnl"].cumsum()
    if not nav_was_provided["vrp_nav"]:
        df["vrp_nav"] = initial_capital + df["vrp_pnl"].cumsum()

    # Verify NAV identity for rows t >= 1:
    #   nav[t] == nav[t-1] + pnl[t]
    for nav_col, pnl_col, label in [
        ("core_nav", "core_pnl", "core"),
        ("vrp_nav", "vrp_pnl", "vrp"),
    ]:
        if nav_was_provided[nav_col]:
            expected = df[nav_col].shift(1).iloc[1:] + df[pnl_col].iloc[1:]
            actual = df[nav_col].iloc[1:]
            max_dev = float(np.abs(expected.values - actual.values).max())
            if max_dev > 1e-4:
                raise DataValidationError(
                    f"{label} NAV identity violated: max deviation = {max_dev:.8f}. "
                    f"Expected nav[t] == nav[t-1] + pnl[t]."
                )

    # ── Compute returns using lagged NAV ─────────────────────────────
    core_nav_prev = df["core_nav"].shift(1)
    vrp_nav_prev = df["vrp_nav"].shift(1)
    core_nav_prev.iloc[0] = initial_capital
    vrp_nav_prev.iloc[0] = initial_capital

    df["core_ret"] = df["core_pnl"] / core_nav_prev
    df["vrp_ret"] = df["vrp_pnl"] / vrp_nav_prev

    # Guard Inf
    df["core_ret"] = df["core_ret"].replace([np.inf, -np.inf], np.nan)
    df["vrp_ret"] = df["vrp_ret"].replace([np.inf, -np.inf], np.nan)

    # Combined portfolio return (using today's weight — no lookahead)
    df["port_ret"] = (1 - df["w_vrp"]) * df["core_ret"] + df["w_vrp"] * df["vrp_ret"]

    # Regime normalisation
    if "regime" in df.columns:
        df["regime"] = df["regime"].astype(str).str.lower().str.strip()
    else:
        df["regime"] = np.nan

    # Drop any NaN return rows (only from lagged computation edge)
    df = df.dropna(subset=["core_ret", "vrp_ret", "port_ret"]).reset_index(drop=True)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — Weight Timing Check
# ═══════════════════════════════════════════════════════════════════════════

def check_weight_timing(
    df: pd.DataFrame,
    rf_annual: float = 0.0,
    sharpe_threshold_pct: float = 5.0,
) -> Dict[str, Any]:
    """Compare portfolio using w_vrp[t] vs w_vrp[t-1].

    If the lagged version materially changes Sharpe (> threshold %),
    flags a potential weight-timing misalignment.

    Returns
    -------
    dict with weight_timing_ok, sharpe_current, sharpe_lagged,
    sharpe_pct_diff, recommendation.
    """
    core_ret = df["core_ret"]
    vrp_ret = df["vrp_ret"]
    w_current = df["w_vrp"]

    # Build lagged-weight portfolio (from index 1 onward)
    w_lagged = df["w_vrp"].shift(1)
    # First row: no prior weight → use 0 (conservative)
    w_lagged.iloc[0] = 0.0

    port_ret_current = (1 - w_current) * core_ret + w_current * vrp_ret
    port_ret_lagged = (1 - w_lagged) * core_ret + w_lagged * vrp_ret

    n = len(port_ret_current)
    if n < MIN_OBS:
        return {
            "weight_timing_ok": True,
            "note": f"Insufficient data ({n} < {MIN_OBS}), skipping timing check.",
        }

    sharpe_current = compute_sharpe(port_ret_current, rf_annual)
    sharpe_lagged = compute_sharpe(port_ret_lagged, rf_annual)

    if abs(sharpe_current) < 1e-8:
        pct_diff = 0.0
    else:
        pct_diff = abs(sharpe_lagged - sharpe_current) / abs(sharpe_current) * 100

    corr = float(port_ret_current.corr(port_ret_lagged))
    mean_diff = float((port_ret_current - port_ret_lagged).mean())

    timing_ok = pct_diff <= sharpe_threshold_pct
    recommendation = "current" if timing_ok else "lagged"

    if not timing_ok:
        logger.warning(
            f"Weight timing mismatch detected: "
            f"Sharpe(current)={sharpe_current:.4f} vs "
            f"Sharpe(lagged)={sharpe_lagged:.4f} "
            f"({pct_diff:.1f}% diff > {sharpe_threshold_pct}% threshold). "
            f"Recommend using {recommendation} weights."
        )

    return {
        "weight_timing_ok": timing_ok,
        "sharpe_current_weights": round(sharpe_current, 6),
        "sharpe_lagged_weights": round(sharpe_lagged, 6),
        "sharpe_pct_diff": round(pct_diff, 4),
        "correlation": round(corr, 6),
        "mean_return_diff": round(mean_diff, 10),
        "recommendation": recommendation,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — Portfolio Reconstruction Check
# ═══════════════════════════════════════════════════════════════════════════

def check_portfolio_reconstruction(
    daily_data: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    rel_tol: float = 1e-6,
) -> Dict[str, Any]:
    """Compare reconstructed NAV from sleeves+weights to portfolio_nav.

    Only runs if 'portfolio_nav' column exists in daily_data.
    Computes sleeve returns internally — does not require port_ret.
    """
    if "portfolio_nav" not in daily_data.columns:
        return {
            "portfolio_reconstruction_ok": True,
            "note": "portfolio_nav column not present; reconstruction check skipped.",
        }

    df = daily_data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Reconstruct NAV from PnL
    if "core_nav" not in df.columns:
        df["core_nav"] = initial_capital + df["core_pnl"].cumsum()
    if "vrp_nav" not in df.columns:
        df["vrp_nav"] = initial_capital + df["vrp_pnl"].cumsum()

    # Compute returns
    core_nav_prev = df["core_nav"].shift(1)
    vrp_nav_prev = df["vrp_nav"].shift(1)
    core_nav_prev.iloc[0] = initial_capital
    vrp_nav_prev.iloc[0] = initial_capital

    core_ret = (df["core_pnl"] / core_nav_prev).replace([np.inf, -np.inf], 0.0)
    vrp_ret = (df["vrp_pnl"] / vrp_nav_prev).replace([np.inf, -np.inf], 0.0)
    port_ret = (1 - df["w_vrp"]) * core_ret + df["w_vrp"] * vrp_ret

    # Reconstruct portfolio NAV from weighted returns
    port_ret_vals = port_ret.values
    reconstructed_nav = np.empty(len(port_ret_vals))
    reconstructed_nav[0] = initial_capital * (1 + port_ret_vals[0])
    for i in range(1, len(port_ret_vals)):
        reconstructed_nav[i] = reconstructed_nav[i - 1] * (1 + port_ret_vals[i])

    actual_nav = df["portfolio_nav"].values
    abs_dev = np.abs(reconstructed_nav - actual_nav)
    max_abs_dev = float(abs_dev.max())

    # Relative deviation
    denom = np.maximum(np.abs(actual_nav), 1e-12)
    rel_dev = abs_dev / denom
    max_rel_dev = float(rel_dev.max())

    reconstruction_ok = max_rel_dev <= rel_tol

    if not reconstruction_ok:
        raise DataValidationError(
            f"Portfolio reconstruction failed: "
            f"max relative deviation = {max_rel_dev:.10f} > tolerance {rel_tol}. "
            f"Max absolute deviation = {max_abs_dev:.6f}."
        )

    return {
        "portfolio_reconstruction_ok": reconstruction_ok,
        "max_absolute_deviation": round(max_abs_dev, 10),
        "max_relative_deviation": round(max_rel_dev, 10),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4 — Enhanced Sharpe + ES95
# ═══════════════════════════════════════════════════════════════════════════

def _compute_es(returns: pd.Series, level: float = 0.95) -> float:
    """Expected shortfall (CVaR) at the given confidence level.

    ES95 = mean of returns below the 5th percentile.
    """
    threshold = returns.quantile(1 - level)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return 0.0
    return float(tail.mean())


def compute_enhanced_sleeve_stats(
    returns: pd.Series,
    rf_annual: float = 0.0,
) -> Dict[str, Any]:
    """Full stats for a single sleeve: Sharpe, return, vol, ES95, max DD."""
    n = len(returns)
    stats: Dict[str, Any] = {"n_obs": n}

    if n >= MIN_OBS:
        stats["sharpe"] = round(compute_sharpe(returns, rf_annual), 6)
    else:
        stats["sharpe"] = None

    stats["ann_return"] = round(_ann_return(returns), 6)
    stats["ann_vol"] = round(_ann_vol(returns), 6)
    stats["max_drawdown"] = round(_max_drawdown(returns), 6)
    stats["es_95"] = round(_compute_es(returns, 0.95), 8)

    return stats


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5 — Diagnostic Metrics (extended)
# ═══════════════════════════════════════════════════════════════════════════

def compute_weight_diagnostics(df: pd.DataFrame) -> Dict[str, Any]:
    """Mean weight by regime + overall."""
    result: Dict[str, Any] = {
        "mean_w_vrp": round(float(df["w_vrp"].mean()), 6),
        "std_w_vrp": round(float(df["w_vrp"].std(ddof=1)), 6),
        "min_w_vrp": round(float(df["w_vrp"].min()), 6),
        "max_w_vrp": round(float(df["w_vrp"].max()), 6),
    }

    if "regime" in df.columns and not df["regime"].isna().all():
        by_regime: Dict[str, float] = {}
        for regime in REGIME_LABELS:
            mask = df["regime"] == regime
            sub = df.loc[mask, "w_vrp"]
            if len(sub) > 0:
                by_regime[regime] = round(float(sub.mean()), 6)
            else:
                by_regime[regime] = None
        result["mean_w_vrp_by_regime"] = by_regime

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6 — Economic Interpretation
# ═══════════════════════════════════════════════════════════════════════════

def classify_vrp_role(
    core_sharpe: float,
    vrp_sharpe: float,
    combined_sharpe: float,
    correlation: float,
    mean_w_vrp: float,
) -> Dict[str, Any]:
    """Produce a structured economic diagnosis of the VRP sleeve's role.

    Rules:
      - vrp_sharpe > 0 and correlation < 0.3 → diversifying alpha sleeve
      - vrp_sharpe ≈ 0   and correlation low → tail hedge / variance dampener
      - combined < core → capital inefficiency
      - combined > core → quantify % improvement
      - mean_w_vrp < 0.01 → under-deployment
    """
    diagnosis: Dict[str, Any] = {
        "classification": "unclassified",
        "flags": [],
        "details": {},
    }

    # ── Classification ───────────────────────────────────────────────
    # Check tail-hedge BEFORE diversifying: a tiny positive Sharpe (<0.3)
    # with low correlation is a variance dampener, not a true alpha source.
    if vrp_sharpe is not None and abs(vrp_sharpe) < 0.3 and correlation < 0.3:
        diagnosis["classification"] = "tail_hedge_variance_dampener"
        diagnosis["details"]["reason"] = (
            f"VRP Sharpe={vrp_sharpe:.4f} near zero with low correlation={correlation:.4f}. "
            f"Acts primarily as a variance dampener / tail hedge."
        )
    elif vrp_sharpe is not None and vrp_sharpe > 0 and correlation < 0.3:
        diagnosis["classification"] = "diversifying_alpha_sleeve"
        diagnosis["details"]["reason"] = (
            f"VRP Sharpe={vrp_sharpe:.4f} > 0 with correlation={correlation:.4f} < 0.3 "
            f"indicates an independent alpha source providing diversification."
        )
    elif vrp_sharpe is not None and vrp_sharpe > 0:
        diagnosis["classification"] = "correlated_alpha_sleeve"
        diagnosis["details"]["reason"] = (
            f"VRP Sharpe={vrp_sharpe:.4f} > 0 but correlation={correlation:.4f} >= 0.3. "
            f"Adds return but limited diversification."
        )

    # ── Flags ────────────────────────────────────────────────────────
    if combined_sharpe is not None and core_sharpe is not None:
        if combined_sharpe < core_sharpe:
            pct_drag = abs(core_sharpe - combined_sharpe) / abs(core_sharpe) * 100 \
                if abs(core_sharpe) > 1e-8 else 0.0
            diagnosis["flags"].append("CAPITAL_INEFFICIENCY")
            diagnosis["details"]["capital_inefficiency"] = (
                f"Combined Sharpe ({combined_sharpe:.4f}) < Core Sharpe ({core_sharpe:.4f}). "
                f"VRP sleeve is a {pct_drag:.1f}% drag on portfolio quality."
            )
        else:
            pct_improvement = (combined_sharpe - core_sharpe) / abs(core_sharpe) * 100 \
                if abs(core_sharpe) > 1e-8 else 0.0
            diagnosis["details"]["sharpe_improvement_pct"] = round(pct_improvement, 2)
            diagnosis["details"]["sharpe_improvement"] = (
                f"Combined Sharpe ({combined_sharpe:.4f}) > Core Sharpe ({core_sharpe:.4f}). "
                f"VRP adds {pct_improvement:.1f}% Sharpe improvement."
            )

    if mean_w_vrp < 0.01:
        diagnosis["flags"].append("UNDER_DEPLOYMENT")
        diagnosis["details"]["under_deployment"] = (
            f"Mean w_vrp={mean_w_vrp:.4f} < 1%. VRP sleeve is under-deployed."
        )

    return diagnosis


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7 — Top-level API
# ═══════════════════════════════════════════════════════════════════════════

def generate_validated_report(
    run_id: str,
    daily_data: pd.DataFrame,
    rf_annual: float = 0.0,
    initial_capital: float = 1_000_000.0,
) -> Dict[str, Any]:
    """Generate a fully validated Sharpe analysis report.

    Pipeline:
      1. Strict input validation
      2. Weight timing check
      3. Portfolio reconstruction check
      4. Enhanced Sharpe + ES95
      5. Full diagnostics
      6. Economic interpretation
      7. Assemble structured dict

    Parameters
    ----------
    run_id : str
        Run identifier.
    daily_data : pd.DataFrame
        Required: date, core_pnl, vrp_pnl, w_vrp.
        Optional: core_nav, vrp_nav, portfolio_nav, regime.
    rf_annual : float
        Annual risk-free rate.
    initial_capital : float
        Starting capital for NAV reconstruction.

    Returns
    -------
    dict with keys: core, vrp, combined, correlation,
    marginal_sharpe_contribution, information_ratio,
    diversification_benefit, weight_diagnostics, regime_breakdown,
    validation_checks, economic_diagnosis, rolling_sharpe_summary.
    """
    # ── Phase 1: Validate & load ─────────────────────────────────────
    df = validate_and_load(daily_data, initial_capital=initial_capital)
    logger.info(f"Validated {len(df)} daily observations for run '{run_id}'.")

    # ── Phase 2: Weight timing ───────────────────────────────────────
    timing = check_weight_timing(df, rf_annual=rf_annual)

    # ── Phase 3: Portfolio reconstruction ─────────────────────────────
    reconstruction = check_portfolio_reconstruction(
        daily_data,  # original, may have portfolio_nav
        initial_capital=initial_capital,
    )

    # ── Phase 4: Enhanced Sharpe + ES95 ───────────────────────────────
    core_stats = compute_enhanced_sleeve_stats(df["core_ret"], rf_annual)
    vrp_stats = compute_enhanced_sleeve_stats(df["vrp_ret"], rf_annual)
    combined_stats = compute_enhanced_sleeve_stats(df["port_ret"], rf_annual)

    # ── Phase 5: Diagnostics ─────────────────────────────────────────
    diagnostics = compute_diagnostics(df, rf_annual=rf_annual)
    weight_diag = compute_weight_diagnostics(df)
    regime_breakdown = compute_regime_sharpe(df, rf_annual=rf_annual)
    rolling = compute_rolling_sharpe(df, rf_annual=rf_annual)
    rolling_summary = _summarize_rolling(rolling)

    # ── Phase 6: Economic interpretation ─────────────────────────────
    economic = classify_vrp_role(
        core_sharpe=core_stats.get("sharpe", 0.0) or 0.0,
        vrp_sharpe=vrp_stats.get("sharpe", 0.0) or 0.0,
        combined_sharpe=combined_stats.get("sharpe", 0.0) or 0.0,
        correlation=diagnostics["correlation"],
        mean_w_vrp=weight_diag["mean_w_vrp"],
    )

    # ── Phase 7: Assemble report ─────────────────────────────────────
    report = {
        "run_id": run_id,
        "core": core_stats,
        "vrp": vrp_stats,
        "combined": combined_stats,
        "correlation": diagnostics["correlation"],
        "marginal_sharpe_contribution": diagnostics["marginal_sharpe_contribution"],
        "information_ratio": diagnostics["information_ratio"],
        "diversification_benefit": diagnostics["diversification_benefit"],
        "weight_diagnostics": weight_diag,
        "regime_breakdown": regime_breakdown,
        "validation_checks": {
            "weight_timing": timing,
            "portfolio_reconstruction": reconstruction,
        },
        "economic_diagnosis": economic,
        "rolling_sharpe_summary": rolling_summary,
    }

    return report


# ═══════════════════════════════════════════════════════════════════════════
# Console Printer
# ═══════════════════════════════════════════════════════════════════════════

def print_validated_report(report: Dict[str, Any]) -> None:
    """Pretty-print a validated Sharpe analysis report."""
    sep = "=" * 74
    dash = "-" * 74

    print(sep)
    print(f"  VALIDATED SHARPE REPORT — run_id: {report.get('run_id', 'N/A')}")
    print(sep)

    # ── Validation checks ────────────────────────────────────────────
    checks = report.get("validation_checks", {})
    timing = checks.get("weight_timing", {})
    recon = checks.get("portfolio_reconstruction", {})

    print(f"\n  VALIDATION CHECKS")
    print(dash)
    _tag = "PASS" if timing.get("weight_timing_ok", True) else "WARN"
    print(f"    [{_tag}]  Weight timing OK:  "
          f"current={timing.get('sharpe_current_weights', 'N/A')}  "
          f"lagged={timing.get('sharpe_lagged_weights', 'N/A')}  "
          f"diff={timing.get('sharpe_pct_diff', 'N/A')}%  "
          f"→ use {timing.get('recommendation', 'N/A')}")

    _tag2 = "PASS" if recon.get("portfolio_reconstruction_ok", True) else "FAIL"
    note = recon.get("note", "")
    if note:
        print(f"    [{_tag2}]  Portfolio reconstruction:  {note}")
    else:
        print(f"    [{_tag2}]  Portfolio reconstruction:  "
              f"max_rel_dev={recon.get('max_relative_deviation', 'N/A')}")

    # ── Per-sleeve summary ───────────────────────────────────────────
    for label in ["core", "vrp", "combined"]:
        s = report.get(label, {})
        print(f"\n  {label.upper()}")
        print(dash)
        sh = s.get("sharpe")
        sh_str = f"{sh:>10.4f}" if sh is not None else "      N/A "
        print(f"    Sharpe:          {sh_str}")
        print(f"    Ann Return:      {s.get('ann_return', 'N/A'):>10}")
        print(f"    Ann Vol:         {s.get('ann_vol', 'N/A'):>10}")
        print(f"    Max Drawdown:    {s.get('max_drawdown', 'N/A'):>10}")
        print(f"    ES-95:           {s.get('es_95', 'N/A'):>10}")
        print(f"    Observations:    {s.get('n_obs', 'N/A'):>10}")

    # ── Cross-sleeve diagnostics ─────────────────────────────────────
    print(f"\n  DIAGNOSTICS")
    print(dash)
    print(f"    Correlation (core, vrp):   {report.get('correlation', 'N/A')}")
    print(f"    Marginal Sharpe (VRP):     {report.get('marginal_sharpe_contribution', 'N/A')}")
    print(f"    Information Ratio:         {report.get('information_ratio', 'N/A')}")
    print(f"    Diversification Benefit:   {report.get('diversification_benefit', 'N/A')}")

    # ── Weight diagnostics ───────────────────────────────────────────
    wd = report.get("weight_diagnostics", {})
    print(f"\n  WEIGHT DIAGNOSTICS")
    print(dash)
    print(f"    Mean w_vrp:    {wd.get('mean_w_vrp', 'N/A')}")
    print(f"    Std w_vrp:     {wd.get('std_w_vrp', 'N/A')}")
    print(f"    Range:         [{wd.get('min_w_vrp', '?')}, {wd.get('max_w_vrp', '?')}]")
    by_regime = wd.get("mean_w_vrp_by_regime", {})
    if by_regime:
        for r, v in by_regime.items():
            v_str = f"{v:.4f}" if v is not None else "N/A"
            print(f"      {r:<16s}: {v_str}")

    # ── Regime breakdown ─────────────────────────────────────────────
    regime = report.get("regime_breakdown", {})
    if regime:
        print(f"\n  REGIME-CONDITIONED SHARPE")
        print(dash)
        for r_label, r_data in regime.items():
            n = r_data.get("n_obs", 0)
            print(f"    {r_label} (n={n}):")
            for sleeve in ["core", "vrp", "combined"]:
                sh = r_data.get(f"{sleeve}_sharpe", "N/A")
                mu = r_data.get(f"{sleeve}_mean_return", "N/A")
                vol = r_data.get(f"{sleeve}_vol", "N/A")
                sh_str = f"{sh:.4f}" if isinstance(sh, (int, float)) else str(sh)
                mu_str = f"{mu:.4f}" if isinstance(mu, (int, float)) else str(mu)
                vol_str = f"{vol:.4f}" if isinstance(vol, (int, float)) else str(vol)
                print(f"      {sleeve:>10s}: Sharpe={sh_str:>9s}  "
                      f"AnnRet={mu_str:>9s}  Vol={vol_str:>9s}")

    # ── Economic diagnosis ───────────────────────────────────────────
    econ = report.get("economic_diagnosis", {})
    if econ:
        print(f"\n  ECONOMIC DIAGNOSIS")
        print(dash)
        print(f"    Classification:  {econ.get('classification', 'N/A')}")
        flags = econ.get("flags", [])
        if flags:
            for f in flags:
                print(f"    ⚠  FLAG: {f}")
        details = econ.get("details", {})
        for k, v in details.items():
            if isinstance(v, str):
                print(f"    {v}")
            else:
                print(f"    {k}: {v}")

    # ── Rolling Sharpe summary ───────────────────────────────────────
    rolling = report.get("rolling_sharpe_summary", {})
    if rolling:
        print(f"\n  ROLLING SHARPE SUMMARY")
        print(dash)
        for window_key, window_data in rolling.items():
            print(f"    Window: {window_key}")
            for sleeve, stats in window_data.items():
                if stats.get("mean") is not None:
                    print(f"      {sleeve:>10s}: mean={stats['mean']:>8.4f}  "
                          f"min={stats['min']:>8.4f}  max={stats['max']:>8.4f}  "
                          f"std={stats['std']:>8.4f}")
                else:
                    print(f"      {sleeve:>10s}: insufficient data")

    print(sep)
