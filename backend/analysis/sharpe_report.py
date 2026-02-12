"""
Sharpe analysis module for core strategy, VRP sleeve, and combined portfolio.

Phases:
  1. Data contract — load / validate daily PnL, NAV, weights, regimes
  2. Sharpe calculation — annualised Sharpe with risk-free adjustment
  3. Diagnostics — correlation, MSC, IR, diversification benefit
  4. Regime-conditioned Sharpe
  5. Rolling Sharpe (63-day, 126-day)
  6. Structured report output
  7. Console printer
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
REGIME_LABELS = ["normal_carry", "caution", "crash_risk"]
MIN_OBS = 30


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Data Contract
# ═══════════════════════════════════════════════════════════════════════════

def load_daily_data(
    daily_data: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """Validate, clean, and enrich daily data with returns.

    Required columns: date, core_pnl, vrp_pnl, w_vrp
    Optional columns: core_nav, vrp_nav, regime

    Returns DataFrame with added columns:
      core_ret, vrp_ret, port_ret, core_nav, vrp_nav
    """
    required = {"date", "core_pnl", "vrp_pnl", "w_vrp"}
    missing = required - set(daily_data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = daily_data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ── Reconstruct NAV if not provided ──────────────────────────────
    if "core_nav" not in df.columns:
        df["core_nav"] = initial_capital + df["core_pnl"].cumsum()
    if "vrp_nav" not in df.columns:
        df["vrp_nav"] = initial_capital + df["vrp_pnl"].cumsum()

    # ── Compute daily returns using lagged NAV ───────────────────────
    core_nav_prev = df["core_nav"].shift(1)
    vrp_nav_prev = df["vrp_nav"].shift(1)

    # First row: use initial_capital as denominator
    core_nav_prev.iloc[0] = initial_capital
    vrp_nav_prev.iloc[0] = initial_capital

    df["core_ret"] = df["core_pnl"] / core_nav_prev
    df["vrp_ret"] = df["vrp_pnl"] / vrp_nav_prev

    # Guard against NaN / Inf from zero NAV
    df["core_ret"] = df["core_ret"].replace([np.inf, -np.inf], np.nan)
    df["vrp_ret"] = df["vrp_ret"].replace([np.inf, -np.inf], np.nan)

    # ── Combined portfolio return (no lookahead — w_vrp is today's) ──
    df["port_ret"] = (1 - df["w_vrp"]) * df["core_ret"] + df["w_vrp"] * df["vrp_ret"]

    # ── Regime label normalisation ───────────────────────────────────
    if "regime" in df.columns:
        df["regime"] = df["regime"].astype(str).str.lower().str.strip()
    else:
        df["regime"] = np.nan

    # Drop rows with NaN returns (should only be leading rows)
    df = df.dropna(subset=["core_ret", "vrp_ret", "port_ret"]).reset_index(drop=True)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — Sharpe Calculation
# ═══════════════════════════════════════════════════════════════════════════

def compute_sharpe(
    returns: pd.Series,
    rf_annual: float = 0.0,
) -> float:
    """Compute annualised Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    rf_annual : float
        Annual risk-free rate (default 0).

    Returns
    -------
    float
        Annualised Sharpe ratio.

    Raises
    ------
    ValueError
        If fewer than MIN_OBS observations.
    """
    if len(returns) < MIN_OBS:
        raise ValueError(
            f"Insufficient data for Sharpe: {len(returns)} < {MIN_OBS} observations"
        )

    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
    excess = returns - rf_daily
    mean_excess = float(excess.mean())
    vol = float(excess.std(ddof=1))

    if vol < 1e-12:
        return 0.0

    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * mean_excess / vol)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def _ann_return(returns: pd.Series) -> float:
    """Annualised arithmetic mean return."""
    return float(returns.mean() * TRADING_DAYS_PER_YEAR)


def _ann_vol(returns: pd.Series) -> float:
    """Annualised volatility (sample std)."""
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from cumulative return series."""
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    return float(dd.min())


def compute_diagnostics(
    df: pd.DataFrame,
    rf_annual: float = 0.0,
) -> Dict[str, Any]:
    """Compute correlation, MSC, IR, diversification benefit.

    Parameters
    ----------
    df : DataFrame
        Must contain core_ret, vrp_ret, port_ret, w_vrp.
    rf_annual : float
        Annual risk-free rate.
    """
    core_ret = df["core_ret"]
    vrp_ret = df["vrp_ret"]
    port_ret = df["port_ret"]

    # ── Per-sleeve stats ─────────────────────────────────────────────
    def _sleeve_stats(returns: pd.Series, name: str) -> Dict[str, Any]:
        sharpe = compute_sharpe(returns, rf_annual)
        return {
            "sharpe": round(sharpe, 6),
            "ann_return": round(_ann_return(returns), 6),
            "ann_vol": round(_ann_vol(returns), 6),
            "max_drawdown": round(_max_drawdown(returns), 6),
            "n_obs": len(returns),
        }

    core_stats = _sleeve_stats(core_ret, "core")
    vrp_stats = _sleeve_stats(vrp_ret, "vrp")
    combined_stats = _sleeve_stats(port_ret, "combined")

    # ── Correlation ──────────────────────────────────────────────────
    correlation = float(core_ret.corr(vrp_ret))

    # ── Marginal Sharpe Contribution ─────────────────────────────────
    msc = combined_stats["sharpe"] - core_stats["sharpe"]

    # ── Information Ratio (VRP vs Core) ──────────────────────────────
    diff = vrp_ret - core_ret
    diff_mean = float(diff.mean())
    diff_vol = float(diff.std(ddof=1))
    if diff_vol < 1e-12:
        ir = 0.0
    else:
        ir = float(np.sqrt(TRADING_DAYS_PER_YEAR) * diff_mean / diff_vol)

    # ── Diversification Benefit ──────────────────────────────────────
    mean_w = float(df["w_vrp"].mean())
    sigma_core = _ann_vol(core_ret)
    sigma_vrp = _ann_vol(vrp_ret)
    sigma_port = _ann_vol(port_ret)

    naive_vol = np.sqrt(
        (1 - mean_w) ** 2 * sigma_core ** 2
        + mean_w ** 2 * sigma_vrp ** 2
    )
    diversification_benefit = float(naive_vol - sigma_port)

    return {
        "core": core_stats,
        "vrp": vrp_stats,
        "combined": combined_stats,
        "correlation": round(correlation, 6),
        "marginal_sharpe_contribution": round(msc, 6),
        "information_ratio": round(ir, 6),
        "diversification_benefit": round(diversification_benefit, 6),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4 — Regime-Conditioned Sharpe
# ═══════════════════════════════════════════════════════════════════════════

def compute_regime_sharpe(
    df: pd.DataFrame,
    rf_annual: float = 0.0,
) -> Dict[str, Any]:
    """Compute Sharpe and descriptive stats per regime bucket.

    Returns dict keyed by regime label.
    """
    if "regime" not in df.columns or df["regime"].isna().all():
        return {}

    result: Dict[str, Any] = {}

    for regime in REGIME_LABELS:
        mask = df["regime"] == regime
        sub = df.loc[mask]
        n = len(sub)

        if n == 0:
            result[regime] = {"n_obs": 0, "sharpe": None, "mean_return": None, "vol": None}
            continue

        entry: Dict[str, Any] = {"n_obs": n}

        for label, col in [("core", "core_ret"), ("vrp", "vrp_ret"), ("combined", "port_ret")]:
            returns = sub[col]
            entry[f"{label}_mean_return"] = round(float(returns.mean() * TRADING_DAYS_PER_YEAR), 6)
            entry[f"{label}_vol"] = round(float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)), 6)

            if n >= MIN_OBS:
                entry[f"{label}_sharpe"] = round(compute_sharpe(returns, rf_annual), 6)
            else:
                entry[f"{label}_sharpe"] = None

        result[regime] = entry

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5 — Rolling Sharpe
# ═══════════════════════════════════════════════════════════════════════════

def _rolling_sharpe_series(
    returns: pd.Series,
    window: int,
    rf_annual: float = 0.0,
) -> pd.Series:
    """Compute rolling annualised Sharpe over a fixed window."""
    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
    excess = returns - rf_daily

    roll_mean = excess.rolling(window, min_periods=window).mean()
    roll_std = excess.rolling(window, min_periods=window).std(ddof=1)

    # Guard against zero vol
    roll_std = roll_std.replace(0, np.nan)
    return np.sqrt(TRADING_DAYS_PER_YEAR) * roll_mean / roll_std


def compute_rolling_sharpe(
    df: pd.DataFrame,
    rf_annual: float = 0.0,
    windows: Optional[List[int]] = None,
) -> Dict[str, pd.DataFrame]:
    """Compute rolling Sharpe for each sleeve at given windows.

    Returns dict keyed by window size, each value a DataFrame
    with columns: date, core, vrp, combined.
    """
    if windows is None:
        windows = [63, 126]

    result: Dict[str, pd.DataFrame] = {}

    for w in windows:
        rolling_df = pd.DataFrame({"date": df["date"]})
        for label, col in [("core", "core_ret"), ("vrp", "vrp_ret"), ("combined", "port_ret")]:
            rolling_df[label] = _rolling_sharpe_series(df[col], w, rf_annual).values

        result[f"{w}d"] = rolling_df

    return result


def _summarize_rolling(rolling: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Produce summary statistics from rolling Sharpe DataFrames."""
    summary: Dict[str, Any] = {}
    for window_key, rdf in rolling.items():
        window_summary: Dict[str, Any] = {}
        for col in ["core", "vrp", "combined"]:
            s = rdf[col].dropna()
            if len(s) == 0:
                window_summary[col] = {"mean": None, "min": None, "max": None, "std": None}
            else:
                window_summary[col] = {
                    "mean": round(float(s.mean()), 6),
                    "min": round(float(s.min()), 6),
                    "max": round(float(s.max()), 6),
                    "std": round(float(s.std(ddof=1)), 6),
                }
        summary[window_key] = window_summary
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6 — Top-level API
# ═══════════════════════════════════════════════════════════════════════════

def generate_sharpe_report(
    run_id: str,
    daily_data: pd.DataFrame,
    rf_annual: float = 0.0,
    initial_capital: float = 1_000_000.0,
) -> Dict[str, Any]:
    """Generate a full Sharpe analysis report.

    Parameters
    ----------
    run_id : str
        Identifier for this analysis run.
    daily_data : pd.DataFrame
        Must have: date, core_pnl, vrp_pnl, w_vrp.
        Optional: core_nav, vrp_nav, regime.
    rf_annual : float
        Annual risk-free rate.
    initial_capital : float
        Starting capital for NAV reconstruction.

    Returns
    -------
    dict
        Structured report with keys: run_id, core, vrp, combined,
        correlation, marginal_sharpe_contribution, information_ratio,
        diversification_benefit, regime_breakdown, rolling_sharpe_summary.
    """
    df = load_daily_data(daily_data, initial_capital=initial_capital)

    diagnostics = compute_diagnostics(df, rf_annual=rf_annual)
    regime_breakdown = compute_regime_sharpe(df, rf_annual=rf_annual)
    rolling = compute_rolling_sharpe(df, rf_annual=rf_annual)
    rolling_summary = _summarize_rolling(rolling)

    report = {
        "run_id": run_id,
        "core": diagnostics["core"],
        "vrp": diagnostics["vrp"],
        "combined": diagnostics["combined"],
        "correlation": diagnostics["correlation"],
        "marginal_sharpe_contribution": diagnostics["marginal_sharpe_contribution"],
        "information_ratio": diagnostics["information_ratio"],
        "diversification_benefit": diagnostics["diversification_benefit"],
        "regime_breakdown": regime_breakdown,
        "rolling_sharpe_summary": rolling_summary,
    }

    return report


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7 — Console Printer
# ═══════════════════════════════════════════════════════════════════════════

def print_report(report: Dict[str, Any]) -> None:
    """Pretty-print a Sharpe analysis report to the console."""
    sep = "=" * 70
    dash = "-" * 70

    print(sep)
    print(f"  SHARPE ANALYSIS REPORT — run_id: {report.get('run_id', 'N/A')}")
    print(sep)

    # ── Per-sleeve summary ───────────────────────────────────────────
    for label in ["core", "vrp", "combined"]:
        s = report.get(label, {})
        print(f"\n  {label.upper()}")
        print(dash)
        print(f"    Sharpe:          {s.get('sharpe', 'N/A'):>10}")
        print(f"    Ann Return:      {s.get('ann_return', 'N/A'):>10}")
        print(f"    Ann Vol:         {s.get('ann_vol', 'N/A'):>10}")
        print(f"    Max Drawdown:    {s.get('max_drawdown', 'N/A'):>10}")
        print(f"    Observations:    {s.get('n_obs', 'N/A'):>10}")

    # ── Cross-sleeve diagnostics ─────────────────────────────────────
    print(f"\n  DIAGNOSTICS")
    print(dash)
    print(f"    Correlation (core, vrp):   {report.get('correlation', 'N/A')}")
    print(f"    Marginal Sharpe (VRP):     {report.get('marginal_sharpe_contribution', 'N/A')}")
    print(f"    Information Ratio:         {report.get('information_ratio', 'N/A')}")
    print(f"    Diversification Benefit:   {report.get('diversification_benefit', 'N/A')}")

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
