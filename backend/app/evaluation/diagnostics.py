"""
Performance diagnostics — rolling risk/return metrics for the VRP sleeve.

Produces time-series CSVs of:
- Rolling Sharpe (63d)
- Rolling Sortino (63d)
- Rolling ES95
- Rolling worst scenario loss
- Rolling forecast calibration error
- Rolling short convexity score
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# Rolling metrics
# ═══════════════════════════════════════════════════════════════════════════

def rolling_sharpe(pnl: pd.Series, window: int = 63, rf_daily: float = 0.0) -> pd.Series:
    """Annualised rolling Sharpe ratio."""
    excess = pnl - rf_daily
    mu = excess.rolling(window, min_periods=window // 2).mean()
    sigma = pnl.rolling(window, min_periods=window // 2).std()
    return (mu / sigma.replace(0, np.nan)) * np.sqrt(252)


def rolling_sortino(pnl: pd.Series, window: int = 63, rf_daily: float = 0.0) -> pd.Series:
    """Annualised rolling Sortino ratio (downside deviation)."""
    excess = pnl - rf_daily
    mu = excess.rolling(window, min_periods=window // 2).mean()
    downside = pnl.clip(upper=0)
    dd = downside.rolling(window, min_periods=window // 2).apply(
        lambda x: np.sqrt(np.mean(x ** 2)), raw=True,
    )
    return (mu / dd.replace(0, np.nan)) * np.sqrt(252)


def rolling_es(pnl: pd.Series, window: int = 63, level: float = 0.95) -> pd.Series:
    """Rolling Expected Shortfall at the given confidence level."""
    def _es(x: np.ndarray) -> float:
        sorted_x = np.sort(x)
        cutoff = max(int(len(sorted_x) * (1 - level)), 1)
        return float(np.mean(sorted_x[:cutoff]))

    return pnl.rolling(window, min_periods=window // 2).apply(_es, raw=True)


def rolling_worst_scenario_loss(
    scenario_losses: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Rolling worst (minimum) scenario loss over window."""
    return scenario_losses.rolling(window, min_periods=window // 2).min()


def rolling_forecast_calibration_error(
    forecast_rv: pd.Series,
    realised_rv: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Rolling MAE between forecast and realised RV."""
    error = (forecast_rv - realised_rv).abs()
    return error.rolling(window, min_periods=window // 2).mean()


def rolling_convexity_score(
    convexity_scores: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Rolling average short convexity score."""
    return convexity_scores.rolling(window, min_periods=window // 2).mean()


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostics report
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DiagnosticsInput:
    """Input container for diagnostics computation."""
    daily_pnl: pd.Series
    scenario_losses: pd.Series
    forecast_rv: pd.Series
    realised_rv: pd.Series
    convexity_scores: pd.Series
    window: int = 63
    rf_daily: float = 0.0


def compute_diagnostics(inp: DiagnosticsInput) -> pd.DataFrame:
    """Compute all rolling diagnostics and return as a single DataFrame."""
    w = inp.window
    rf = inp.rf_daily

    df = pd.DataFrame(index=inp.daily_pnl.index)
    df["rolling_sharpe"] = rolling_sharpe(inp.daily_pnl, w, rf)
    df["rolling_sortino"] = rolling_sortino(inp.daily_pnl, w, rf)
    df["rolling_es95"] = rolling_es(inp.daily_pnl, w, 0.95)
    df["rolling_worst_scenario"] = rolling_worst_scenario_loss(inp.scenario_losses, w)
    df["rolling_forecast_mae"] = rolling_forecast_calibration_error(
        inp.forecast_rv, inp.realised_rv, w,
    )
    df["rolling_convexity"] = rolling_convexity_score(inp.convexity_scores, w)

    return df


def save_diagnostics(
    diagnostics: pd.DataFrame,
    root: Path,
    run_id: str = "default",
) -> Path:
    """Save diagnostics DataFrame to CSV."""
    out_dir = root / "vrp_diagnostics" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "rolling_metrics.csv"
    diagnostics.to_csv(path)
    return path
