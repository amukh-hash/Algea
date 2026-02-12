#!/usr/bin/env python
"""
Selector portfolio backtest runner.

Runs a non-overlapping 10-day rebalance backtest with:
  - Buffer-zone turnover controls (hysteresis, slot cap, hold bonus)
  - Transaction-cost model (commission + impact)
  - Volatility scaling (leverage-capped)
  - Optional walk-forward evaluation with validation tuning

Usage examples
--------------
# Simple backtest from scored parquet
python backend/scripts/run_selector_portfolio.py \\
    --scored-parquet backend/data/selector/runs/SEL-.../scored_test.parquet \\
    --output-dir /tmp/port_eval

# Walk-forward with tuning
python backend/scripts/run_selector_portfolio.py \\
    --scored-parquet backend/data/selector/runs/SEL-.../scored_val.parquet \\
    --scored-test-parquet backend/data/selector/runs/SEL-.../scored_test.parquet \\
    --tune-on-val --output-dir /tmp/port_wf
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Add project root to path ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algaie.portfolio.portfolio_rules import PortfolioConfig, construct_portfolio
from algaie.portfolio.cost_model import CostConfig, compute_turnover_and_cost
from algaie.portfolio.vol_scaling import VolTargetConfig, compute_leverage, apply_leverage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
# Core backtest engine
# ═════════════════════════════════════════════════════════════════════════

def build_rebalance_schedule(
    dates: List,
    horizon: int = 10,
) -> List:
    """Select non-overlapping rebalance dates spaced by *horizon* trading days.

    Parameters
    ----------
    dates : list
        Sorted unique trading dates from the scored DataFrame.
    horizon : int
        Spacing between rebalance dates.

    Returns
    -------
    list of dates
    """
    return list(dates[::horizon])


def run_backtest(
    scored_df: pd.DataFrame,
    port_cfg: PortfolioConfig,
    cost_cfg: CostConfig,
    vol_cfg: Optional[VolTargetConfig] = None,
    score_col: str = "score_final",
    target_col: str = "y_ret",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run a single-pass portfolio backtest.

    Parameters
    ----------
    scored_df : DataFrame
        Must have: date, symbol, score_col, target_col.
    port_cfg, cost_cfg, vol_cfg : configs
    score_col, target_col : str

    Returns
    -------
    returns_df : DataFrame
        Per-period: date, gross_ret, net_ret, lev, turnover, cost, n_holdings, ...
    summary : dict
        Aggregate metrics.
    """
    dates = sorted(scored_df["date"].unique())
    rebal_dates = build_rebalance_schedule(dates, port_cfg.rebalance_horizon_days)
    logger.info(f"  Rebalance schedule: {len(rebal_dates)} periods from {len(dates)} dates "
                f"(horizon={port_cfg.rebalance_horizon_days}d)")

    if len(rebal_dates) < 2:
        logger.warning("  Not enough rebalance dates for a backtest")
        return pd.DataFrame(), {}

    periods_per_year = 252.0 / port_cfg.rebalance_horizon_days

    prev_weights: Dict[str, float] = {}
    hold_ages: Dict[str, int] = {}
    records = []
    gross_history = []  # for vol scaling

    for i, dt in enumerate(rebal_dates):
        day_df = scored_df[scored_df["date"] == dt].copy()
        if day_df.empty:
            continue

        # ── Construct portfolio ───────────────────────────────────────
        weights, hold_ages, diag = construct_portfolio(
            day_df, prev_weights if prev_weights else None, port_cfg, hold_ages,
        )

        if not weights:
            continue

        # ── Gross return for this period ──────────────────────────────
        ret_map = dict(zip(day_df["symbol"], day_df[target_col]))
        gross_ret = sum(
            weights.get(s, 0.0) * ret_map.get(s, 0.0)
            for s in weights
            if s in ret_map
        )

        # Count missing returns
        n_missing = sum(1 for s in weights if s not in ret_map or np.isnan(ret_map.get(s, np.nan)))
        if n_missing > 0:
            logger.debug(f"    {dt}: {n_missing} symbols missing y_ret, reweighted")

        # ── Compute raw turnover & cost (pure — no returns arg) ────────
        turnover, raw_cost = compute_turnover_and_cost(prev_weights, weights, cost_cfg)

        # ── Vol scaling: leverage scales BOTH returns AND costs ───────
        lev = 1.0
        if vol_cfg is not None and vol_cfg.target_vol_ann > 0:
            gross_arr = np.array(gross_history) if gross_history else np.array([])
            if len(gross_arr) >= vol_cfg.lookback_periods:
                lev = compute_leverage(gross_arr, len(gross_arr), vol_cfg, periods_per_year)

        # Unscaled series (notional = 1.0)
        net_ret_unscaled = gross_ret - raw_cost

        # Vol-scaled series (leverage-coupled)
        gross_scaled = gross_ret * lev
        cost_scaled = raw_cost * lev
        net_ret_scaled = gross_scaled - cost_scaled

        gross_history.append(gross_ret)

        records.append({
            "date": dt,
            "gross_ret": gross_ret,
            "net_ret": net_ret_unscaled,
            "scaled_net_ret": net_ret_scaled,
            "scaled_gross_ret": gross_scaled,
            "lev": lev,
            "turnover": turnover,
            "cost": raw_cost,
            "cost_scaled": cost_scaled,
            "n_holdings": diag["n_portfolio"],
            "n_new": diag["n_new"],
            "n_exits": diag["n_exits"],
            "n_held": diag["n_held"],
        })

        prev_weights = weights

    if not records:
        return pd.DataFrame(), {}

    returns_df = pd.DataFrame(records)

    # ── Compute summary metrics ───────────────────────────────────────
    summary = _compute_metrics(returns_df, periods_per_year, vol_cfg is not None)
    summary["n_periods"] = len(returns_df)
    summary["periods_per_year"] = periods_per_year
    summary["rebalance_horizon"] = port_cfg.rebalance_horizon_days

    return returns_df, summary


def _compute_metrics(
    df: pd.DataFrame,
    ppy: float,
    has_vol_scaling: bool = False,
) -> Dict[str, Any]:
    """Compute aggregate portfolio metrics from period returns.

    Produces two sets of net metrics:

    * **Unscaled** (notional = 1.0) — suffixed ``_unscaled``.
    * **Vol-scaled** (leverage-coupled) — suffixed ``_volscaled`` (only
      when ``has_vol_scaling`` is True).

    For backward compatibility, the bare key ``net_sharpe`` is always an
    alias for ``net_sharpe_unscaled``.
    """
    n = len(df)
    if n < 2:
        return {}

    gross = df["gross_ret"].values
    net = df["net_ret"].values  # unscaled net

    def _sharpe(r, ppy):
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1))
        return round(mu / max(sigma, 1e-8) * np.sqrt(ppy), 4)

    def _cagr(total_log_ret, years):
        if years <= 0:
            return 0.0
        return float((np.exp(total_log_ret) ** (1 / max(years, 0.01))) - 1)

    years = n / ppy

    # ── Gross metrics ─────────────────────────────────────────────────
    ann_ret_g = float(np.mean(gross) * ppy)
    ann_vol_g = float(np.std(gross, ddof=1) * np.sqrt(ppy))
    cagr_g = _cagr(float(np.sum(gross)), years)

    # ── Unscaled net metrics (notional = 1.0) ─────────────────────────
    ann_ret_n = float(np.mean(net) * ppy)
    ann_vol_n = float(np.std(net, ddof=1) * np.sqrt(ppy))
    cagr_n = _cagr(float(np.sum(net)), years)
    sharpe_unscaled = _sharpe(net, ppy)

    # Max drawdown (unscaled)
    cum_net = np.cumsum(net)
    running_max = np.maximum.accumulate(cum_net)
    max_dd = float(np.min(cum_net - running_max))

    # Turnover / cost
    avg_to = float(df["turnover"].mean())
    avg_cost = float(df["cost"].mean())

    # Avg hold (approx)
    avg_replace_frac = float(df["n_new"].mean()) / max(float(df["n_holdings"].mean()), 1)
    avg_hold_periods = 1.0 / max(avg_replace_frac, 1e-8) if avg_replace_frac > 0.01 else 999.0
    avg_hold_days = avg_hold_periods * (252.0 / ppy)

    # Leverage / exposure stats
    lev_arr = df["lev"].values
    leverage_mean = float(np.mean(lev_arr))
    leverage_std = float(np.std(lev_arr, ddof=1)) if n > 1 else 0.0
    effective_cost_ann = round(avg_cost * ppy, 6)

    result = {
        # Gross
        "gross_sharpe": _sharpe(gross, ppy),
        "ann_return_gross": round(ann_ret_g, 6),
        "ann_vol_gross": round(ann_vol_g, 6),
        "cagr_gross": round(cagr_g, 6),
        # Unscaled net (explicit)
        "net_sharpe_unscaled": sharpe_unscaled,
        "ann_return_net_unscaled": round(ann_ret_n, 6),
        "ann_vol_net_unscaled": round(ann_vol_n, 6),
        "cagr_net_unscaled": round(cagr_n, 6),
        # Backward-compat aliases (= unscaled)
        "net_sharpe": sharpe_unscaled,
        "ann_return_net": round(ann_ret_n, 6),
        "ann_vol_net": round(ann_vol_n, 6),
        "cagr_net": round(cagr_n, 6),
        # Drawdown / turnover / cost
        "max_drawdown": round(max_dd, 6),
        "avg_turnover_1way": round(avg_to, 6),
        "avg_cost_per_period": round(avg_cost, 6),
        "avg_hold_days": round(min(avg_hold_days, 999.0), 1),
        # Leverage / exposure
        "leverage_mean": round(leverage_mean, 4),
        "leverage_std": round(leverage_std, 4),
        "effective_cost_ann": effective_cost_ann,
        "gross_exposure_mean": round(leverage_mean, 4),
        "rebalance_count": n,
    }

    # ── Vol-scaled net metrics (only when vol targeting enabled) ───────
    if has_vol_scaling:
        scaled = df["scaled_net_ret"].values
        ann_ret_vs = float(np.mean(scaled) * ppy)
        ann_vol_vs = float(np.std(scaled, ddof=1) * np.sqrt(ppy))
        cagr_vs = _cagr(float(np.sum(scaled)), years)
        sharpe_vs = _sharpe(scaled, ppy)

        result["net_sharpe_volscaled"] = sharpe_vs
        result["ann_return_net_volscaled"] = round(ann_ret_vs, 6)
        result["ann_vol_net_volscaled"] = round(ann_vol_vs, 6)
        result["cagr_net_volscaled"] = round(cagr_vs, 6)
        # Legacy alias
        result["vol_scaled_sharpe"] = sharpe_vs
        result["vol_scaled_ann_vol"] = round(ann_vol_vs, 6)

    return result


# ═════════════════════════════════════════════════════════════════════════
# Validation tuning
# ═════════════════════════════════════════════════════════════════════════

def tune_on_val(
    scored_val: pd.DataFrame,
    base_port_cfg: PortfolioConfig,
    cost_cfg: CostConfig,
    vol_cfg: Optional[VolTargetConfig],
    score_col: str = "score_final",
    target_col: str = "y_ret",
    turnover_penalty: float = 0.0,
) -> Tuple[PortfolioConfig, VolTargetConfig, Dict]:
    """Grid-search turnover-control hyperparams on validation data.

    Maximizes net Sharpe subject to avg turnover ≤ turnover_limit.
    Optional turnover_penalty (Part 5) adds continuous penalty term.

    Returns
    -------
    best_port_cfg, best_vol_cfg, tuning_log
    """
    logger.info("\n" + "═" * 60)
    logger.info("VALIDATION TUNING: searching turnover-control params")
    logger.info("═" * 60)

    # Define search grid
    grid = {
        "buffer_entry_rank": [30, 35, 40, 45],
        "buffer_exit_rank": [60, 70, 80, 90],
        "max_replacements": [None, 5, 10, 15],
        "hold_bonus": [0.0, 0.1, 0.2],
        "target_vol": [0.10, 0.15, 0.20],
    }

    # Generate valid combinations (exit must be > entry + 10)
    combos = []
    for entry, exit_, max_r, hb, tv in itertools.product(
        grid["buffer_entry_rank"],
        grid["buffer_exit_rank"],
        grid["max_replacements"],
        grid["hold_bonus"],
        grid["target_vol"],
    ):
        if exit_ <= entry + 10:
            continue
        combos.append((entry, exit_, max_r, hb, tv))

    logger.info(f"  Grid size: {len(combos)} valid combinations")

    best_sharpe = -999.0
    best_combo = None
    best_summary = None
    results = []

    for i, (entry, exit_, max_r, hb, tv) in enumerate(combos):
        cfg = PortfolioConfig(
            top_k=base_port_cfg.top_k,
            long_only=base_port_cfg.long_only,
            weight_scheme=base_port_cfg.weight_scheme,
            rebalance_horizon_days=base_port_cfg.rebalance_horizon_days,
            buffer_entry_rank=entry,
            buffer_exit_rank=exit_,
            max_replacements=max_r,
            hold_bonus=hb,
            turnover_limit=base_port_cfg.turnover_limit,
        )

        vcfg = VolTargetConfig(
            target_vol_ann=tv,
            lookback_periods=vol_cfg.lookback_periods if vol_cfg else 12,
            max_leverage=vol_cfg.max_leverage if vol_cfg else 1.0,
            min_leverage=vol_cfg.min_leverage if vol_cfg else 0.0,
        ) if vol_cfg else None

        _, summary = run_backtest(
            scored_val, cfg, cost_cfg, vcfg, score_col, target_col,
        )

        if not summary:
            continue

        sharpe = summary.get("net_sharpe", -999.0)
        to = summary.get("avg_turnover_1way", 999.0)
        penalized = sharpe

        # Penalize if turnover exceeds limit
        if to > base_port_cfg.turnover_limit:
            penalized -= 10.0 * (to - base_port_cfg.turnover_limit)

        # Part 5: continuous turnover penalty
        penalized -= turnover_penalty * to

        results.append({
            "entry": entry, "exit": exit_, "max_r": max_r,
            "hb": hb, "tv": tv,
            "net_sharpe": summary.get("net_sharpe", 0),
            "turnover": to,
            "penalized_sharpe": round(penalized, 4),
        })

        if penalized > best_sharpe:
            best_sharpe = penalized
            best_combo = (entry, exit_, max_r, hb, tv)
            best_summary = summary

    if best_combo is None:
        logger.warning("  Tuning failed — no valid combos. Using defaults.")
        return base_port_cfg, vol_cfg, {"status": "failed"}

    entry, exit_, max_r, hb, tv = best_combo
    logger.info(f"\n  Best combo: entry={entry} exit={exit_} max_r={max_r} "
                f"hb={hb} tv={tv}")
    logger.info(f"  Net Sharpe: {best_summary['net_sharpe']:.4f}  "
                f"Turnover: {best_summary['avg_turnover_1way']:.4f}")

    best_port_cfg = PortfolioConfig(
        top_k=base_port_cfg.top_k,
        long_only=base_port_cfg.long_only,
        weight_scheme=base_port_cfg.weight_scheme,
        rebalance_horizon_days=base_port_cfg.rebalance_horizon_days,
        buffer_entry_rank=entry,
        buffer_exit_rank=exit_,
        max_replacements=max_r,
        hold_bonus=hb,
        turnover_limit=base_port_cfg.turnover_limit,
    )

    best_vol_cfg = VolTargetConfig(
        target_vol_ann=tv,
        lookback_periods=vol_cfg.lookback_periods if vol_cfg else 12,
        max_leverage=vol_cfg.max_leverage if vol_cfg else 1.0,
        min_leverage=vol_cfg.min_leverage if vol_cfg else 0.0,
    ) if vol_cfg else None

    # Sort results by penalized sharpe
    results.sort(key=lambda x: x["penalized_sharpe"], reverse=True)

    tuning_log = {
        "status": "success",
        "n_combos": len(combos),
        "n_evaluated": len(results),
        "best_params": {
            "buffer_entry_rank": entry,
            "buffer_exit_rank": exit_,
            "max_replacements": max_r,
            "hold_bonus": hb,
            "target_vol": tv,
        },
        "best_metrics": best_summary,
        "top_5": results[:5],
    }

    return best_port_cfg, best_vol_cfg, tuning_log


# ═════════════════════════════════════════════════════════════════════════
# Output / reporting
# ═════════════════════════════════════════════════════════════════════════

def save_artifacts(
    out_dir: Path,
    returns_df: pd.DataFrame,
    summary: Dict,
    port_cfg: PortfolioConfig,
    cost_cfg: CostConfig,
    vol_cfg: Optional[VolTargetConfig],
    tuning_log: Optional[Dict] = None,
    label: str = "test",
):
    """Save backtest artifacts to output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Equity curve
    if not returns_df.empty:
        eq_cols = ["date", "gross_ret", "net_ret", "scaled_net_ret",
                   "lev", "turnover", "cost", "cost_scaled", "n_holdings"]
        eq = returns_df[[c for c in eq_cols if c in returns_df.columns]].copy()
        eq["cum_gross"] = np.exp(np.cumsum(eq["gross_ret"].values)) - 1
        eq["cum_net_unscaled"] = np.exp(np.cumsum(eq["net_ret"].values)) - 1
        if "scaled_net_ret" in eq.columns:
            eq["cum_net_volscaled"] = np.exp(np.cumsum(eq["scaled_net_ret"].values)) - 1
        # Backward-compat alias
        eq["cum_net"] = eq["cum_net_unscaled"]
        eq_path = out_dir / "portfolio_equity_curve.csv"
        eq.to_csv(eq_path, index=False)
        logger.info(f"  Equity curve: {eq_path}")

    # Metrics
    pm_path = out_dir / "portfolio_metrics.json"
    with open(pm_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"  Metrics: {pm_path}")

    # Config decision
    from dataclasses import asdict
    decision = {
        "portfolio": asdict(port_cfg),
        "cost": asdict(cost_cfg),
        "vol_scaling": asdict(vol_cfg) if vol_cfg else None,
        "label": label,
    }
    if tuning_log:
        decision["tuning"] = tuning_log

    dec_path = out_dir / "tuning_decision.json"
    with open(dec_path, "w") as f:
        json.dump(decision, f, indent=2, default=str)
    logger.info(f"  Decision: {dec_path}")


def log_summary(summary: Dict, label: str = ""):
    """Pretty-print summary metrics with clear unscaled / vol-scaled labels."""
    if not summary:
        return
    tag = f" ({label})" if label else ""
    logger.info(f"\n  ╔═══════════════════════════════════════════════════╗")
    logger.info(f"  ║  Portfolio Backtest Results{tag:>22s} ║")
    logger.info(f"  ╠═══════════════════════════════════════════════════╣")
    logger.info(f"  ║  Gross Sharpe:               {summary.get('gross_sharpe',0):+8.4f}         ║")
    logger.info(f"  ║  Net Sharpe (unscaled n=1):   {summary.get('net_sharpe_unscaled',0):+8.4f}         ║")
    if "net_sharpe_volscaled" in summary:
        logger.info(f"  ║  Net Sharpe (vol-scaled):    {summary['net_sharpe_volscaled']:+8.4f}         ║")
    logger.info(f"  ║  Ann Vol net (unscaled):      {summary.get('ann_vol_net_unscaled',0)*100:8.1f}%        ║")
    logger.info(f"  ║  CAGR net (unscaled):         {summary.get('cagr_net_unscaled',0)*100:8.1f}%        ║")
    if "cagr_net_volscaled" in summary:
        logger.info(f"  ║  CAGR net (vol-scaled):       {summary['cagr_net_volscaled']*100:8.1f}%        ║")
    logger.info(f"  ║  Max DD:                      {summary.get('max_drawdown',0)*100:8.1f}%        ║")
    logger.info(f"  ║  Turnover 1w:                 {summary.get('avg_turnover_1way',0):8.4f}         ║")
    logger.info(f"  ║  Avg Hold:                    {summary.get('avg_hold_days',0):8.1f}d        ║")
    logger.info(f"  ║  Periods:                     {summary.get('n_periods',0):8d}         ║")
    logger.info(f"  ╚═══════════════════════════════════════════════════╝")


# ═════════════════════════════════════════════════════════════════════════
# Walk-forward evaluation (Part 3)
# ═════════════════════════════════════════════════════════════════════════

def run_walk_forward(
    scored_df: pd.DataFrame,
    base_port_cfg: PortfolioConfig,
    cost_cfg: CostConfig,
    vol_cfg: Optional[VolTargetConfig],
    score_col: str = "score_final",
    target_col: str = "y_ret",
    train_years: int = 3,
    val_years: int = 1,
    test_years: int = 1,
    turnover_penalty: float = 0.0,
) -> Dict[str, Any]:
    """Walk-forward validation with yearly rolling folds.

    Splits data into rolling windows:
      Train: train_years → Val: val_years → Test: test_years
    Tunes portfolio params on val, evaluates on test for each fold.

    Returns
    -------
    summary dict with per-fold and aggregate metrics.
    """
    dates = sorted(scored_df["date"].unique())
    years = sorted(set(pd.Timestamp(d).year for d in dates))

    if len(years) < train_years + val_years + test_years:
        logger.warning(f"  Not enough years for walk-forward: {len(years)} available, "
                       f"need {train_years + val_years + test_years}")
        return {"status": "insufficient_data", "n_years": len(years)}

    folds = []
    total_window = train_years + val_years + test_years

    for start_idx in range(0, len(years) - total_window + 1):
        fold_years = years[start_idx : start_idx + total_window]
        train_yrs = fold_years[:train_years]
        val_yrs = fold_years[train_years : train_years + val_years]
        test_yrs = fold_years[train_years + val_years :]
        folds.append({
            "train_years": train_yrs,
            "val_years": val_yrs,
            "test_years": test_yrs,
        })

    logger.info(f"\n" + "═" * 60)
    logger.info(f"WALK-FORWARD: {len(folds)} folds, "
                f"{train_years}yr train / {val_years}yr val / {test_years}yr test")
    logger.info("═" * 60)

    fold_results = []
    total_test_periods = 0

    for fi, fold in enumerate(folds):
        # Filter scored data by year
        val_mask = scored_df["date"].dt.year.isin(fold["val_years"])
        test_mask = scored_df["date"].dt.year.isin(fold["test_years"])
        val_data = scored_df[val_mask].copy()
        test_data = scored_df[test_mask].copy()

        if val_data.empty or test_data.empty:
            continue

        logger.info(f"\n  Fold {fi+1}/{len(folds)}: "
                    f"val={fold['val_years']} test={fold['test_years']}")

        # Tune on val
        best_port, best_vol, _ = tune_on_val(
            val_data, base_port_cfg, cost_cfg, vol_cfg,
            score_col, target_col, turnover_penalty,
        )

        # Evaluate on test
        _, test_summary = run_backtest(
            test_data, best_port, cost_cfg, best_vol,
            score_col, target_col,
        )

        if test_summary:
            n_p = test_summary.get("n_periods", 0)
            total_test_periods += n_p
            fold_results.append({
                "fold": fi + 1,
                "val_years": fold["val_years"],
                "test_years": fold["test_years"],
                "net_sharpe": test_summary.get("net_sharpe", 0),
                "gross_sharpe": test_summary.get("gross_sharpe", 0),
                "avg_turnover_1way": test_summary.get("avg_turnover_1way", 0),
                "max_drawdown": test_summary.get("max_drawdown", 0),
                "n_periods": n_p,
            })
            logger.info(f"    Net Sharpe: {test_summary['net_sharpe']:+.4f}  "
                        f"TO: {test_summary['avg_turnover_1way']:.4f}  "
                        f"periods: {n_p}")

    if not fold_results:
        return {"status": "no_valid_folds"}

    # Aggregate metrics
    net_sharpes = [f["net_sharpe"] for f in fold_results]
    aggregate = {
        "status": "success",
        "n_folds": len(fold_results),
        "total_test_periods": total_test_periods,
        "median_net_sharpe": round(float(np.median(net_sharpes)), 4),
        "mean_net_sharpe": round(float(np.mean(net_sharpes)), 4),
        "worst_fold_net_sharpe": round(float(np.min(net_sharpes)), 4),
        "best_fold_net_sharpe": round(float(np.max(net_sharpes)), 4),
        "std_net_sharpe": round(float(np.std(net_sharpes, ddof=1)), 4) if len(net_sharpes) > 1 else 0.0,
        "folds": fold_results,
    }

    logger.info(f"\n  Walk-Forward Summary:")
    logger.info(f"    Folds:     {aggregate['n_folds']}")
    logger.info(f"    Periods:   {total_test_periods}")
    logger.info(f"    Median Σ:  {aggregate['median_net_sharpe']:+.4f}")
    logger.info(f"    Mean Σ:    {aggregate['mean_net_sharpe']:+.4f}")
    logger.info(f"    Worst Σ:   {aggregate['worst_fold_net_sharpe']:+.4f}")
    logger.info(f"    Std Σ:     {aggregate['std_net_sharpe']:.4f}")

    return aggregate


# ═════════════════════════════════════════════════════════════════════════
# Cost sensitivity diagnostic (Part 6)
# ═════════════════════════════════════════════════════════════════════════

def run_cost_sensitivity(
    scored_df: pd.DataFrame,
    port_cfg: PortfolioConfig,
    vol_cfg: Optional[VolTargetConfig],
    cost_bps_list: List[float] = None,
    score_col: str = "score_final",
    target_col: str = "y_ret",
) -> Dict[str, Any]:
    """Run backtest at multiple cost levels and compare robustness."""
    if cost_bps_list is None:
        cost_bps_list = [5.0, 10.0, 20.0]

    logger.info(f"\n  Cost sensitivity: testing bps = {cost_bps_list}")
    results = []

    for bps in cost_bps_list:
        cfg = CostConfig(cost_bps=bps)
        _, summary = run_backtest(scored_df, port_cfg, cfg, vol_cfg, score_col, target_col)
        if summary:
            results.append({
                "cost_bps": bps,
                "net_sharpe": summary.get("net_sharpe", 0),
                "cagr_net": summary.get("cagr_net", 0),
                "avg_turnover_1way": summary.get("avg_turnover_1way", 0),
                "effective_cost_ann": summary.get("effective_cost_ann", 0),
            })
            logger.info(f"    {bps:5.1f} bps → Sharpe {summary['net_sharpe']:+.4f}  "
                        f"CAGR {summary['cagr_net']*100:.2f}%")

    return {"cost_sensitivity": results}


# ═════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ═════════════════════════════════════════════════════════════════════════

def _load_scored(path: str) -> pd.DataFrame:
    """Load scored parquet, converting dates if needed."""
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    required = {"date", "symbol", "score_final", "y_ret"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Scored data missing columns: {missing}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run selector portfolio backtest with turnover controls",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data inputs ───────────────────────────────────────────────────
    parser.add_argument("--scored-parquet", required=True,
                        help="Path to scored parquet (val or test)")
    parser.add_argument("--scored-test-parquet", default=None,
                        help="Path to scored test parquet (for tune-on-val mode)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for artifacts")

    # ── Portfolio rules ───────────────────────────────────────────────
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--rebalance-horizon", type=int, default=10)
    parser.add_argument("--buffer-entry-rank", type=int, default=40)
    parser.add_argument("--buffer-exit-rank", type=int, default=70)
    parser.add_argument("--max-replacements", type=int, default=10)
    parser.add_argument("--hold-bonus", type=float, default=0.0)
    parser.add_argument("--turnover-limit", type=float, default=0.50)

    # ── Cost model ────────────────────────────────────────────────────
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--impact-bps", type=float, default=0.0)

    # ── Vol scaling ───────────────────────────────────────────────────
    parser.add_argument("--target-vol", type=float, default=0.15,
                        help="Target annualized vol (0 = disabled)")
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--vol-lookback", type=int, default=12)

    # ── Tuning ────────────────────────────────────────────────────────
    parser.add_argument("--tune-on-val", action="store_true", default=False,
                        help="Grid-search params on scored-parquet (val), "
                             "then evaluate on scored-test-parquet (test)")
    parser.add_argument("--turnover-penalty", type=float, default=0.0,
                        help="Penalty weight on turnover in tuning objective (Part 5)")

    # ── Walk-forward (Part 3) ─────────────────────────────────────────
    parser.add_argument("--walk-forward", action="store_true", default=False,
                        help="Run walk-forward evaluation with yearly rolling folds")
    parser.add_argument("--wf-train-years", type=int, default=3,
                        help="Walk-forward: training window in years")
    parser.add_argument("--wf-val-years", type=int, default=1,
                        help="Walk-forward: validation window in years")
    parser.add_argument("--wf-test-years", type=int, default=1,
                        help="Walk-forward: test window in years")

    # ── Baseline comparison ───────────────────────────────────────────
    parser.add_argument("--run-baseline", action="store_true", default=True,
                        help="Also run full-refresh baseline for comparison")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build configs ─────────────────────────────────────────────────
    port_cfg = PortfolioConfig(
        top_k=args.top_k,
        rebalance_horizon_days=args.rebalance_horizon,
        buffer_entry_rank=args.buffer_entry_rank,
        buffer_exit_rank=args.buffer_exit_rank,
        max_replacements=args.max_replacements,
        hold_bonus=args.hold_bonus,
        turnover_limit=args.turnover_limit,
    )
    cost_cfg = CostConfig(
        cost_bps=args.cost_bps,
        impact_bps=args.impact_bps,
    )
    vol_cfg = VolTargetConfig(
        target_vol_ann=args.target_vol,
        lookback_periods=args.vol_lookback,
        max_leverage=args.max_leverage,
    ) if args.target_vol > 0 else None

    # ══════════════════════════════════════════════════════════════════
    # MODE: Tune on val, evaluate on test
    # ══════════════════════════════════════════════════════════════════
    if args.tune_on_val:
        logger.info("Loading val scored data for tuning...")
        scored_val = _load_scored(args.scored_parquet)
        logger.info(f"  Val: {len(scored_val)} rows, {scored_val['date'].nunique()} dates")

        best_port, best_vol, tuning_log = tune_on_val(
            scored_val, port_cfg, cost_cfg, vol_cfg,
            turnover_penalty=args.turnover_penalty,
        )

        # Now evaluate on test with tuned params
        if args.scored_test_parquet:
            logger.info(f"\nEvaluating tuned params on test data...")
            scored_test = _load_scored(args.scored_test_parquet)
            logger.info(f"  Test: {len(scored_test)} rows, {scored_test['date'].nunique()} dates")

            returns_df, summary = run_backtest(
                scored_test, best_port, cost_cfg, best_vol,
            )
            log_summary(summary, "test w/ tuned params")
            save_artifacts(out_dir, returns_df, summary, best_port, cost_cfg, best_vol,
                          tuning_log, label="test_tuned")

            # Baseline comparison
            if args.run_baseline:
                baseline_cfg = PortfolioConfig(
                    top_k=args.top_k,
                    rebalance_horizon_days=args.rebalance_horizon,
                    buffer_entry_rank=args.top_k,  # no buffer
                    buffer_exit_rank=9999,
                    max_replacements=None,           # no slot cap
                    hold_bonus=0.0,
                )
                ret_base, sum_base = run_backtest(
                    scored_test, baseline_cfg, cost_cfg, vol_cfg,
                )
                log_summary(sum_base, "test BASELINE")

                if sum_base:
                    improvement = {
                        "net_sharpe_delta": round(
                            summary.get("net_sharpe", 0) - sum_base.get("net_sharpe", 0), 4),
                        "turnover_delta": round(
                            summary.get("avg_turnover_1way", 0) - sum_base.get("avg_turnover_1way", 0), 4),
                    }
                    logger.info(f"\n  Improvement vs baseline:")
                    logger.info(f"    Net Sharpe: {improvement['net_sharpe_delta']:+.4f}")
                    logger.info(f"    Turnover:   {improvement['turnover_delta']:+.4f}")

                    # Save baseline metrics too
                    base_path = out_dir / "baseline_metrics.json"
                    with open(base_path, "w") as f:
                        json.dump({"baseline": sum_base, "improvement": improvement},
                                  f, indent=2, default=str)
        else:
            # No test data — just report val tuning results
            logger.info("\n  No --scored-test-parquet provided; reporting val results only.")
            val_ret, val_sum = run_backtest(scored_val, best_port, cost_cfg, best_vol)
            log_summary(val_sum, "val w/ tuned params")
            save_artifacts(out_dir, val_ret, val_sum, best_port, cost_cfg, best_vol,
                          tuning_log, label="val_tuned")

    # ══════════════════════════════════════════════════════════════════
    # MODE: Simple single-pass backtest
    # ══════════════════════════════════════════════════════════════════
    else:
        logger.info("Loading scored data...")
        scored = _load_scored(args.scored_parquet)
        logger.info(f"  Data: {len(scored)} rows, {scored['date'].nunique()} dates")

        returns_df, summary = run_backtest(scored, port_cfg, cost_cfg, vol_cfg)
        log_summary(summary, "single-pass")
        save_artifacts(out_dir, returns_df, summary, port_cfg, cost_cfg, vol_cfg,
                      label="backtest")

        # Baseline comparison
        if args.run_baseline:
            baseline_cfg = PortfolioConfig(
                top_k=args.top_k,
                rebalance_horizon_days=args.rebalance_horizon,
                buffer_entry_rank=args.top_k,
                buffer_exit_rank=9999,
                max_replacements=None,
                hold_bonus=0.0,
            )
            ret_base, sum_base = run_backtest(scored, baseline_cfg, cost_cfg, vol_cfg)
            log_summary(sum_base, "BASELINE")

    # ══════════════════════════════════════════════════════════════════
    # Walk-forward evaluation (Part 3)
    # ══════════════════════════════════════════════════════════════════
    if args.walk_forward:
        logger.info("\n" + "═" * 60)
        logger.info("WALK-FORWARD MODE")
        logger.info("═" * 60)
        scored_all = _load_scored(args.scored_parquet)
        # Merge test data if available
        if args.scored_test_parquet:
            scored_test2 = _load_scored(args.scored_test_parquet)
            scored_all = pd.concat([scored_all, scored_test2], ignore_index=True)
            scored_all = scored_all.drop_duplicates(subset=["date", "symbol"])

        wf_summary = run_walk_forward(
            scored_all, port_cfg, cost_cfg, vol_cfg,
            train_years=args.wf_train_years,
            val_years=args.wf_val_years,
            test_years=args.wf_test_years,
            turnover_penalty=args.turnover_penalty,
        )
        wf_path = out_dir / "walk_forward_summary.json"
        with open(wf_path, "w") as f:
            json.dump(wf_summary, f, indent=2, default=str)
        logger.info(f"  Walk-forward summary: {wf_path}")

    # ══════════════════════════════════════════════════════════════════
    # Cost sensitivity diagnostic (Part 6)
    # ══════════════════════════════════════════════════════════════════
    if not args.walk_forward:
        # Run cost sensitivity on whichever data we last used
        scored_for_cs = _load_scored(args.scored_test_parquet or args.scored_parquet)
        cs_report = run_cost_sensitivity(
            scored_for_cs, port_cfg, vol_cfg,
        )
        cs_path = out_dir / "cost_sensitivity.json"
        with open(cs_path, "w") as f:
            json.dump(cs_report, f, indent=2, default=str)
        logger.info(f"  Cost sensitivity: {cs_path}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
