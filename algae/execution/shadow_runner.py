"""R7: Paper-trading shadow harness.

Validates model predictions against actual fills, comparing realized PnL
to proxy-assumed PnL.  Used post-deployment to decide whether shadow
results justify live promotion.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShadowRunConfig:
    """Configuration for a shadow evaluation run.

    Parameters
    ----------
    min_days
        Minimum number of matching fill days to produce a valid report.
    max_slippage_drift_bps
        When realized minus assumed slippage exceeds this, flag as drift.
    promotion_sharpe_min
        Realized Sharpe must meet this threshold for shadow promotion.
    promotion_worst1_min
        Realized worst-1% daily return floor.
    fill_mismatch_max
        Maximum allowed fraction of model-targeted orders with no matching fill.
    """
    min_days: int = 20
    max_slippage_drift_bps: float = 5.0
    promotion_sharpe_min: float = 0.8
    promotion_worst1_min: float = -0.03
    fill_mismatch_max: float = 0.10

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class ShadowReport:
    """Result of a shadow (paper-trading) evaluation.

    Fields
    ------
    n_days
        Number of matched trading days.
    realized_sharpe
        Annualized Sharpe ratio on realized fills.
    assumed_sharpe
        Sharpe from proxy assumptions (model alpha × proxy cost model).
    realized_worst_1pct
        1st-percentile daily return on realized fills.
    assumed_worst_1pct
        1st-percentile daily return from proxy.
    mean_slippage_drift_bps
        Average `realized_slippage - assumed_slippage` in basis points.
    fill_mismatch_rate
        Fraction of model orders with no matching fill.
    promotion_gate_passed
        Whether the shadow result passes all promotion thresholds.
    gate_details
        Per-gate pass/fail details for diagnostics.
    """
    n_days: int = 0
    realized_sharpe: float = 0.0
    assumed_sharpe: float = 0.0
    realized_worst_1pct: float = 0.0
    assumed_worst_1pct: float = 0.0
    mean_slippage_drift_bps: float = 0.0
    fill_mismatch_rate: float = 0.0
    promotion_gate_passed: bool = False
    gate_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def shadow_evaluate(
    orders: pd.DataFrame,
    fills: pd.DataFrame,
    proxy_daily: pd.DataFrame,
    cfg: Optional[ShadowRunConfig] = None,
) -> ShadowReport:
    """Match model orders with broker fills and compute realised performance.

    Parameters
    ----------
    orders
        Model-generated order log.  Required columns:
        ``trading_day``, ``root``, ``side`` (``"BUY"``/``"SELL"``),
        ``target_qty``, ``assumed_slippage_bps``.
    fills
        Actual broker fill log.  Required columns:
        ``trading_day``, ``root``, ``side``, ``filled_qty``,
        ``fill_price``, ``realized_slippage_bps``.
    proxy_daily
        Daily proxy return series.  Required columns:
        ``trading_day``, ``proxy_return``.
    cfg
        Shadow run configuration.

    Returns
    -------
    ShadowReport
    """
    if cfg is None:
        cfg = ShadowRunConfig()

    report = ShadowReport()

    # -----------------------------------------------------------------------
    # 1. Match orders ↔ fills on (trading_day, root, side)
    # -----------------------------------------------------------------------
    merge_keys = ["trading_day", "root", "side"]
    merged = orders.merge(fills, on=merge_keys, how="left", suffixes=("_ord", "_fill"))

    n_orders = len(orders)
    n_unmatched = int(merged["filled_qty"].isna().sum())
    report.fill_mismatch_rate = n_unmatched / max(n_orders, 1)

    # Drop unmatched
    matched = merged.dropna(subset=["filled_qty"]).copy()

    # -----------------------------------------------------------------------
    # 2. Compute per-fill slippage drift
    # -----------------------------------------------------------------------
    if "assumed_slippage_bps" in matched.columns and "realized_slippage_bps" in matched.columns:
        matched["slip_drift_bps"] = (
            matched["realized_slippage_bps"] - matched["assumed_slippage_bps"]
        )
        report.mean_slippage_drift_bps = float(matched["slip_drift_bps"].mean()) if len(matched) > 0 else 0.0
    else:
        report.mean_slippage_drift_bps = 0.0

    # -----------------------------------------------------------------------
    # 3. Compute realized daily returns from fills
    # -----------------------------------------------------------------------
    if "realized_pnl" in matched.columns:
        daily_pnl = matched.groupby("trading_day")["realized_pnl"].sum()
    else:
        # Approximate: fill_price × filled_qty × realized_slippage_bps / 1e4
        # This is a simplified fallback; real systems should provide realized_pnl
        daily_pnl = matched.groupby("trading_day").size().astype(float) * 0.0

    realized_days = daily_pnl.index
    report.n_days = len(realized_days)

    # -----------------------------------------------------------------------
    # 4. Align with proxy daily returns
    # -----------------------------------------------------------------------
    proxy_indexed = proxy_daily.set_index("trading_day")["proxy_return"] if "trading_day" in proxy_daily.columns else proxy_daily
    common_days = realized_days.intersection(proxy_indexed.index)

    if len(common_days) == 0:
        report.assumed_sharpe = 0.0
        report.realized_sharpe = 0.0
        return report

    realized_ret = daily_pnl.reindex(common_days).fillna(0.0)
    proxy_ret = proxy_indexed.reindex(common_days).fillna(0.0)

    # -----------------------------------------------------------------------
    # 5. Compute Sharpe ratios + worst-1%
    # -----------------------------------------------------------------------
    ann = np.sqrt(252)

    r_mean = realized_ret.mean()
    r_std = realized_ret.std()
    report.realized_sharpe = float(r_mean / r_std * ann) if r_std > 1e-12 else 0.0

    p_mean = proxy_ret.mean()
    p_std = proxy_ret.std()
    report.assumed_sharpe = float(p_mean / p_std * ann) if p_std > 1e-12 else 0.0

    n = len(realized_ret)
    q_idx = max(int(np.floor(n * 0.01)), 1)
    sorted_realized = np.sort(realized_ret.values)
    sorted_proxy = np.sort(proxy_ret.values)
    report.realized_worst_1pct = float(sorted_realized[:q_idx].mean()) if n > 0 else 0.0
    report.assumed_worst_1pct = float(sorted_proxy[:q_idx].mean()) if n > 0 else 0.0

    # -----------------------------------------------------------------------
    # 6. Promotion gate
    # -----------------------------------------------------------------------
    gates: Dict[str, Dict[str, Any]] = {}

    # Min days
    days_ok = report.n_days >= cfg.min_days
    gates["min_days"] = {"passed": days_ok, "value": report.n_days, "threshold": cfg.min_days}

    # Realized Sharpe
    sharpe_ok = report.realized_sharpe >= cfg.promotion_sharpe_min
    gates["realized_sharpe"] = {"passed": sharpe_ok, "value": report.realized_sharpe, "threshold": cfg.promotion_sharpe_min}

    # Realized worst 1%
    worst_ok = report.realized_worst_1pct >= cfg.promotion_worst1_min
    gates["realized_worst_1pct"] = {"passed": worst_ok, "value": report.realized_worst_1pct, "threshold": cfg.promotion_worst1_min}

    # Fill mismatch
    fill_ok = report.fill_mismatch_rate <= cfg.fill_mismatch_max
    gates["fill_mismatch"] = {"passed": fill_ok, "value": report.fill_mismatch_rate, "threshold": cfg.fill_mismatch_max}

    # Slippage drift
    drift_ok = abs(report.mean_slippage_drift_bps) <= cfg.max_slippage_drift_bps
    gates["slippage_drift"] = {"passed": drift_ok, "value": report.mean_slippage_drift_bps, "threshold": cfg.max_slippage_drift_bps}

    report.gate_details = gates
    report.promotion_gate_passed = all(g["passed"] for g in gates.values())

    return report
