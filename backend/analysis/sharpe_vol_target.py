"""
Vol Target Scaling Study — tests whether high Sharpe survives at realistic vol.

Given a strategy classified as UNDER_DEPLOYMENT_DRIVEN_SHARPE (e.g. 2.5% vol),
this module scales returns to target volatilities (5%/8%/10%/12%) and measures
how Sharpe, ES95, max drawdown, and cost-adjusted Sharpe behave.

Entry point: ``run_vol_target_study()``
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.analysis.sharpe_report import (
    TRADING_DAYS_PER_YEAR,
    MIN_OBS,
    compute_sharpe,
    _ann_return,
    _ann_vol,
    _max_drawdown,
)
from backend.analysis.sharpe_validator import _compute_es

logger = logging.getLogger(__name__)

DEFAULT_TARGET_VOLS = [0.05, 0.08, 0.10, 0.12]
DEFAULT_COST_BPS = [5.0, 10.0, 20.0]
MAX_DD_THRESHOLD = 0.20  # 20% drawdown flag


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Baseline Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_baseline_metrics(
    returns: pd.Series,
    rf_annual: float = 0.0,
) -> Dict[str, float]:
    """Compute baseline suite: return, vol, Sharpe, ES95, max drawdown."""
    n = len(returns)
    ann_ret = _ann_return(returns)
    ann_vol = _ann_vol(returns)
    sharpe = compute_sharpe(returns, rf_annual) if n >= MIN_OBS else 0.0
    es_95 = _compute_es(returns, 0.95)

    # Max drawdown from cumulative returns
    cum = (1 + returns).cumprod()
    dd = _max_drawdown(cum)

    return {
        "ann_return": round(float(ann_ret), 8),
        "ann_vol": round(float(ann_vol), 8),
        "sharpe": round(float(sharpe), 6),
        "es_95": round(float(es_95), 8),
        "max_drawdown": round(float(dd), 8),
        "n_obs": n,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — Vol Target Scaling
# ═══════════════════════════════════════════════════════════════════════════

def compute_scaling_factor(
    baseline_vol: float,
    target_vol: float,
) -> float:
    """scale = target_vol / baseline_vol."""
    if baseline_vol < 1e-12:
        return 0.0
    return target_vol / baseline_vol


def scale_returns(
    returns: pd.Series,
    scale: float,
) -> pd.Series:
    """Apply linear scaling: scaled_ret_t = baseline_ret_t * scale."""
    return returns * scale


def compute_scaled_metrics(
    baseline_returns: pd.Series,
    target_vol: float,
    baseline_vol: float,
    rf_annual: float = 0.0,
) -> Dict[str, Any]:
    """Compute full metric suite for a given vol target."""
    scale = compute_scaling_factor(baseline_vol, target_vol)
    scaled_ret = scale_returns(baseline_returns, scale)

    ann_ret = _ann_return(scaled_ret)
    ann_vol = _ann_vol(scaled_ret)
    sharpe = compute_sharpe(scaled_ret, rf_annual) if len(scaled_ret) >= MIN_OBS else 0.0
    es_95 = _compute_es(scaled_ret, 0.95)

    cum = (1 + scaled_ret).cumprod()
    dd = _max_drawdown(cum)

    return {
        "target_vol": round(target_vol, 4),
        "scale_factor": round(scale, 6),
        "ann_return": round(float(ann_ret), 8),
        "ann_vol": round(float(ann_vol), 8),
        "sharpe": round(float(sharpe), 6),
        "es_95": round(float(es_95), 8),
        "max_drawdown": round(float(dd), 8),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — Cost Adjustment
# ═══════════════════════════════════════════════════════════════════════════

def cost_adjusted_sharpe(
    baseline_returns: pd.Series,
    scale: float,
    daily_turnover: pd.Series,
    cost_bps: float,
    rf_annual: float = 0.0,
) -> float:
    """Sharpe after subtracting per-trade cost from scaled returns.

    cost_per_day = turnover * scale * (cost_bps / 10_000)
    """
    cost_fraction = cost_bps / 10_000.0
    scaled_ret = baseline_returns * scale
    daily_cost = daily_turnover * scale * cost_fraction
    adjusted_ret = scaled_ret - daily_cost

    if len(adjusted_ret) < MIN_OBS:
        return 0.0
    return float(compute_sharpe(adjusted_ret, rf_annual))


def compute_cost_adjusted_table(
    baseline_returns: pd.Series,
    daily_turnover: pd.Series,
    target_vols: List[float],
    baseline_vol: float,
    cost_bps_list: List[float],
    rf_annual: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    """For each target vol × cost scenario, compute adjusted Sharpe."""
    result: Dict[str, Dict[str, float]] = {}

    for tv in target_vols:
        scale = compute_scaling_factor(baseline_vol, tv)
        label = f"{tv*100:.0f}%"
        row: Dict[str, float] = {}
        for bps in cost_bps_list:
            key = f"+{int(bps)}bps"
            row[key] = round(cost_adjusted_sharpe(
                baseline_returns, scale, daily_turnover, bps, rf_annual
            ), 6)
        result[label] = row

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4 — Nonlinear Risk Adjustment
# ═══════════════════════════════════════════════════════════════════════════

def nonlinear_risk_adjustment(
    baseline_es: float,
    baseline_dd: float,
    scale: float,
) -> Dict[str, Any]:
    """Scale ES and MaxDD by the scaling factor and flag danger zones.

    ES_scaled ≈ ES * scale
    MaxDD_scaled ≈ MaxDD * scale
    """
    es_scaled = baseline_es * scale
    dd_scaled = baseline_dd * scale

    return {
        "es_95_scaled": round(float(es_scaled), 8),
        "max_drawdown_scaled": round(float(dd_scaled), 8),
        "max_dd_exceeds_20pct": abs(dd_scaled) > MAX_DD_THRESHOLD,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5 — Comparison Table & Top-Level API
# ═══════════════════════════════════════════════════════════════════════════

def build_comparison_table(
    baseline: Dict[str, float],
    scaled_rows: List[Dict[str, Any]],
    cost_table: Dict[str, Dict[str, float]],
    risk_adjustments: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Assemble the final comparison table rows."""
    rows: List[Dict[str, Any]] = []

    # Baseline row
    rows.append({
        "target_vol": f"{baseline['ann_vol']*100:.1f}% (baseline)",
        "cagr": round(baseline["ann_return"], 6),
        "sharpe": baseline["sharpe"],
        "es_95": baseline["es_95"],
        "max_drawdown": baseline["max_drawdown"],
        "cost_adj_sharpe_10bps": None,  # baseline has no scaling cost
        "max_dd_exceeds_20pct": abs(baseline["max_drawdown"]) > MAX_DD_THRESHOLD,
    })

    # Scaled rows
    for sr in scaled_rows:
        tv_label = f"{sr['target_vol']*100:.0f}%"
        cost_10 = cost_table.get(tv_label, {}).get("+10bps")
        risk = risk_adjustments.get(tv_label, {})

        rows.append({
            "target_vol": tv_label,
            "cagr": sr["ann_return"],
            "sharpe": sr["sharpe"],
            "es_95": risk.get("es_95_scaled", sr["es_95"]),
            "max_drawdown": risk.get("max_drawdown_scaled", sr["max_drawdown"]),
            "cost_adj_sharpe_10bps": cost_10,
            "max_dd_exceeds_20pct": risk.get("max_dd_exceeds_20pct", False),
        })

    return rows


def run_vol_target_study(
    run_id: str,
    df: pd.DataFrame,
    rf_annual: float = 0.0,
    sleeve: str = "combined",
    target_vols: Optional[List[float]] = None,
    cost_bps_list: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Run full Vol Target Scaling Study.

    Parameters
    ----------
    run_id : str
        Run identifier.
    df : pd.DataFrame
        Output of validate_and_load. Must have core_ret, vrp_ret, port_ret, w_vrp.
    rf_annual : float
        Annual risk-free rate.
    sleeve : str
        Which sleeve to scale: 'core', 'vrp', or 'combined'.
    target_vols : list of float, optional
        Target annualized volatilities (decimal). Default [0.05, 0.08, 0.10, 0.12].
    cost_bps_list : list of float, optional
        Cost scenarios in bps. Default [5, 10, 20].

    Returns
    -------
    Structured dict with all phases.
    """
    if target_vols is None:
        target_vols = DEFAULT_TARGET_VOLS
    if cost_bps_list is None:
        cost_bps_list = DEFAULT_COST_BPS

    col_map = {"core": "core_ret", "vrp": "vrp_ret", "combined": "port_ret"}
    ret_col = col_map.get(sleeve, "port_ret")
    returns = df[ret_col]

    logger.info(f"Vol Target Study for '{run_id}', sleeve='{sleeve}', "
                f"n={len(returns)}, targets={target_vols}")

    # ── Phase 1: Baseline ────────────────────────────────────────────
    baseline = compute_baseline_metrics(returns, rf_annual)

    # ── Phase 2: Vol-target scaling ──────────────────────────────────
    baseline_vol = baseline["ann_vol"]
    scaled_rows: List[Dict[str, Any]] = []
    for tv in target_vols:
        row = compute_scaled_metrics(returns, tv, baseline_vol, rf_annual)
        scaled_rows.append(row)

    # ── Phase 3: Cost adjustment ─────────────────────────────────────
    daily_turnover = df["w_vrp"].diff().abs()
    daily_turnover.iloc[0] = abs(df["w_vrp"].iloc[0])

    cost_table = compute_cost_adjusted_table(
        returns, daily_turnover, target_vols, baseline_vol, cost_bps_list, rf_annual
    )

    # ── Phase 4: Nonlinear risk adjustment ───────────────────────────
    risk_adjustments: Dict[str, Dict[str, Any]] = {}
    for tv in target_vols:
        scale = compute_scaling_factor(baseline_vol, tv)
        label = f"{tv*100:.0f}%"
        risk_adjustments[label] = nonlinear_risk_adjustment(
            baseline["es_95"], baseline["max_drawdown"], scale
        )

    # ── Phase 5: Comparison table ────────────────────────────────────
    table = build_comparison_table(baseline, scaled_rows, cost_table, risk_adjustments)

    # ── Scalability verdict ──────────────────────────────────────────
    verdict = _scalability_verdict(baseline, scaled_rows, cost_table, risk_adjustments)

    return {
        "run_id": run_id,
        "sleeve": sleeve,
        "baseline": baseline,
        "scaled_scenarios": {f"{r['target_vol']*100:.0f}%": r for r in scaled_rows},
        "cost_adjusted_sharpe": cost_table,
        "risk_adjustments": risk_adjustments,
        "comparison_table": table,
        "verdict": verdict,
    }


def _scalability_verdict(
    baseline: Dict[str, float],
    scaled_rows: List[Dict[str, Any]],
    cost_table: Dict[str, Dict[str, float]],
    risk_adjustments: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Classify whether the strategy scales."""
    flags: List[str] = []
    details: Dict[str, str] = {}

    baseline_sharpe = baseline["sharpe"]

    # Check 10% vol scenario at +10bps
    cost_10_at_10pct = cost_table.get("10%", {}).get("+10bps")
    if cost_10_at_10pct is not None and baseline_sharpe > 0:
        retention = cost_10_at_10pct / baseline_sharpe
        if retention > 0.7:
            details["cost_resilience"] = (
                f"At 10% vol + 10bps cost: Sharpe={cost_10_at_10pct:.2f} "
                f"retains {retention*100:.0f}% of baseline ({baseline_sharpe:.2f})."
            )
        else:
            flags.append("COST_ERODES_AT_SCALE")
            details["cost_erosion"] = (
                f"At 10% vol + 10bps cost: Sharpe={cost_10_at_10pct:.2f} "
                f"retains only {retention*100:.0f}% of baseline ({baseline_sharpe:.2f})."
            )

    # Check drawdown at highest target
    last_label = f"{max(DEFAULT_TARGET_VOLS)*100:.0f}%"
    risk = risk_adjustments.get(last_label, {})
    if risk.get("max_dd_exceeds_20pct"):
        flags.append("DRAWDOWN_EXCEEDS_20PCT")
        details["drawdown_warning"] = (
            f"At {last_label} vol: scaled MaxDD = "
            f"{risk.get('max_drawdown_scaled', 0)*100:.1f}% exceeds 20% threshold."
        )

    # Check if Sharpe preserves under scaling (linear scaling should preserve)
    scaled_10 = next(
        (r for r in scaled_rows if abs(r["target_vol"] - 0.10) < 0.001), None
    )
    if scaled_10 is not None and baseline_sharpe > 0:
        sharpe_diff_pct = abs(scaled_10["sharpe"] - baseline_sharpe) / baseline_sharpe * 100
        if sharpe_diff_pct < 2:
            details["sharpe_invariance"] = (
                f"Sharpe ratio preserved under linear scaling "
                f"(baseline={baseline_sharpe:.2f}, @10%={scaled_10['sharpe']:.2f}, "
                f"diff={sharpe_diff_pct:.1f}%)."
            )

    if not flags:
        classification = "SCALES_WELL"
    elif "COST_ERODES_AT_SCALE" in flags and "DRAWDOWN_EXCEEDS_20PCT" in flags:
        classification = "DOES_NOT_SCALE"
    elif "COST_ERODES_AT_SCALE" in flags:
        classification = "COST_LIMITED_SCALABILITY"
    elif "DRAWDOWN_EXCEEDS_20PCT" in flags:
        classification = "RISK_LIMITED_SCALABILITY"
    else:
        classification = "INCONCLUSIVE"

    return {
        "classification": classification,
        "flags": flags,
        "details": details,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Console Printer
# ═══════════════════════════════════════════════════════════════════════════

def print_vol_target_report(report: Dict[str, Any]) -> None:
    """Pretty-print the Vol Target Scaling Study."""
    sep = "=" * 78
    dash = "-" * 78

    print(sep)
    print(f"  VOL TARGET SCALING STUDY — run_id: {report.get('run_id', 'N/A')}")
    print(f"  Sleeve: {report.get('sleeve', 'N/A')}")
    print(sep)

    # ── Baseline ─────────────────────────────────────────────────────
    bl = report.get("baseline", {})
    print(f"\n  BASELINE")
    print(dash)
    print(f"    Ann Return:    {bl.get('ann_return', 'N/A'):>10}")
    print(f"    Ann Vol:       {bl.get('ann_vol', 'N/A'):>10}")
    print(f"    Sharpe:        {bl.get('sharpe', 'N/A'):>10}")
    print(f"    ES-95:         {bl.get('es_95', 'N/A'):>10}")
    print(f"    Max Drawdown:  {bl.get('max_drawdown', 'N/A'):>10}")
    print(f"    Observations:  {bl.get('n_obs', 'N/A'):>10}")

    # ── Comparison Table ─────────────────────────────────────────────
    table = report.get("comparison_table", [])
    print(f"\n  COMPARISON TABLE")
    print(dash)
    header = (f"  {'Target Vol':>16s}  {'CAGR':>10s}  {'Sharpe':>8s}  "
              f"{'ES95':>10s}  {'MaxDD':>10s}  {'Sharpe@10bps':>14s}  {'DD>20%':>6s}")
    print(header)
    print(f"  {'-'*16}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*14}  {'-'*6}")

    for row in table:
        tv = row.get("target_vol", "?")
        cagr = f"{row.get('cagr', 0):>10.4f}" if row.get("cagr") is not None else f"{'N/A':>10}"
        sh = f"{row.get('sharpe', 0):>8.3f}" if row.get("sharpe") is not None else f"{'N/A':>8}"
        es = f"{row.get('es_95', 0):>10.6f}" if row.get("es_95") is not None else f"{'N/A':>10}"
        dd = f"{row.get('max_drawdown', 0)*100:>9.2f}%" if row.get("max_drawdown") is not None else f"{'N/A':>10}"
        ca = (f"{row.get('cost_adj_sharpe_10bps', 0):>14.3f}"
              if row.get("cost_adj_sharpe_10bps") is not None else f"{'—':>14}")
        flag = "  ⚠" if row.get("max_dd_exceeds_20pct") else ""
        print(f"  {tv:>16s}  {cagr}  {sh}  {es}  {dd}  {ca}{flag}")

    # ── Cost Sensitivity Grid ────────────────────────────────────────
    cost_table = report.get("cost_adjusted_sharpe", {})
    if cost_table:
        print(f"\n  COST-ADJUSTED SHARPE GRID")
        print(dash)
        # Header
        first_row = next(iter(cost_table.values()), {})
        bps_labels = list(first_row.keys())
        hdr = f"  {'Target':>10s}"
        for bl in bps_labels:
            hdr += f"  {bl:>10s}"
        print(hdr)
        print(f"  {'-'*10}" + f"  {'-'*10}" * len(bps_labels))

        for tv_label, costs in cost_table.items():
            line = f"  {tv_label:>10s}"
            for bl_key in bps_labels:
                val = costs.get(bl_key, 0)
                line += f"  {val:>10.3f}"
            print(line)

    # ── Risk Adjustments ─────────────────────────────────────────────
    risk = report.get("risk_adjustments", {})
    if risk:
        print(f"\n  NONLINEAR RISK ADJUSTMENTS")
        print(dash)
        for tv_label, r in risk.items():
            flag = " ⚠ DD>20%" if r.get("max_dd_exceeds_20pct") else ""
            print(f"    {tv_label:>6s}: ES95={r.get('es_95_scaled', 0):>10.6f}  "
                  f"MaxDD={r.get('max_drawdown_scaled', 0)*100:>7.2f}%{flag}")

    # ── Verdict ──────────────────────────────────────────────────────
    verdict = report.get("verdict", {})
    classification = verdict.get("classification", "UNKNOWN")
    print(f"\n  SCALABILITY VERDICT")
    print(sep)
    print(f"    ╔════════════════════════════════════════════════════╗")
    print(f"    ║  {classification:<51s}║")
    print(f"    ╚════════════════════════════════════════════════════╝")
    flags = verdict.get("flags", [])
    if flags:
        print(f"    Flags: {', '.join(flags)}")
    for k, v in verdict.get("details", {}).items():
        print(f"    • {v}")

    print(sep)
