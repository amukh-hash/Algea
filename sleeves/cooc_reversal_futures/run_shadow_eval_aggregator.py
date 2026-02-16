"""Aggregate shadow eval reports into a promotion decision.

Reads all ``shadow_eval_*.json`` files from a directory, computes window-level
metrics, and emits a ``shadow_eval_summary.json`` with a go/no-go decision.

Usage
-----
::

    python -m sleeves.cooc_reversal_futures.run_shadow_eval_aggregator \\
        --shadow-dir runs/shadow_eval \\
        --output shadow_eval_summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision thresholds (configurable via CLI or config)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    "min_sessions": 30,
    "worst_1pct_improvement_bps": 10.0,
    "max_drawdown_must_improve": True,
    "mean_return_degradation_tolerance_bps": 5.0,
    "max_concentrated_days_pct": 0.20,  # no more than 20% of total PnL from top 2 days
}


def aggregate_shadow_evals(shadow_dir: Path) -> dict:
    """Read all shadow_eval_*.json files and aggregate metrics."""
    files = sorted(shadow_dir.glob("shadow_eval_*.json"))
    if not files:
        return {"error": "No shadow eval files found", "n_sessions": 0}

    sessions: List[dict] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            sessions.append(data)
        except Exception as e:
            logger.warning("Failed to read %s: %s", f, e)

    if not sessions:
        return {"error": "All shadow eval files failed to parse", "n_sessions": 0}

    # Extract daily PnL series
    model_pnls = [s.get("model_pnl", 0.0) for s in sessions]
    heuristic_pnls = [s.get("heuristic_pnl", 0.0) for s in sessions]
    dates = [s.get("asof", "unknown") for s in sessions]
    fill_rates = [s.get("fill_completeness", 1.0) for s in sessions]

    model_arr = np.array(model_pnls, dtype=float)
    heuristic_arr = np.array(heuristic_pnls, dtype=float)

    def _stats(arr: np.ndarray) -> dict:
        if len(arr) == 0:
            return {}
        return {
            "mean_bps": float(np.mean(arr) * 1e4),
            "std_bps": float(np.std(arr, ddof=1) * 1e4) if len(arr) > 1 else 0.0,
            "sharpe_proxy": float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252)) if np.std(arr, ddof=1) > 1e-12 else 0.0,
            "max_drawdown": float(_max_drawdown(arr)),
            "worst_1pct_bps": float(np.percentile(arr, 1) * 1e4) if len(arr) >= 20 else float(np.min(arr) * 1e4),
            "hit_rate": float(np.mean(arr > 0)),
            "n_days": len(arr),
        }

    def _max_drawdown(arr: np.ndarray) -> float:
        cum = np.cumsum(arr)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        return float(np.min(dd)) if len(dd) > 0 else 0.0

    model_stats = _stats(model_arr)
    heuristic_stats = _stats(heuristic_arr)

    # Concentration check: top 2 days' contribution
    if len(model_arr) > 4:
        sorted_abs = np.sort(np.abs(model_arr))[::-1]
        top2_contribution = float(np.sum(sorted_abs[:2]) / (np.sum(np.abs(model_arr)) + 1e-12))
    else:
        top2_contribution = 1.0

    # Fill completeness
    mean_fill_rate = float(np.mean(fill_rates))

    return {
        "n_sessions": len(sessions),
        "date_range": {"start": dates[0], "end": dates[-1]},
        "model": model_stats,
        "heuristic": heuristic_stats,
        "concentration_top2_pct": top2_contribution,
        "mean_fill_completeness": mean_fill_rate,
        "daily_pnl": {
            "dates": dates,
            "model": [float(x) for x in model_pnls],
            "heuristic": [float(x) for x in heuristic_pnls],
        },
    }


def make_promotion_decision(
    summary: dict,
    thresholds: dict | None = None,
) -> dict:
    """Apply go/no-go rules to aggregated shadow eval."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    decision: Dict[str, Any] = {
        "thresholds_used": t,
        "checks": [],
    }

    n = summary.get("n_sessions", 0)
    model = summary.get("model", {})
    heuristic = summary.get("heuristic", {})

    # Check 1: minimum sessions
    enough_sessions = n >= t["min_sessions"]
    decision["checks"].append({
        "name": "min_sessions",
        "passed": enough_sessions,
        "detail": f"{n} sessions (need {t['min_sessions']})",
    })

    # Check 2: worst 1% improvement
    model_w1 = model.get("worst_1pct_bps", 0)
    heuristic_w1 = heuristic.get("worst_1pct_bps", 0)
    w1_improvement = model_w1 - heuristic_w1  # higher = better (less negative)
    w1_pass = w1_improvement >= t["worst_1pct_improvement_bps"]
    decision["checks"].append({
        "name": "worst_1pct_improvement",
        "passed": w1_pass,
        "detail": f"Model={model_w1:.1f}bps vs Heuristic={heuristic_w1:.1f}bps → Δ={w1_improvement:.1f}bps (need ≥{t['worst_1pct_improvement_bps']})",
    })

    # Check 3: max drawdown improvement
    model_dd = model.get("max_drawdown", 0)
    heuristic_dd = heuristic.get("max_drawdown", 0)
    dd_pass = model_dd >= heuristic_dd if t["max_drawdown_must_improve"] else True  # less negative = better
    decision["checks"].append({
        "name": "max_drawdown_improvement",
        "passed": dd_pass,
        "detail": f"Model DD={model_dd:.4f} vs Heuristic DD={heuristic_dd:.4f}",
    })

    # Check 4: mean return not degraded
    model_mean = model.get("mean_bps", 0)
    heuristic_mean = heuristic.get("mean_bps", 0)
    degradation = heuristic_mean - model_mean
    mean_pass = degradation <= t["mean_return_degradation_tolerance_bps"]
    decision["checks"].append({
        "name": "mean_return_not_degraded",
        "passed": mean_pass,
        "detail": f"Model={model_mean:.1f}bps vs Heuristic={heuristic_mean:.1f}bps → degradation={degradation:.1f}bps (tolerance={t['mean_return_degradation_tolerance_bps']})",
    })

    # Check 5: not concentrated
    conc = summary.get("concentration_top2_pct", 1.0)
    conc_pass = conc <= t["max_concentrated_days_pct"]
    decision["checks"].append({
        "name": "not_concentrated",
        "passed": conc_pass,
        "detail": f"Top 2 days = {conc:.1%} of total |PnL| (max {t['max_concentrated_days_pct']:.0%})",
    })

    # Overall decision
    all_passed = all(c["passed"] for c in decision["checks"])
    decision["promote"] = all_passed
    decision["recommendation"] = (
        "PROMOTE — switch to MODEL execution"
        if all_passed
        else "HOLD — keep HEURISTIC execution"
    )

    return decision


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Aggregate shadow eval reports and make promotion decision."
    )
    parser.add_argument("--shadow-dir", required=True, help="Directory with shadow_eval_*.json files")
    parser.add_argument("--output", default="shadow_eval_summary.json", help="Output summary path")
    parser.add_argument("--min-sessions", type=int, default=30, help="Minimum sessions required")
    parser.add_argument("--worst-1pct-threshold", type=float, default=10.0,
                        help="Minimum worst-1%% improvement in bps")
    args = parser.parse_args()

    shadow_dir = Path(args.shadow_dir)
    summary = aggregate_shadow_evals(shadow_dir)

    thresholds = {
        "min_sessions": args.min_sessions,
        "worst_1pct_improvement_bps": args.worst_1pct_threshold,
    }
    decision = make_promotion_decision(summary, thresholds)

    combined = {**summary, "decision": decision}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(combined, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("Shadow Eval Summary: %d sessions", summary.get("n_sessions", 0))
    logger.info("Decision: %s", decision["recommendation"])
    logger.info("Report → %s", output_path)
    logger.info("=" * 60)
    for c in decision["checks"]:
        flag = "✅" if c["passed"] else "❌"
        logger.info("  %s %s: %s", flag, c["name"], c["detail"])

    if not decision["promote"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
