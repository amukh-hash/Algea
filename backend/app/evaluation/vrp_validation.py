"""
Multi-regime empirical validation suite for the VRP sleeve.

Produces:
- Regime frequency distribution, duration, transition matrix
- PnL / drawdown / ES per regime bucket
- Scenario loss compliance rates
- Danger-zone trigger frequency
- De-risk actions per regime
- Allocation weight analysis (churn detection)
- Forecast health time-series
- Stress-window specific reports

All outputs persisted to {artifact_root}/vrp_validation/{run_id}/
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RegimeStats:
    """Statistics for a single regime bucket."""
    regime: str
    count: int = 0
    pct: float = 0.0
    avg_duration_days: float = 0.0
    total_pnl: float = 0.0
    avg_daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    es_95: float = 0.0
    es_99: float = 0.0
    hit_rate: float = 0.0
    profit_factor: float = 0.0
    avg_scenario_loss_pct: float = 0.0
    danger_zone_triggers: int = 0
    derisk_actions: int = 0
    # v4 expansion metrics
    mean_w_vrp: float = 0.0               # mean weight in this regime
    return_per_unit_risk: float = 0.0     # pnl / (es_95 * count) if es nonzero
    allocation_efficiency: float = 0.0    # pnl / avg_scenario_loss if nonzero


@dataclass
class StressWindowResult:
    """Result of validating a known stress window."""
    window_name: str
    start_date: str
    end_date: str
    crash_risk_triggered: bool = False
    crash_risk_days: int = 0
    max_scenario_loss_pct: float = 0.0
    budget_breaches: int = 0
    total_days: int = 0


@dataclass
class AllocationAnalysis:
    """Allocation weight stability metrics."""
    mean_w_vrp: float = 0.0
    std_w_vrp: float = 0.0
    max_daily_change: float = 0.0
    avg_daily_change: float = 0.0
    days_at_zero: int = 0
    churn_days: int = 0  # days with |Δw| > 0.02
    capital_utilization_ratio: float = 0.0  # mean_w_vrp / w_max_vrp


@dataclass
class ValidationSummary:
    """Top-level summary of the empirical validation run."""
    run_id: str
    total_days: int = 0
    regime_stats: List[RegimeStats] = field(default_factory=list)
    transition_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    stress_windows: List[StressWindowResult] = field(default_factory=list)
    allocation_analysis: Optional[AllocationAnalysis] = None
    scenario_budget_compliance_pct: float = 0.0
    forecast_health_mean: float = 0.0
    forecast_health_min: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════
# Regime analysis functions
# ═══════════════════════════════════════════════════════════════════════════

KNOWN_STRESS_WINDOWS = [
    ("2018-Q4", "2018-10-01", "2018-12-31"),
    ("2020-COVID", "2020-02-15", "2020-04-15"),
    ("2022-Jun", "2022-06-01", "2022-06-30"),
    ("2022-Sep", "2022-09-01", "2022-09-30"),
    ("2023-Banking", "2023-03-08", "2023-03-24"),
]


def compute_regime_frequency(regimes: pd.Series) -> Dict[str, float]:
    """Compute frequency distribution of regimes."""
    counts = Counter(regimes.dropna())
    total = sum(counts.values())
    return {str(k): v / max(total, 1) for k, v in counts.items()}


def compute_regime_durations(regimes: pd.Series) -> Dict[str, float]:
    """Compute average consecutive duration in each regime."""
    durations: Dict[str, List[int]] = defaultdict(list)
    if len(regimes) == 0:
        return {}

    current = str(regimes.iloc[0])
    run_len = 1
    for i in range(1, len(regimes)):
        val = str(regimes.iloc[i])
        if val == current:
            run_len += 1
        else:
            durations[current].append(run_len)
            current = val
            run_len = 1
    durations[current].append(run_len)

    return {k: float(np.mean(v)) for k, v in durations.items()}


def compute_transition_matrix(regimes: pd.Series) -> Dict[str, Dict[str, int]]:
    """Compute regime transition counts."""
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    vals = [str(v) for v in regimes.dropna().values]
    for i in range(1, len(vals)):
        matrix[vals[i - 1]][vals[i]] += 1
    return {k: dict(v) for k, v in matrix.items()}


def compute_pnl_by_regime(
    regimes: pd.Series,
    daily_pnl: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """Compute PnL statistics bucketed by regime."""
    result = {}
    aligned = pd.DataFrame({"regime": regimes, "pnl": daily_pnl}).dropna()
    for regime in aligned["regime"].unique():
        bucket = aligned[aligned["regime"] == regime]["pnl"]
        wins = bucket[bucket > 0]
        losses = bucket[bucket < 0]
        gross_profit = wins.sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-9
        result[str(regime)] = {
            "total_pnl": float(bucket.sum()),
            "avg_daily_pnl": float(bucket.mean()),
            "hit_rate": float(len(wins) / max(len(bucket), 1)),
            "profit_factor": float(gross_profit / max(gross_loss, 1e-9)),
            "count": len(bucket),
        }
    return result


def compute_es(pnl: pd.Series, level: float = 0.95) -> float:
    """Expected shortfall at the given confidence level."""
    if len(pnl) < 5:
        return 0.0
    sorted_pnl = np.sort(pnl.values)
    cutoff = int(len(sorted_pnl) * (1 - level))
    tail = sorted_pnl[:max(cutoff, 1)]
    return float(np.mean(tail))


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Compute drawdown series from cumulative equity curve."""
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1.0
    return dd


def analyze_allocation_stability(
    w_vrp_series: pd.Series,
    churn_threshold: float = 0.02,
) -> AllocationAnalysis:
    """Analyze allocation weight stability."""
    if len(w_vrp_series) == 0:
        return AllocationAnalysis()

    changes = w_vrp_series.diff().abs().dropna()
    return AllocationAnalysis(
        mean_w_vrp=float(w_vrp_series.mean()),
        std_w_vrp=float(w_vrp_series.std()),
        max_daily_change=float(changes.max()) if len(changes) > 0 else 0.0,
        avg_daily_change=float(changes.mean()) if len(changes) > 0 else 0.0,
        days_at_zero=int((w_vrp_series == 0.0).sum()),
        churn_days=int((changes > churn_threshold).sum()) if len(changes) > 0 else 0,
    )


def validate_stress_window(
    regimes: pd.Series,
    scenario_losses_pct: pd.Series,
    budget_limit: float,
    window_name: str,
    start_date: str,
    end_date: str,
) -> StressWindowResult:
    """Validate behavior during a known stress window."""
    mask = (regimes.index >= start_date) & (regimes.index <= end_date)
    if mask.sum() == 0:
        return StressWindowResult(
            window_name=window_name,
            start_date=start_date,
            end_date=end_date,
            total_days=0,
        )

    window_regimes = regimes[mask]
    window_losses = scenario_losses_pct.reindex(window_regimes.index, fill_value=0.0)

    crash_days = (window_regimes.astype(str).str.contains("crash")).sum()
    budget_breaches = (abs(window_losses) > budget_limit).sum()

    return StressWindowResult(
        window_name=window_name,
        start_date=start_date,
        end_date=end_date,
        crash_risk_triggered=crash_days > 0,
        crash_risk_days=int(crash_days),
        max_scenario_loss_pct=float(abs(window_losses).max()) if len(window_losses) > 0 else 0.0,
        budget_breaches=int(budget_breaches),
        total_days=int(mask.sum()),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main validation runner
# ═══════════════════════════════════════════════════════════════════════════

def run_validation(
    regimes: pd.Series,
    daily_pnl: pd.Series,
    scenario_losses_pct: pd.Series,
    w_vrp_series: pd.Series,
    forecast_health: pd.Series,
    danger_zone_counts: pd.Series,
    derisk_action_counts: pd.Series,
    budget_limit: float = 0.06,
    run_id: str = "default",
) -> ValidationSummary:
    """Run the full empirical validation suite.

    All series must be date-indexed (str or datetime).
    """
    # Regime stats
    freq = compute_regime_frequency(regimes)
    durations = compute_regime_durations(regimes)
    transition = compute_transition_matrix(regimes)
    pnl_by_regime = compute_pnl_by_regime(regimes, daily_pnl)

    regime_stats_list = []
    for regime_name in ["normal_carry", "caution", "crash_risk"]:
        pnl_info = pnl_by_regime.get(regime_name, {})
        regime_mask = regimes.astype(str) == regime_name
        has_days = regime_name in regimes.astype(str).values
        regime_bucket_pnl = daily_pnl[regime_mask] if has_days else pd.Series(dtype=float)
        regime_bucket_losses = scenario_losses_pct[regime_mask] if has_days else pd.Series(dtype=float)
        regime_bucket_dz = danger_zone_counts[regime_mask] if has_days else pd.Series(dtype=float)
        regime_bucket_dr = derisk_action_counts[regime_mask] if has_days else pd.Series(dtype=float)
        regime_bucket_w = w_vrp_series[regime_mask] if has_days else pd.Series(dtype=float)

        count = pnl_info.get("count", 0)
        total_pnl = pnl_info.get("total_pnl", 0.0)
        es95 = compute_es(regime_bucket_pnl, 0.95)
        es99 = compute_es(regime_bucket_pnl, 0.99)
        avg_scen = float(abs(regime_bucket_losses).mean()) if len(regime_bucket_losses) > 0 else 0.0
        mean_w = float(regime_bucket_w.mean()) if len(regime_bucket_w) > 0 else 0.0

        # Return per unit risk: total_pnl / (|es95| * count) if applicable
        rpur = 0.0
        if es95 != 0 and count > 0:
            rpur = total_pnl / (abs(es95) * count)

        # Allocation efficiency: total_pnl / avg_scenario_loss
        alloc_eff = 0.0
        if avg_scen > 0:
            alloc_eff = total_pnl / (avg_scen * count) if count > 0 else 0.0

        rs = RegimeStats(
            regime=regime_name,
            count=count,
            pct=freq.get(regime_name, 0.0),
            avg_duration_days=durations.get(regime_name, 0.0),
            total_pnl=total_pnl,
            avg_daily_pnl=pnl_info.get("avg_daily_pnl", 0.0),
            max_drawdown=float(compute_drawdown_series(regime_bucket_pnl.cumsum()).min()) if len(regime_bucket_pnl) > 0 else 0.0,
            es_95=es95,
            es_99=es99,
            hit_rate=pnl_info.get("hit_rate", 0.0),
            profit_factor=pnl_info.get("profit_factor", 0.0),
            avg_scenario_loss_pct=avg_scen,
            danger_zone_triggers=int(regime_bucket_dz.sum()) if len(regime_bucket_dz) > 0 else 0,
            derisk_actions=int(regime_bucket_dr.sum()) if len(regime_bucket_dr) > 0 else 0,
            mean_w_vrp=mean_w,
            return_per_unit_risk=rpur,
            allocation_efficiency=alloc_eff,
        )
        regime_stats_list.append(rs)

    # Stress windows
    stress_results = []
    for name, start, end in KNOWN_STRESS_WINDOWS:
        sr = validate_stress_window(regimes, scenario_losses_pct, budget_limit, name, start, end)
        stress_results.append(sr)

    # Allocation analysis
    alloc_analysis = analyze_allocation_stability(w_vrp_series)
    # Capital utilization ratio
    alloc_analysis.capital_utilization_ratio = alloc_analysis.mean_w_vrp / 0.25 if 0.25 > 0 else 0.0

    # Scenario budget compliance
    budget_compliance = float((abs(scenario_losses_pct) <= budget_limit).mean()) if len(scenario_losses_pct) > 0 else 1.0

    # Forecast health
    fh_mean = float(forecast_health.mean()) if len(forecast_health) > 0 else 0.0
    fh_min = float(forecast_health.min()) if len(forecast_health) > 0 else 0.0

    return ValidationSummary(
        run_id=run_id,
        total_days=len(regimes),
        regime_stats=regime_stats_list,
        transition_matrix=transition,
        stress_windows=stress_results,
        allocation_analysis=alloc_analysis,
        scenario_budget_compliance_pct=budget_compliance,
        forecast_health_mean=fh_mean,
        forecast_health_min=fh_min,
    )


def save_validation(
    summary: ValidationSummary,
    root: Path,
) -> Path:
    """Persist validation results."""
    out_dir = root / "vrp_validation" / summary.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON summary
    json_path = out_dir / "summary.json"
    json_path.write_text(
        json.dumps(summary.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )

    # CSV — regime stats
    regime_df = pd.DataFrame([asdict(rs) for rs in summary.regime_stats])
    regime_df.to_csv(out_dir / "regime_stats.csv", index=False)

    # CSV — stress windows
    stress_df = pd.DataFrame([asdict(sw) for sw in summary.stress_windows])
    stress_df.to_csv(out_dir / "stress_windows.csv", index=False)

    return json_path
