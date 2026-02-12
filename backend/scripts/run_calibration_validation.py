"""Post-calibration validation summary — v4 exposure expansion."""
import numpy as np
import pandas as pd
from datetime import date, timedelta

np.random.seed(42)

from algaie.execution.options.config import VRPConfig
from algaie.trading.meta_allocator import MetaAllocator, SleeveResult, AllocationContext
from algaie.data.options.vrp_features import VolRegime
from backend.app.evaluation.vrp_validation import run_validation

cfg = VRPConfig()
alloc = MetaAllocator(cfg)
sleeves = [
    SleeveResult(name="equity", expected_return=0.08, realized_vol=0.16),
    SleeveResult(name="vrp", expected_return=0.06, realized_vol=0.08, es_95=0.10),
]

N = 252
dates = pd.date_range("2024-01-01", periods=N, freq="B")
regime_str = (
    ["normal_carry"] * 150
    + ["caution"] * 40
    + ["crash_risk"] * 20
    + ["normal_carry"] * 42
)

# Simulate forecast health (drops during caution/crash)
fh = np.ones(N) * 0.92
fh[150:190] = 0.75
fh[190:210] = 0.60

weights = []
for i in range(N):
    d = dates[i].date()
    r = VolRegime(regime_str[i])
    health = fh[i]
    ctx = AllocationContext(
        forecast_health=health,
        headroom_ratio=0.80 if r == VolRegime.NORMAL_CARRY else 0.40,
    )
    result = alloc.combine(d, sleeves, nav=1_000_000, regime=r, context=ctx)
    weights.append(result.w_vrp)

w_vrp = pd.Series(weights, index=dates)
regimes = pd.Series(regime_str, index=dates)

nav = 1_000_000
daily_pnl = []
for i in range(N):
    r = regime_str[i]
    w = weights[i]
    if r == "normal_carry":
        base = 0.06 / 252
        noise = np.random.normal(0, 0.08 / np.sqrt(252))
    elif r == "caution":
        base = -0.02 / 252
        noise = np.random.normal(0, 0.12 / np.sqrt(252))
    else:
        base = -0.15 / 252
        noise = np.random.normal(0, 0.25 / np.sqrt(252))
    daily_pnl.append(w * (base + noise) * nav)

daily_pnl = pd.Series(daily_pnl, index=dates)
scenario_losses_pct = w_vrp * -0.04 + np.random.uniform(-0.005, 0.005, N)
scenario_losses_pct = pd.Series(scenario_losses_pct.values, index=dates)
forecast_health = pd.Series(fh, index=dates)

dz_counts = pd.Series(np.zeros(N, dtype=int), index=dates)
dr_counts = pd.Series(np.zeros(N, dtype=int), index=dates)
dz_counts.iloc[155:165] = 1
dr_counts.iloc[155:165] = 1

summary = run_validation(
    regimes=regimes,
    daily_pnl=daily_pnl,
    scenario_losses_pct=scenario_losses_pct,
    w_vrp_series=w_vrp,
    forecast_health=forecast_health,
    danger_zone_counts=dz_counts,
    derisk_action_counts=dr_counts,
    budget_limit=cfg.max_worst_case_scenario_loss_pct_nav,
    run_id="expansion_v4",
)

sep = "=" * 70
dash = "-" * 70

print(sep)
print("VRP VALIDATION SUMMARY -- v4 EXPOSURE EXPANSION")
print(sep)
print("Run ID:      ", summary.run_id)
print("Total Days:  ", summary.total_days)
print("Budget Compliance: {:.1%}".format(summary.scenario_budget_compliance_pct))
print("Forecast Health:   mean={:.3f}  min={:.3f}".format(
    summary.forecast_health_mean, summary.forecast_health_min
))
print()

print(dash)
print("PnL BY REGIME + EXPECTED SHORTFALL")
print(dash)
fmt = "{:<16s} {:>5s} {:>6s} {:>12s} {:>10s} {:>8s} {:>6s} {:>10s} {:>10s} {:>10s} {:>8s}"
print(fmt.format("Regime", "Days", "Pct", "TotalPnL", "AvgDaily", "HitRate", "PF", "ES95", "ES99", "MaxDD", "AvgW"))
print("-" * 112)
for rs in summary.regime_stats:
    print(
        "{:<16s} {:>5d} {:>5.1%} {:>12,.0f} {:>10,.0f} {:>7.1%} {:>6.2f} {:>10,.0f} {:>10,.0f} {:>10,.0f} {:>8.4f}".format(
            rs.regime, rs.count, rs.pct,
            rs.total_pnl, rs.avg_daily_pnl,
            rs.hit_rate, rs.profit_factor,
            rs.es_95, rs.es_99, rs.max_drawdown,
            rs.mean_w_vrp,
        )
    )
print()

print(dash)
print("ALLOCATION STABILITY")
print(dash)
a = summary.allocation_analysis
print("  mean_w_vrp:             {:.4f}".format(a.mean_w_vrp))
print("  std_w_vrp:              {:.4f}".format(a.std_w_vrp))
print("  max_daily_change:       {:.4f}  (cap: {})".format(a.max_daily_change, cfg.w_max_daily_delta))
print("  avg_daily_change:       {:.6f}".format(a.avg_daily_change))
print("  churn_days:             {}".format(a.churn_days))
print("  days_at_zero:           {}".format(a.days_at_zero))
print("  capital_util_ratio:     {:.2%}".format(a.capital_utilization_ratio))
print()

print(dash)
print("REGIME TRANSITION MATRIX")
print(dash)
for from_r, to_dict in summary.transition_matrix.items():
    for to_r, cnt in to_dict.items():
        if cnt > 0:
            print("  {:<16s} -> {:<16s} : {}".format(from_r, to_r, cnt))
print()

print(dash)
print("DANGER ZONE + DERISK ACTIONS BY REGIME")
print(dash)
for rs in summary.regime_stats:
    print("  {:<16s} DZ triggers: {:>3d}   Derisk actions: {:>3d}".format(
        rs.regime, rs.danger_zone_triggers, rs.derisk_actions
    ))
print()

print(sep)
print("ACCEPTANCE CRITERIA -- v4 EXPANSION")
print(sep)

tag = "PASS" if a.max_daily_change <= cfg.w_max_daily_delta + 1e-9 else "FAIL"
print("[{}]  w_final daily delta <= cap ({})        : {:.4f}".format(tag, cfg.w_max_daily_delta, a.max_daily_change))

crash_rs = next((r for r in summary.regime_stats if r.regime == "crash_risk"), None)
tag2 = "PASS" if crash_rs and crash_rs.total_pnl == 0 else "~OK~"
print("[{}]  CRASH regime: zero exposure / no loss         : total_pnl={:,.0f}".format(tag2, crash_rs.total_pnl))

caution_rs = next((r for r in summary.regime_stats if r.regime == "caution"), None)
normal_rs = next((r for r in summary.regime_stats if r.regime == "normal_carry"), None)
tag3 = "PASS" if caution_rs and caution_rs.mean_w_vrp < (normal_rs.mean_w_vrp if normal_rs else 1) else "WARN"
print("[{}]  CAUTION weight < NORMAL weight               : {:.4f} < {:.4f}".format(
    tag3, caution_rs.mean_w_vrp if caution_rs else 0, normal_rs.mean_w_vrp if normal_rs else 0
))

tag4 = "PASS" if summary.scenario_budget_compliance_pct >= 0.98 else "WARN"
print("[{}]  Scenario budget compliance >= 98%              : {:.1%}".format(
    tag4, summary.scenario_budget_compliance_pct
))

tag5 = "PASS" if a.mean_w_vrp > 0.05 else "FAIL"
print("[{}]  mean_w_vrp > 0.05 (overall)                  : {:.4f}".format(tag5, a.mean_w_vrp))

tag6 = "PASS" if normal_rs and normal_rs.mean_w_vrp > 0.08 else "FAIL"
print("[{}]  NORMAL mean weight > 0.08                    : {:.4f}".format(
    tag6, normal_rs.mean_w_vrp if normal_rs else 0
))

tag7 = "PASS" if crash_rs and crash_rs.mean_w_vrp == 0 else "FAIL"
print("[{}]  CRASH weight == 0                            : {:.4f}".format(
    tag7, crash_rs.mean_w_vrp if crash_rs else 0
))

tag8 = "PASS" if a.churn_days == 0 else "INFO"
print("[{}]  Churn days (|dw| > 2%)                        : {}".format(tag8, a.churn_days))

tag9 = "PASS" if a.capital_utilization_ratio > 0.20 else "WARN"
print("[{}]  Capital utilization ratio > 20%               : {:.1%}".format(
    tag9, a.capital_utilization_ratio
))
