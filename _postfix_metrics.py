"""Post-fix metric table: oracle vs baseline vs anti-oracle vs random."""
import numpy as np
import pandas as pd
from sleeves.cooc_reversal_futures.pipeline.trade_proxy import evaluate_trade_proxy

# ---------- synthetic panel (same as tests, deterministic) ----------
rng = np.random.RandomState(42)
n_days, n_inst = 200, 8
insts = [f"ROOT{i}" for i in range(n_inst)]
base = pd.Timestamp("2024-01-02")
rows = []
for d in range(n_days):
    td = (base + pd.offsets.BDay(d)).date()
    r_co = rng.randn(n_inst) * 0.01
    r_oc = -0.7 * r_co + rng.randn(n_inst) * 0.003
    for i, inst in enumerate(insts):
        rows.append({
            "trading_day": td, "root": inst, "instrument": inst,
            "r_co": r_co[i], "r_oc": r_oc[i],
        })
panel = pd.DataFrame(rows)
panel["y"] = -panel["r_oc"]

zero_cost = {"cost_per_contract": 0.0, "slippage_bps_open": 0.0, "slippage_bps_close": 0.0}
real_cost = {"cost_per_contract": 2.5, "slippage_bps_open": 1.0, "slippage_bps_close": 1.0}

scenarios = {
    "Oracle (y=-r_oc)":       panel["y"].values,
    "Baseline (-r_co)":       panel["r_co"].values,       # alpha_low_long negates → alpha=-r_co
    "Anti-oracle (r_oc)":     (-panel["y"]).values,
    "Random":                 np.random.RandomState(99).randn(len(panel)),
}

header = f"{'Scenario':<25} {'Sharpe':>8} {'Mean':>10} {'Vol':>8} {'HitRate':>8} {'MaxDD':>8} {'Skew':>7} {'Kurt':>7} {'CVaR1%':>8} {'Zero%':>6}"
sep = "-" * len(header)

for cost_label, cost_cfg in [("Zero-cost", zero_cost), ("Realistic cost", real_cost)]:
    print(f"\n=== {cost_label} ===")
    print(header)
    print(sep)
    for name, preds in scenarios.items():
        r = evaluate_trade_proxy(dataset=panel, preds=preds, config=cost_cfg)
        zero_pct = r.n_zero_return_days / max(r.n_days, 1) * 100
        print(f"{name:<25} {r.sharpe_model:>8.2f} {r.mean_daily_return:>10.6f} {r.vol:>8.5f} {r.hit_rate:>8.1%} {r.max_drawdown:>8.3%} {r.skew:>7.2f} {r.kurtosis:>7.2f} {r.cvar_1pct:>8.5f} {zero_pct:>5.1f}%")
