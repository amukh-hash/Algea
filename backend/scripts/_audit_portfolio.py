"""One-shot portfolio sanity audit."""
import json
import numpy as np

RUN = r"C:\Users\Aishik\Documents\Workshop\Algaie\backend\data\selector\runs\SEL-20260211-141209"

pm = json.load(open(f"{RUN}/portfolio_metrics.json"))
rp = json.load(open(f"{RUN}/selector_full_report.json"))

v = pm["ann_vol_net"]
s = pm["net_sharpe"]
c = pm["cagr_net"]
r = pm["ann_return_net"]

print("=" * 60)
print("PORTFOLIO SANITY AUDIT")
print("=" * 60)

# 1. Sharpe x Vol identity
print("\n1. CAGR vs Sharpe x Vol")
print(f"   Net Sharpe:     {s:.4f}")
print(f"   Ann Vol (net):  {v*100:.1f}%")
print(f"   Sharpe x Vol:   {s*v*100:.1f}%")
print(f"   Ann Return:     {r*100:.1f}%")
print(f"   CAGR:           {c*100:.1f}%")
match = "YES" if abs(s * v - r) < 0.02 else "NO"
print(f"   Sharpe*Vol ~= AnnRet?  {match}")

# 2. Vol diagnosis
print("\n2. VOLATILITY DIAGNOSIS")
print(f"   Ann vol (gross):  {pm['ann_vol_gross']*100:.1f}%")
print(f"   Ann vol (net):    {v*100:.1f}%")
print(f"   Expected for long-only top-50: ~15-25%")
print(f"   Actual / expected: {v/0.20:.1f}x")
print(f"   Scaled vol (/ sqrt(10)): {v/np.sqrt(10)*100:.1f}%")
print(f"   DIAGNOSIS: y_ret is 10d log return, rebalanced daily")
print(f"   -> each 'daily' period return = weighted-avg of 10d forward returns")
print(f"   -> overlapping windows inflate per-period variance by ~10x")

# 3. Leverage
print("\n3. LEVERAGE")
print(f"   Portfolio is long-only, equal-weight, sum(|w|)=1")
print(f"   No explicit leverage. Vol scaling clamp=[0.5, 2.0]")
vs_vol = pm.get("vol_scaled_vol", 0)
if vs_vol > 0:
    print(f"   Vol-scaled vol: {vs_vol*100:.1f}%  (before: {v*100:.1f}%)")
    print(f"   Ratio: {vs_vol/v:.2f}x  (vol scaling reduces, not amplifies)")

# 4. Turnover
print("\n4. TURNOVER")
print(f"   Avg daily turnover: {pm['avg_turnover']*100:.1f}%")
print(f"   Ann turnover: {pm['ann_turnover']:.0f}")
print(f"   Avg holding: {pm['avg_holding_period']:.1f}d")
print(f"   83% daily turnover -> ~42 of 50 stocks change each day")
print(f"   This is VERY HIGH for a 10d horizon signal")

# 5. Root cause
print("\n5. ROOT CAUSE OF INFLATED METRICS")
print(f"   y_ret = log(P[t+10]/P[t])  (10d forward return)")
print(f"   Portfolio evaluates sum(w*y_ret) EVERY day")
print(f"   But y_ret[t] and y_ret[t+1] share 9/10 of the same period")
print(f"   -> returns are NOT independent across days")
print(f"   -> compounding them treats 10d gains as if earned daily")
print(f"   -> CAGR, vol, and return are all inflated")
print(f"   -> Sharpe is ~correct (inflation cancels in ratio)")

# 6. Suggested fix
print("\n6. FIX OPTIONS")
print(f"   A) Rebalance every 10 days (non-overlapping windows)")
print(f"      -> select stocks on day t, hold 10 days, measure realized return")
print(f"      -> vol will drop to ~20-35%, CAGR to ~20-35%")
print(f"   B) Use daily returns for PnL (y_ret_1d)")
print(f"      -> select stocks using 10d signal, but measure 1d return")
print(f"      -> requires daily return column in priors_frame")
print(f"   C) Divide y_ret by holding period (approximate)")
print(f"      -> treat y_ret/10 as daily-equivalent return")
print(f"      -> quick fix but slightly wrong for log returns")
print(f"")
print(f"   RECOMMENDED: Option A (10d rebalance, non-overlapping)")
