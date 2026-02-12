# Options Volatility Risk Premium (VRP) Sleeve

## What is VRP?

The **Volatility Risk Premium** is the empirical tendency for implied volatility to exceed realised volatility. Option buyers pay an "insurance premium" for downside protection; option sellers collect this premium systematically. The VRP sleeve monetises this structural edge by selling defined-risk credit spreads on index ETFs.

## Why Defined-Risk Spreads?

Naked short options have unlimited downside. We **always** use credit spreads (e.g., short put + long put at lower strike), which cap the worst-case loss to the spread width minus the credit received. This makes risk fully quantifiable and bounded.

### Supported Structures
- **PUT_CREDIT_SPREAD**: Short put (15–20 delta) + long put (further OTM). Collects premium from elevated put IV.
- **IRON_CONDOR** (optional): Symmetric credit spreads on both put and call sides. Used when skew is balanced and regime is stable.

## Universe

SPY, QQQ, IWM — highly liquid index ETFs with tight bid-ask spreads and deep open interest.

## Regime Filtering

The sleeve uses a rules-based regime classifier with three states:

| Regime | Description | Trading |
|--------|-------------|---------|
| **NORMAL_CARRY** | Vol stable, no stress signals | Full size entries |
| **CAUTION** | Mild stress signals | Half size entries |
| **CRASH_RISK** | High-stress cluster detected | No new entries; force close |

### Signals Used
- VIX level and 5-day change
- VIX term structure (contango vs backwardation)
- Realised vol ratio (RV10/RV60)
- Equity drawdown from 63-day high
- 50/200 MA trend filter
- Credit proxy (e.g., HYG/LQD ratio change)

## Entry Conditions

1. `iv_rank_252 >= 0.50` (IV is elevated vs. recent history)
2. Regime is `NORMAL_CARRY` (or `CAUTION` at reduced size)
3. Term structure not in extreme backwardation
4. Liquidity constraints met (min OI, max spread %)
5. Per-structure max loss ≤ 2% of NAV
6. Aggregate VRP risk ≤ 10% of NAV

## Exit Rules

- **Profit take**: Close when 50% of credit captured
- **Stop loss**: Close when mark-to-close ≥ 2× entry credit
- **Time exit**: Force close at DTE ≤ 7
- **Regime exit**: In CRASH_RISK, force close at next available mark

## Risk Management

### Limits (Configurable)
| Limit | Default |
|-------|---------|
| Max loss per structure | 2% NAV |
| Total VRP risk | 10% NAV |
| VRP notional allocation | 25% NAV |
| Max positions per underlying | 1 |
| Vega limit per underlying | 500 |
| Gamma limit per underlying | 50 |

### Scenario Shocks
PnL computed under:
- Spot: ±5%, ±10%, −20%
- Vol: +25%, +50%
- Crash combo: −10% spot + +50% vol

### Sleeve Vol Targeting
Target 8% annualised. A risk scaler adjusts new trade sizing based on recent PnL volatility (bounded [0.25, 2.0]).

### Convexity Controls
Tracks aggregate short convexity score (gamma + vega exposure). Forced reduction when score exceeds limit.

## Capital Allocation (MetaAllocator)

The MetaAllocator combines core equity and VRP sleeves using a grid-search optimisation over the tail-adjusted Sharpe ratio:

```
TailAdjustedSharpe = (E[Rp] - rf) / max(Vol_p, λ · ES95_p) - m · MaxDD
```

- Grid: `w_vrp ∈ [0, 0.25]` in 1% steps
- Correlation estimated from trailing 60-day returns
- In CRASH_RISK: `w_vrp = 0` unconditionally

## Failure Modes

1. **Crash clustering**: Multiple positions expire ITM simultaneously. Mitigated by position limits and regime gating.
2. **Vol regime shifts**: Mean reversion breaks down in sustained high-vol. Mitigated by vol targeting and risk scaler.
3. **Liquidity drying up**: Wide spreads in panic markets. Mitigated by min OI/volume gates and slippage model.
4. **Assignment risk**: Short puts may be assigned early. Mitigated by force-close at DTE ≤ 7.

## How to Run

### Backtest
```python
from algaie.execution.options.config import VRPConfig
from algaie.execution.options.vrp_strategy import VRPStrategy

config = VRPConfig(underlyings=("SPY",), delta_target=0.15)
strategy = VRPStrategy(config)
# ... wire up chain data, run predict() loop, use backtest_adapter
```

### Tests
```bash
python -m pytest backend/tests/options_vrp/ -v
```

### Risk Report
```python
from algaie.eval.vrp_report import build_vrp_report
report = build_vrp_report(as_of_date, backtest=backtest_result, positions=pf)
print(report.to_dict())
```

## Live Trading Considerations

- Use real-time chain data (not EOD snapshots)
- Account for halts and price uncertainty during fast markets
- Implement IBKR/Alpaca options execution adapter (stub exists)
- Monitor assignment notifications and handle exercise
- Run regime classifier every 15 min intraday if VIX spikes
