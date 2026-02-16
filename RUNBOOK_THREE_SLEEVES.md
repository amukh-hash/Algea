# RUNBOOK: Three-Sleeve IBKR Paper Trading

## Quick Start

```bash
# 1. Preflight — verify connection
python scripts/ib_preflight.py

# 2. Dry run — compute signals, no orders
python scripts/run_three_sleeves_ibkr.py --mode noop --asof 2026-02-17

# 3. Live paper — submit orders
python scripts/run_three_sleeves_ibkr.py --mode ibkr --asof 2026-02-17
```

---

## Prerequisites

1. **TWS** running in paper mode on port **7497**
2. **API settings** in TWS: File → Global Configuration → API → Settings
   - ✅ Enable ActiveX and Socket Clients
   - ✅ Read-Only API **unchecked**
   - Socket port: **7497**
   - Allow connections from localhost
3. **ib_insync** installed: `pip install ib_insync`
4. **`.env`** configured with:
   ```
   IBKR_GATEWAY_URL=127.0.0.1:7497
   IBKR_ACCOUNT_ID=<your_paper_account>
   IBKR_CLIENT_ID=17
   IBKR_PAPER_ONLY=1
   IBKR_READONLY=0
   ```

---

## Sleeve Capital Split

| Sleeve | Allocation | Instruments |
|--------|-----------|-------------|
| **Core** (CO→OC) | 50% | ES, NQ, RTY, YM futures |
| **VRP** | 30% | SPY options (noop week 1) |
| **Selector** | 20% | Individual equities (top/bottom 10) |

Capital is computed as `Total NLV × allocation %` at runtime.

---

## Sleeve Details

### 1. Core (CO→OC Reversal Futures)

Delegates to the existing `run_paper_cycle_ibkr.py` infrastructure. Uses the YAML config at `sleeves/cooc_reversal_futures/cooc_reversal_futures.yaml`. See the dedicated [RUNBOOK_PAPER_IBKR.md](sleeves/cooc_reversal_futures/RUNBOOK_PAPER_IBKR.md) for full details.

### 2. VRP (Options)

**Week 1: NOOP only** — computes target positions but does not submit orders.

> ⚠️ **SPX sizing warning**: 1 SPX contract = ~$500,000 notional. This can dominate or blow through your entire paper account. Week-1 noop mode prevents accidental oversizing. When ready, use `--vrp-mode ibkr` to enable (with SPY, not SPX).

### 3. Selector (Individual Equities)

Loads scored rankings from `backend/data/selector/runs/SEL-PROD-CANDIDATE/scored_test.parquet`. Picks the top N and bottom N stocks by `score_final`, creates market-neutral equal-weight positions.

```
AAPL, MSFT, GOOGL, ...  ← top 10 (LONG)
XYZ, ABC, DEF, ...      ← bottom 10 (SHORT)
```

---

## Daily Schedule (New York Time)

| Time (ET) | Action | Command |
|-----------|--------|---------|
| 09:20 | Preflight check | `python scripts/ib_preflight.py` |
| 09:25 | Dry run (review signals) | `python scripts/run_three_sleeves_ibkr.py --mode noop --asof YYYY-MM-DD` |
| 09:29 | Submit orders | `python scripts/run_three_sleeves_ibkr.py --mode ibkr --asof YYYY-MM-DD` |
| Intraday | Monitor in TWS | Check positions, margin, Greeks |
| 15:58 | Flatten Core futures | `python backend/scripts/paper/run_paper_cycle_ibkr.py --phase close --mode ibkr --asof YYYY-MM-DD` |

---

## CLI Reference

```
python scripts/run_three_sleeves_ibkr.py [OPTIONS]

Options:
  --mode {noop,ibkr}      Execution mode (default: noop)
  --asof YYYY-MM-DD       Trading date (default: today)
  --vrp-mode {noop,ibkr}  Override VRP mode (default: always noop)
  --selector-top-n N      Top/bottom stocks count (default: 10)
```

---

## Output Artifacts

```
data_lake/three_sleeves/paper/<asof>/
└── three_sleeve_report.json     ← combined report for all sleeves
```

---

## Monitoring Checklist

While running, watch in TWS / Client Portal:

- [ ] Net liquidation value
- [ ] Margin cushion > 30%
- [ ] Position exposure by sleeve
- [ ] Greeks (for VRP when enabled)
- [ ] No unexpected cross-sleeve concentration

---

## ⚠️ Paper Account Limitations

- Fills are unrealistically good
- Options fills are optimistic
- Slippage is understated

**Focus on**: exposure behavior, risk stability, cross-sleeve interaction — not raw PnL quality.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ConnectionError` | Ensure TWS is running and API is enabled |
| `PermissionError: readonly` | Set `IBKR_READONLY=0` in `.env` |
| Selector data missing | Run selector inference first |
| Core inputs missing | Run `python backend/scripts/data/update_futures_daily.py` |
| Margin warning | Reduce selector `--selector-top-n` or check Core sizing |
