# RUNBOOK: IBKR Paper Trading — CO→OC Reversal Futures

## Environment Variables

Set these before running any commands:

```bash
export IBKR_GATEWAY_URL="127.0.0.1:7497"    # TWS paper port
export IBKR_ACCOUNT_ID="<your_paper_account>" # e.g. DUP*****2I
export IBKR_CLIENT_ID="17"                    # unique per connection
export IBKR_PAPER_ONLY="1"                    # enforce paper checks
export IBKR_READONLY="0"                      # set to "1" for read-only
```

> **Security**: Never log the full account ID. The broker masks it automatically.

---

## Prerequisites

1. **TWS or IB Gateway** running in paper mode on port 7497
2. **API settings enabled** in TWS: File → Global Configuration → API → Settings
   - Enable ActiveX and Socket Clients
   - Socket port: 7497 (paper) or 4002 (Gateway paper)
   - Allow connections from localhost
3. **ib_insync** installed: `pip install ib_insync==0.9.86`

---

## Daily Schedule (New York Time)

| Time (ET) | Phase | Command |
|-----------|-------|---------|
| 09:25 | Build signals | `--phase open --mode noop` (dry run) |
| 09:29:45 | Submit entries | `--phase open --mode ibkr` |
| 15:58 | Flatten positions | `--phase close --mode ibkr` |
| 16:10 | Reconcile | `--phase reconcile --mode ibkr` |

---

## Commands

### 1. Update Daily Data (run before open)

```bash
python backend/scripts/data/update_futures_daily.py
```

### 2. Phase: Open (entry orders)

**Dry run (noop)**:
```bash
python backend/scripts/paper/run_paper_cycle_ibkr.py \
    --config backend/configs/cooc_reversal_futures.yaml \
    --inputs data_cache/canonical_futures_daily.parquet \
    --asof 2026-02-14 \
    --phase open \
    --mode noop
```

**Live paper**:
```bash
python backend/scripts/paper/run_paper_cycle_ibkr.py \
    --config backend/configs/cooc_reversal_futures.yaml \
    --inputs data_cache/canonical_futures_daily.parquet \
    --asof 2026-02-14 \
    --phase open \
    --mode ibkr
```

### 3. Phase: Close (flatten)

```bash
python backend/scripts/paper/run_paper_cycle_ibkr.py \
    --config backend/configs/cooc_reversal_futures.yaml \
    --asof 2026-02-14 \
    --phase close \
    --mode ibkr
```

### 4. Phase: Reconcile

```bash
python backend/scripts/paper/run_paper_cycle_ibkr.py \
    --config backend/configs/cooc_reversal_futures.yaml \
    --asof 2026-02-14 \
    --phase reconcile \
    --mode ibkr
```

---

## Output Artifacts

```
data_lake/futures/paper/<asof>/
├── open/
│   ├── intents_open.json
│   ├── order_intents_open.parquet
│   ├── orders_open.parquet
│   └── guard_report_open.json
├── close/
│   ├── intents_close.json
│   └── orders_close.parquet
└── reconcile/
    ├── fills_<asof>.parquet
    └── reconciliation_<asof>.json
```

---

## First-Day Protocol

For the very first paper trading day, override guard limits in config:

```yaml
paper:
  max_contracts_per_order: 1
  max_contracts_per_instrument: 1
  max_gross_notional: 500000.0
```

This limits exposure to 1 contract per instrument until you verify fills are working correctly.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ConnectionError: Failed to connect` | Ensure TWS/Gateway is running and API is enabled |
| `PermissionError: readonly mode` | Set `IBKR_READONLY=0` |
| Orders rejected | Check contract qualification; ensure paper account has margin |
| No fills appearing | Wait 30s after placement; check `ib.trades()` manually |
| `ValueError: Cannot parse` | Active contract format mismatch; check roll logic |

---

## Signal Mode

The runner auto-detects signal mode:

- **Heuristic** (default): Uses `-ret_co` reversal baseline. Active when no model pack exists or `model_sanity` gate failed.
- **Model**: Uses trained Ridge model from production pack. Only activates when `model_sanity` gate passed.

Override in config:
```yaml
signal:
  mode: heuristic  # or "model" or "auto"
```

---

## Shadow Evaluator (MODEL vs HEURISTIC Comparison)

After accumulating several days of paper fills, run the shadow evaluator to compare actual HEURISTIC fills against MODEL counterfactual performance:

```bash
python backend/scripts/paper/run_shadow_eval_cooc_ibkr.py \
    --fills-dir data_lake/futures/paper/*/reconcile \
    --intents-dir data_lake/futures/paper/*/open \
    --pack-dir packs/cooc_latest \
    --output-dir output/shadow_eval \
    --start-date 2025-01-01 \
    --end-date 2025-01-31
```

Outputs:
- `shadow_eval_report.json`: side-by-side Sharpe, hit rate, PnL metrics
- `shadow_fills.parquet`: consolidated fill data
- `shadow_intents.parquet`: consolidated intent data
