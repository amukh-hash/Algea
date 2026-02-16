---
description: End-to-end evaluation protocol for CO→OC reversal futures (universe expansion → training → shadow eval → promotion decision)
---

# CO→OC Reversal Futures Evaluation Protocol

## Phase 0: Preconditions

### 0.1 Contract Qualification Sanity Pass (STOP CONDITION)
// turbo
```
python -m sleeves.cooc_reversal_futures.run_contract_qualification --dry-run --output runs/contract_spec_check_report.json
```
- Check `runs/contract_spec_check_report.json` — `all_passed` must be `true`
- If any root fails, fix `contract_master.py` or `ibkr_contracts.py` before proceeding

### 0.2 With IBKR connected (optional, recommended before production training):
```
python -m sleeves.cooc_reversal_futures.run_contract_qualification --output runs/contract_spec_check_report_ibkr.json
```

### 0.3 Decide schema version
- **Recommended:** Start with V2 (`features.schema_version: 2`) for the first training run
- V3 is opt-in via `features.schema_version: 3` — run as a second candidate

---

## Phase 1: Train & Validate on IBKR Historical

### 1.1 Run A: V2 Training (baseline promotion attempt)
```
python -m sleeves.cooc_reversal_futures.pipeline.cli --config sleeves/cooc_reversal_futures/cooc_reversal_futures.yaml --data-provider ibkr_hist --run-dir runs/v2_ibkr_14root
```

Expected outputs in `runs/v2_ibkr_14root/`:
- `production_pack/` (model + scaler + promotion report)
- `validation_report.json`
- `promotion_windows_report.json`
- `regime_slices_report.json`

### 1.2 Run B: V3 Training (candidate)
Edit `cooc_reversal_futures.yaml`: `features.schema_version: 3`
```
python -m sleeves.cooc_reversal_futures.pipeline.cli --config sleeves/cooc_reversal_futures/cooc_reversal_futures.yaml --data-provider ibkr_hist --run-dir runs/v3_ibkr_14root
```

### 1.3 Check Risk Calibration
Both runs should produce `risk_calibration_report.json`. Verify:
- Spearman correlation > 0
- Bucket monotonicity: rising mean |r_oc| across quintiles

### 1.4 Check Score Mode Comparison
Both runs should produce `score_mode_comparison.json`. Notes:
- If risk-adjusted has better worst-tail and drawdown → use derived score
- If raw has higher mean but worse tails → use raw cautiously

### 1.5 Confirm Promotion Status
Check `promotion_windows_report.json` for `promotion_status: PASS`
- If FAIL → paper trade in HEURISTIC, use MODEL for shadow eval only

---

## Phase 2: Paper Trading (HEURISTIC live, MODEL shadow)

### 2.1 Start paper trading in HEURISTIC mode
Daily cadence:
1. Open phase: submit entry orders
2. Close phase: flatten all positions
3. Reconcile: pull fills/positions, write daily artifacts

### 2.2 Run daily shadow evaluation (after each reconcile)
// turbo
```
python -m sleeves.cooc_reversal_futures.run_shadow_eval_cooc_ibkr --asof <YYYY-MM-DD> --pack-dir runs/v2_ibkr_14root/production_pack --artifacts-dir runs/paper_trading --output-dir runs/shadow_eval
```

Replace `<YYYY-MM-DD>` with today's date.

---

## Phase 3: Promotion Decision (after 30-60 sessions)

### 3.1 Aggregate shadow eval results
// turbo
```
python -m sleeves.cooc_reversal_futures.run_shadow_eval_aggregator --shadow-dir runs/shadow_eval --output runs/shadow_eval_summary.json --min-sessions 30
```

### 3.2 Review decision
Check `runs/shadow_eval_summary.json` → `decision.promote`:
- `true` → switch to MODEL execution
- `false` → keep HEURISTIC, investigate which checks failed

Decision criteria:
- worst 1% day improves by ≥ 10 bps
- max drawdown improves
- mean daily return not degraded by > 5 bps
- results not concentrated in outlier days

---

## Tactical Notes

- Keep shock `gross_multiplier_on_shock: 0.5` initially (conservative)
- After 30-60 days, tune to 0.25 if tails still dominate, or 0.0 if shock days are consistently toxic
- If any root causes recurring RESEARCH_ONLY flags, temporarily remove and re-run
