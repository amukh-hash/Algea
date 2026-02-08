# Data Contracts

## Canonical Daily
**Path:** `backend/artifacts/canonical/daily/ticker=XYZ/data.parquet`

Required columns:
- `date`, `open`, `high`, `low`, `close`, `volume`

Optional columns:
- `vwap`, `vix`, `rate_proxy`

Rules:
- `close > 0`
- `high >= max(open, close)`
- `low <= min(open, close)`
- monotonic dates, no duplicates
- no zero-filled rows

## Eligibility Frame
**Path:** `backend/artifacts/eligibility/asof=YYYY-MM-DD/eligibility.parquet`

Required columns:
- `date`, `ticker`, `is_eligible`

Optional:
- `reason_codes`

Rule: eligibility as-of date uses only data `<= T`.

## Feature Frame
**Path:** `backend/artifacts/features/features.parquet`

Required columns:
- `date`, `ticker`, feature columns

Rules:
- Strict finite checks
- Out-of-bounds warn + quarantine by default

## Priors Frame
**Path:** `backend/artifacts/priors/date=YYYY-MM-DD/priors.parquet`

Required columns:
- `date`, `ticker`, `p_mu5`, `p_mu10`, `p_sig5`, `p_sig10`, `p_pdown5`, `p_pdown10`

Rules:
- `p_sig* > 0`
- `p_pdown*` in `[0, 1]`

## Signals
**Path:** `backend/artifacts/signals/date=YYYY-MM-DD/signals.parquet`

Required columns:
- `date`, `ticker`, `score`, `rank`

## Backtest Artifacts
**Path:** `backend/artifacts/backtests/<run_id>/`

Required outputs:
- `equity_curve.parquet` (date, equity, cash, gross_exposure, net_exposure)
- `trades.parquet` (entry_date, exit_date, ticker, qty, entry_px, exit_px, pnl, ret, hold_days, reason)
- `orders.parquet` (date, ticker, qty, side, fill_px, status)
- `orders.parquet` includes `commission`
- `metrics.json`
- `summary.md`

## Paper Trading Artifacts
**Path:** `backend/artifacts/paper/<run_id>/`

Required outputs:
- `order_intents.parquet`
- `orders.parquet`
- `reconciliation_YYYY-MM-DD.json`

## Live Trading Artifacts
**Path:** `backend/artifacts/live/<run_id>/`

Required outputs:
- `intents.json` (noop mode) or `orders.parquet` (live mode)
