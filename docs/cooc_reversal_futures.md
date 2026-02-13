# CO→OC Reversal Futures Sleeve (Pattern B)

## Train
```bash
python -m sleeves.cooc_reversal_futures.model.train
```
Use config `COOCReversalConfig.model` (backend, d_model, layers, heads, dropout, epochs, lr, batch).

## Backtest
```bash
pytest backend/tests/test_cooc_reversal_futures.py
```
The integration backtest test uses the same anchor + feature path used by sleeve methods.

## Live (paper)
1. Instantiate `COOCReversalFuturesSleeve`.
2. Freeze features before `09:30:00 ET` and run `assert_feature_provenance`.
3. Generate orders via `build_daily_orders(...)`.
4. Submit entry orders near open, close all via `force_eod_flatten(...)` near `16:00 ET`.
5. If non-flat by `16:10 ET`, run emergency flatten and alert.

## Config reference
- Universe, micros, anchor methods, lookback, micro windows.
- Gross target, CAUTION scaling, per-instrument caps, net cap.
- Price-limit gating policy.
- Model backend + hyperparameters.
- CV fold/embargo parameters.

## Operational runbook
- **Data outage near open:** block entries (`no_trade_flags`) and emit alert.
- **Partial fills:** recompute residual contracts and retry with bounded backoff.
- **Limit/circuit regimes:** if CRASH_RISK or price-limit policy triggers, set target contracts to zero.
- **EOD unflattened:** execute emergency flatten path and raise an ops notification.
