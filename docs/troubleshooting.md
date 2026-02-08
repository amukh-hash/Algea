# Troubleshooting

## Invalid return encountered
- Ensure `close > 0`.
- The canonical validator quarantines invalid return rows to:
  `backend/artifacts/reports/data_quality/invalid_returns.parquet`.

## Quantile head mismatch
- Quantile heads are disabled by default (`enable_quantiles=False`).
- When enabled, ensure encoder pooling uses hidden states `[B, L, D]`.

## Schema validation failures
- Missing columns or non-finite values throw errors with the file path.
- Re-run canonicalization or feature build with fixed inputs.

## Broker reconciliation failures
- Validate broker credentials are set via environment variables.
- Re-run reconciliation after corporate action updates (splits may change quantities).

## Rounding policy issues
- If live brokers reject fractional shares, set rounding policy to `round`.
- Use `noop` live mode to verify intents before submitting.

## Slippage / transaction cost modeling
- Increase `slippage_bps` or `slippage_volume_impact` to stress-test fills.
- Use commission settings in config to validate fee impact on net returns.
