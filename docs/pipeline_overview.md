# Pipeline Overview

1. **canonicalize_daily**
   - Validates canonical OHLCV and writes partitioned parquet.
2. **build_eligibility**
   - Builds universe eligibility as-of date.
3. **build_features**
   - Computes feature frame from canonical data.
4. **train_foundation_model**
   - Trains foundation model scaffold (Chronos-2 placeholder).
5. **build_priors**
   - Generates priors using the foundation model.
6. **train_ranker_model**
   - Trains ranker model scaffold.
7. **run_nightly_cycle**
   - Orchestrates nightly production sequence and writes summary report.
8. **research → backtest → paper → live**
   - Research: generate signals and walk-forward evaluations.
   - Backtest: produce equity curves, trade logs, orders, and metrics.
   - Paper: submit broker orders and write reconciliation artifacts.
   - Live: write intents (noop) or submit live orders via broker adapter.
