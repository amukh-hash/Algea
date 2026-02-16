# ALGAIE SYSTEM INTEGRATION WORKFLOWS

*End-to-End Operational Pipelines & Integration Patterns*

---

## OVERVIEW

This document details the complete integration workflows for the Algaie trading system, showing how all components connect in production environments. It covers nightly operations, weekly retraining, live trading integration, and error handling patterns.

**Workflow Categories**:
1. Nightly Production Cycle
2. Weekly Retraining Workflow
3. Live Trading Integration
4. Monitoring & Alerting
5. Disaster Recovery & Failover

---

## 1. NIGHTLY PRODUCTION CYCLE

### 1.1 Complete Nightly Workflow (6 PM → 9:30 AM)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NIGHTLY PRODUCTION WORKFLOW                           │
│              (Automated daily execution after market close)              │
└─────────────────────────────────────────────────────────────────────────┘

TIME: 6:00 PM ET (Monday-Friday, trading days only)
TRIGGER: Cron job OR market close event
DURATION: ~3.5 minutes total

PHASE 1: DATA ACQUISITION
═══════════════════════════════════════════════════════════════════════
┌───────────────────────────────────────────────────────────────────┐
│ 6:00:00 PM - Start nightly cycle                                  │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ $ python backend/scripts/run/run_nightly_cycle.py \\        │  │
│ │     --asof $(date +%Y-%m-%d) \\                             │  │
│ │     --mode production                                        │  │
│ │                                                              │  │
│ │ Args:                                                        │  │
│ │   --asof: Date to process (default: today)                  │  │
│ │   --mode: production | backtest | dry-run                   │  │
│ └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│ 6:00:01 PM - Initialize logging & check preconditions             │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ ✓ Trading day check (NYSE calendar)                         │  │
│ │   - If weekend/holiday → EXIT (no action)                   │  │
│ │                                                              │  │
│ │ ✓ Market close confirmation                                 │  │
│ │   - Query market status API                                 │  │
│ │   - If market still open → WAIT or EXIT                     │  │
│ │                                                              │  │
│ │ ✓ Artifact locks check                                      │  │
│ │   - Check for running nightly process (PID file)            │  │
│ │   - If locked → EXIT (prevent concurrent runs)              │  │
│ │                                                              │  │
│ │ ✓ Load configuration                                        │  │
│ │   - PipelineConfig from config/production.yaml              │  │
│ │   - Validate all required fields present                    │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 6:00:05 PM - Fetch raw OHLCV data                                 │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ Data sources (priority order):                              │  │
│ │   1. Yahoo Finance (primary, free)                          │  │
│ │   2. Polygon.io (backup, requires API key)                  │  │
│ │   3. Alpha Vantage (fallback, rate limited)                 │  │
│ │                                                              │  │
│ │ Execution:                                                   │  │
│ │   from algaie.data.ingest import fetch_ohlcv_batch          │  │
│ │                                                              │  │
│ │   symbols = load_universe_symbols()  # 500 symbols          │  │
│ │   raw_data = fetch_ohlcv_batch(                             │  │
│ │       symbols=symbols,                                      │  │
│ │       start_date=asof_date,                                 │  │
│ │       end_date=asof_date,                                   │  │
│ │       provider="yahoo",                                     │  │
│ │       max_concurrent=20  # Parallel requests                │  │
│ │   )                                                          │  │
│ │                                                              │  │
│ │ Result:                                                      │  │
│ │   - 497 symbols fetched successfully                        │  │
│ │   - 3 symbols failed (retry with backup)                    │  │
│ │   - Final: 500/500 symbols (100% coverage)                  │  │
│ │                                                              │  │
│ │ Duration: ~2 minutes (API rate limited)                     │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 6:02:05 PM - Canonicalize & validate data                         │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ Per-symbol processing (parallel, 20 workers):               │  │
│ │                                                              │  │
│ │ for symbol in symbols:                                      │  │
│ │     try:                                                     │  │
│ │         df = raw_data[symbol]                               │  │
│ │         df = normalize_schema(df)  # ticker→symbol, etc.    │  │
│ │         validate_canonical_ohlcv(df, config)  # 8 stages    │  │
│ │         write_canonical(df, symbol)  # Parquet write        │  │
│ │         log.info(f"{symbol}: OK")                           │  │
│ │     except ValidationError as e:                            │  │
│ │         quarantine(symbol, e)  # Write to quarantine        │  │
│ │         log.warning(f"{symbol}: QUARANTINED - {e}")         │  │
│ │                                                              │  │
│ │ Validation results:                                         │  │
│ │   ✓ 497 symbols validated and written                       │  │
│ │   ⚠ 3 symbols quarantined (split issues)                    │  │
│ │   → Continue with 497 symbols                               │  │
│ │                                                              │  │
│ │ Duration: ~30 seconds (Polars parallel writes)              │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

PHASE 2: UNIVERSE & ENRICHMENT
═══════════════════════════════════════════════════════════════════════
┌───────────────────────────────────────────────────────────────────┐
│ 6:02:35 PM - Rebuild universe for today                           │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ from algaie.data.eligibility import UniverseBuilder         │  │
│ │                                                              │  │
│ │ builder = UniverseBuilder(config.universe)                  │  │
│ │ universe = builder.build(                                   │  │
│ │     canonical_dir="artifacts/canonical/",                   │  │
│ │     asof_date=asof_date                                     │  │
│ │ )                                                            │  │
│ │                                                              │  │
│ │ Filters applied:                                            │  │
│ │   1. Observable: close > $1, hist_days >= 252               │  │
│ │      Result: 485/497 symbols                                │  │
│ │   2. Tradable: dollar_vol >= $5M                            │  │
│ │      Result: 450/485 symbols                                │  │
│ │   3. Tier assignment (0/1/2 based on dollar vol)            │  │
│ │      Tier 0: 150 symbols (mega cap)                         │  │
│ │      Tier 1: 153 symbols (mid cap)                          │  │
│ │      Tier 2: 147 symbols (small cap)                        │  │
│ │   4. Weight normalization (tier-weighted)                   │  │
│ │                                                              │  │
│ │ Write: artifacts/eligibility/universe_20240115.parquet      │  │
│ │ Symlink: universe_latest.parquet → universe_20240115.parquet│  │
│ │                                                              │  │
│ │ Duration: ~5 seconds                                        │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 6:02:40 PM - Compute market enrichment (parallel)                 │
│                                                                    │
│ ┌─────────────────────────────────────┐  ┌──────────────────────┐│
│ │ Thread 1: Breadth Indicators        │  │ Thread 2: Covariates ││
│ │                                     │  │                      ││
│ │ from algaie.data.market import:     │  │ from algaie.data     ││
│ │   build_breadth_daily               │  │ .market import:      ││
│ │                                     │  │   build_covariates   ││
│ │ breadth = build_breadth_daily(      │  │                      ││
│ │     canonical_dir,                  │  │ cov = build_covariates(││
│ │     universe,                       │  │     date=asof_date   ││
│ │     date=asof_date                  │  │ )                    ││
│ │ )                                    │  │                      ││
│ │                                     │  │ Fetch indices:       ││
│ │ Compute:                            │  │   SPY, QQQ, IWM, IEF ││
│ │   AD Line (advancers-decliners)     │  │                      ││
│ │   BPI 21-day (% up days)            │  │ Compute:             ││
│ │                                     │  │   Index returns      ││
│ │ Result:                             │  │   RV21 change        ││
│ │   ad_line = 0.24                    │  │                      ││
│ │   bpi_21d = 0.67                    │  │ Result:              ││
│ │                                     │  │   spy_ret = 0.015    ││
│ │ Duration: ~10 sec                   │  │   rv21_chg = 0.014   ││
│ │                                     │  │                      ││
│ │                                     │  │ Duration: ~5 sec     ││
│ └─────────────────────────────────────┘  └──────────────────────┘│
│                                                                    │
│ 6:02:50 PM - Join into MarketFrame                                │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ from algaie.data.features import build_market_frame         │  │
│ │                                                              │  │
│ │ market_frame = build_market_frame(                          │  │
│ │     canonical_dir,                                          │  │
│ │     universe,                                               │  │
│ │     breadth,                                                │  │
│ │     covariates,                                             │  │
│ │     asof_date                                               │  │
│ │ )                                                            │  │
│ │                                                              │  │
│ │ Join keys: (symbol, date)                                   │  │
│ │ Output shape: 450 symbols × 1 date = 450 rows               │  │
│ │ Columns: OHLCV + covariates + breadth + tier + weight       │  │
│ │                                                              │  │
│ │ Duration: ~1 second                                         │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

PHASE 3: FEATURE ENGINEERING
═══════════════════════════════════════════════════════════════════════
┌───────────────────────────────────────────────────────────────────┐
│ 6:02:51 PM - Build selector features                              │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ from algaie.data.features import SelectorFeatureBuilder     │  │
│ │                                                              │  │
│ │ builder = SelectorFeatureBuilder(config.features)           │  │
│ │ features = builder.build(                                   │  │
│ │     market_frame,                                           │  │
│ │     asof_date,                                              │  │
│ │     compute_targets=False  # No fwd returns in inference    │  │
│ │ )                                                            │  │
│ │                                                              │  │
│ │ Pipeline stages:                                            │  │
│ │   1. Raw features (returns, vol, rel volume)                │  │
│ │   2. Cross-sectional rank normalization [-1, +1]            │  │
│ │   3. Market regime features (breadth, volatility)           │  │
│ │   4. Validate bounds (all ranks in valid range)             │  │
│ │                                                              │  │
│ │ Output shape: 450 rows × 28 feature columns                 │  │
│ │                                                              │  │
│ │ Checkpoint: Write to disk for lineage tracking              │  │
│ │   artifacts/features/selector_features_20240115.parquet     │  │
│ │                                                              │  │
│ │ Duration: ~2 minutes (Polars lazy + rank computation)       │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

PHASE 4: MODEL INFERENCE
═══════════════════════════════════════════════════════════════════════
┌───────────────────────────────────────────────────────────────────┐
│ 6:04:51 PM - Generate Chronos teacher priors (GPU)                │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ from algaie.models.foundation import Chronos2Teacher        │  │
│ │                                                              │  │
│ │ # Load trained checkpoint (from weekly training)            │  │
│ │ checkpoint_path = (                                         │  │
│ │     "artifacts/models/chronos_teacher_v1_lora/"             │  │
│ │ )                                                            │  │
│ │ teacher = Chronos2Teacher.from_pretrained(                  │  │
│ │     checkpoint_path,                                        │  │
│ │     device="cuda"  # GPU acceleration                       │  │
│ │ )                                                            │  │
│ │                                                              │  │
│ │ # Prepare context windows (512 days per symbol)             │  │
│ │ contexts = prepare_contexts(                                │  │
│ │     canonical_dir,                                          │  │
│ │     symbols=universe.tradable_symbols,                      │  │
│ │     end_date=asof_date,                                     │  │
│ │     context_length=512                                      │  │
│ │ )                                                            │  │
│ │                                                              │  │
│ │ # Batched inference                                         │  │
│ │ batch_size = 32                                             │  │
│ │ n_batches = ceil(450 / 32) = 15 batches                     │  │
│ │                                                              │  │
│ │ priors = []                                                 │  │
│ │ for batch_contexts in batched(contexts, batch_size):        │  │
│ │     with torch.no_grad():                                   │  │
│ │         distributions = teacher.predict(batch_contexts)     │  │
│ │         batch_priors = extract_priors(distributions)        │  │
│ │         priors.extend(batch_priors)                         │  │
│ │                                                              │  │
│ │ # Extract aggregate statistics                              │  │
│ │ priors_df = pl.DataFrame([                                  │  │
│ │     {                                                        │  │
│ │         "symbol": symbol,                                   │  │
│ │         "date": asof_date,                                  │  │
│ │         "teacher_drift_20d": prior.drift,                   │  │
│ │         "teacher_vol_20d": prior.vol,                       │  │
│ │         "teacher_tail_risk_20d": prior.tail_risk,           │  │
│ │         "teacher_prob_up_20d": prior.prob_up                │  │
│ │     }                                                        │  │
│ │     for symbol, prior in zip(symbols, priors)               │  │
│ │ ])                                                           │  │
│ │                                                              │  │
│ │ Write: artifacts/priors/chronos_priors_20240115.parquet     │  │
│ │                                                              │  │
│ │ Duration: ~30 seconds (GPU bottleneck)                      │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 6:05:21 PM - Run selector inference                               │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ from algaie.models.ranker import selector_inference         │  │
│ │                                                              │  │
│ │ # Load trained selector (from weekly training)              │  │
│ │ model_path = "artifacts/models/selector_v1_fold3.pt"        │  │
│ │ scaler_path = "artifacts/models/scaler_v1.pkl"              │  │
│ │                                                              │  │
│ │ selector = load_selector_model(model_path, scaler_path)     │  │
│ │                                                              │  │
│ │ # Combine features + priors                                 │  │
│ │ inference_data = features.join(                             │  │
│ │     priors_df,                                              │  │
│ │     on=["symbol", "date"],                                  │  │
│ │     how="inner"                                             │  │
│ │ )                                                            │  │
│ │                                                              │  │
│ │ # Single-batch inference (all 450 symbols together)         │  │
│ │ scores = selector_inference.score_universe(                 │  │
│ │     model=selector,                                         │  │
│ │     features=inference_data,                                │  │
│ │     config=config                                           │  │
│ │ )                                                            │  │
│ │                                                              │  │
│ │ # Cross-sectional ranking                                   │  │
│ │ scores = scores.with_columns([                              │  │
│ │     pl.col("score").rank("ordinal").alias("rank"),          │  │
│ │     (pl.col("score").rank() / pl.col("symbol").count())     │  │
│ │         .alias("percentile")                                │  │
│ │ ])                                                           │  │
│ │                                                              │  │
│ │ Write: artifacts/scores/scores_20240115.parquet             │  │
│ │ Symlink: leaderboard_latest.parquet → scores_20240115.parquet│ │
│ │                                                              │  │
│ │ Duration: ~2 seconds (GPU)                                  │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

PHASE 5: PORTFOLIO CONSTRUCTION
═══════════════════════════════════════════════════════════════════════
┌───────────────────────────────────────────────────────────────────┐
│ 6:05:23 PM - Risk posture assessment                              │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ from algaie.risk import get_risk_posture                    │  │
│ │                                                              │  │
│ │ posture = get_risk_posture(                                 │  │
│ │     market_data=market_frame,                               │  │
│ │     current_state=load_current_state()                      │  │
│ │ )                                                            │  │
│ │                                                              │  │
│ │ Inputs checked:                                             │  │
│ │   - rv21_level: 0.15 (15% SPY volatility)                   │  │
│ │   - market_breadth_ad: 0.24 (bullish)                       │  │
│ │   - market_breadth_bpi_21d: 0.67 (strong)                   │  │
│ │   - max_drawdown_5d: -0.03 (-3%, normal)                    │  │
│ │                                                              │  │
│ │ Result: posture = "NORMAL" (100% allocation)                │  │
│ │                                                              │  │
│ │ State transition logged:                                    │  │
│ │   Previous: NORMAL                                          │  │
│ │   Current:  NORMAL                                          │  │
│ │   (No change)                                               │  │
│ │                                                              │  │
│ │ Duration: <1 second                                         │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 6:05:24 PM - Portfolio optimization (HRP allocation)               │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ from algaie.portfolio import build_target_portfolio         │  │
│ │                                                              │  │
│ │ # Top-K selection based on scores                           │  │
│ │ K = config.portfolio.n_positions  # e.g., 30                │  │
│ │ threshold_percentile = (450 - K) / 450 = 93.3%              │  │
│ │                                                              │  │
│ │ selected = scores.filter(                                   │  │
│ │     pl.col("percentile") >= 0.933                           │  │
│ │ ).sort("rank")                                              │  │
│ │                                                              │  │
│ │ Selected symbols (top 30):                                  │  │
│ │   NVDA, AAPL, MSFT, GOOGL, AMZN, META, ...                  │  │
│ │                                                              │  │
│ │ # Compute 60-day covariance matrix                          │  │
│ │ returns_60d = get_returns_matrix(                           │  │
│ │     selected.symbols,                                       │  │
│ │     end_date=asof_date,                                     │  │
│ │     window=60                                               │  │
│ │ )                                                            │  │
│ │ cov_matrix = returns_60d.cov()                              │  │
│ │                                                              │  │
│ │ # Hierarchical Risk Parity allocation                       │  │
│ │ from algaie.portfolio import hrp_allocation                 │  │
│ │                                                              │  │
│ │ weights = hrp_allocation(cov_matrix)                        │  │
│ │                                                              │  │
│ │ # Apply risk posture adjustment                             │  │
│ │ if posture == "NORMAL":                                     │  │
│ │     allocation_factor = 1.0                                 │  │
│ │ elif posture == "CAUTIOUS":                                 │  │
│ │     allocation_factor = 0.7                                 │  │
│ │ elif posture == "DEFENSIVE":                                │  │
│ │     allocation_factor = 0.4                                 │  │
│ │                                                              │  │
│ │ weights = weights * allocation_factor                       │  │
│ │                                                              │  │
│ │ # Position sizing (account value: $100,000)                 │  │
│ │ account_value = get_account_value()                         │  │
│ │ prices = get_latest_prices(selected.symbols)                │  │
│ │                                                              │  │
│ │ positions = {                                               │  │
│ │     symbol: int(weight * account_value / price)             │  │
│ │     for symbol, weight, price                               │  │
│ │     in zip(selected.symbols, weights, prices)               │  │
│ │ }                                                            │  │
│ │                                                              │  │
│ │ Example positions:                                          │  │
│ │   NVDA: 9 shares × $500.20 = $4,502                         │  │
│ │   AAPL: 22 shares × $185.60 = $4,083                        │  │
│ │   MSFT: 12 shares × $380.40 = $4,565                        │  │
│ │   ...                                                        │  │
│ │   CASH: $5,850 (remaining + risk buffer)                    │  │
│ │                                                              │  │
│ │ Write: artifacts/portfolio/target_20240115.json             │  │
│ │                                                              │  │
│ │ Duration: ~3 seconds (HRP clustering)                       │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

PHASE 6: ORDER GENERATION & QUEUEING
═══════════════════════════════════════════════════════════════════════
┌───────────────────────────────────────────────────────────────────┐
│ 6:05:27 PM - Generate rebalancing orders                          │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ from algaie.execution import generate_orders                │  │
│ │                                                              │  │
│ │ # Load current portfolio (from yesterday)                   │  │
│ │ current = load_current_portfolio()                          │  │
│ │                                                              │  │
│ │ # Compute delta (target - current)                          │  │
│ │ orders = []                                                 │  │
│ │ for symbol in set(target.symbols + current.symbols):        │  │
│ │     target_qty = target.get(symbol, 0)                      │  │
│ │     current_qty = current.get(symbol, 0)                    │  │
│ │     delta = target_qty - current_qty                        │  │
│ │                                                              │  │
│ │     if delta > 0:                                           │  │
│ │         orders.append(Order(                                │  │
│ │             symbol=symbol,                                  │  │
│ │             side="BUY",                                     │  │
│ │             qty=delta,                                      │  │
│ │             type="MKT"                                      │  │
│ │         ))                                                   │  │
│ │     elif delta < 0:                                         │  │
│ │         orders.append(Order(                                │  │
│ │             symbol=symbol,                                  │  │
│ │             side="SELL",                                    │  │
│ │             qty=abs(delta),                                 │  │
│ │             type="MKT"                                      │  │
│ │         ))                                                   │  │
│ │                                                              │  │
│ │ Example orders:                                             │  │
│ │   BUY  NVDA  9 shares  @ MKT  (new position)                │  │
│ │   BUY  AAPL  2 shares  @ MKT  (increase from 20→22)         │  │
│ │   SELL GOOGL 5 shares  @ MKT  (exit position)               │  │
│ │   ...                                                        │  │
│ │                                                              │  │
│ │ Total: 45 orders (15 buys, 30 sells)                        │  │
│ │ Estimated market value: $94,150                             │  │
│ │ Turnover: 28% of portfolio value                            │  │
│ │                                                              │  │
│ │ Write: artifacts/orders/pending_20240116.json               │  │
│ │                                                              │  │
│ │ Duration: <1 second                                         │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 6:05:28 PM - Prepare execution report for human review            │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ from algaie.execution import generate_execution_report      │  │
│ │                                                              │  │
│ │ report = generate_execution_report(                         │  │
│ │     orders=orders,                                          │  │
│ │     target_portfolio=target,                                │  │
│ │     current_portfolio=current,                              │  │
│ │     scores=scores,                                          │  │
│ │     risk_posture=posture                                    │  │
│ │ )                                                            │  │
│ │                                                              │  │
│ │ Report contents:                                            │  │
│ │ ┌───────────────────────────────────────────────────────┐  │  │
│ │ │ ALGAIE NIGHTLY EXECUTION REPORT                       │  │  │
│ │ │ Date: 2024-01-15                                      │  │  │
│ │ │ ═══════════════════════════════════════════════════   │  │  │
│ │ │                                                        │  │  │
│ │ │ RISK POSTURE: NORMAL (100% allocation)                │  │  │
│ │ │                                                        │  │  │
│ │ │ UNIVERSE SUMMARY:                                     │  │  │
│ │ │   - Tradable: 450 symbols                             │  │  │
│ │ │   - Tier 0: 150, Tier 1: 153, Tier 2: 147            │  │  │
│ │ │                                                        │  │  │
│ │ │ TOP 10 SCORES:                                        │  │  │
│ │ │   1. NVDA   0.92 (rank 1,   99.8%ile)                 │  │  │
│ │ │   2. AAPL   0.89 (rank 2,   99.6%ile)                 │  │  │
│ │ │   3. MSFT   0.87 (rank 3,   99.3%ile)                 │  │  │
│ │ │   ...                                                  │  │  │
│ │ │   10. AMD   0.72 (rank 10,  97.8%ile)                 │  │  │
│ │ │                                                        │  │  │
│ │ │ PORTFOLIO CHANGES:                                    │  │  │
│ │ │   - New entries: 5 symbols (NVDA, PLTR, ...)          │  │  │
│ │ │   - Exits: 5 symbols (GOOGL, TSLA, ...)               │  │  │
│ │ │   - Increased: 12 positions                           │  │  │
│ │ │   - Decreased: 8 positions                            │  │  │
│ │ │   - Unchanged: 5 positions                            │  │  │
│ │ │                                                        │  │  │
│ │ │ ORDER SUMMARY:                                        │  │  │
│ │ │   - Total orders: 45 (15 buys, 30 sells)              │  │  │
│ │ │   - Estimated value: $94,150                          │  │  │
│ │ │   - Turnover: 28%                                     │  │  │
│ │ │                                                        │  │  │
│ │ │ APPROVAL STATUS: PENDING                              │  │  │
│ │ │ Review deadline: 2024-01-16 09:25 AM ET               │  │  │
│ │ │                                                        │  │  │
│ │ │ To approve: touch artifacts/orders/APPROVED           │  │  │
│ │ │ To reject:  touch artifacts/orders/REJECTED           │  │  │
│ │ └───────────────────────────────────────────────────────┘  │  │
│ │                                                              │  │
│ │ Write: artifacts/reports/execution_report_20240115.html     │  │
│ │                                                              │  │
│ │ Email notification sent to: trader@example.com              │  │
│ │ Slack notification posted to: #algaie-trading               │  │
│ │                                                              │  │
│ │ Duration: <1 second                                         │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

PHASE 7: LOGGING & CLEANUP
═══════════════════════════════════════════════════════════════════════
┌───────────────────────────────────────────────────────────────────┐
│ 6:05:29 PM - Finalize nightly cycle                               │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ # Update artifact registry                                  │  │
│ │ register_artifacts([                                        │  │
│ │     ("canonical", canonical_paths),                         │  │
│ │     ("universe", universe_path),                            │  │
│ │     ("features", features_path),                            │  │
│ │     ("priors", priors_path),                                │  │
│ │     ("scores", scores_path),                                │  │
│ │     ("portfolio", portfolio_path)                           │  │
│ │ ])                                                           │  │
│ │                                                              │  │
│ │ # Log provenance metadata                                   │  │
│ │ provenance = {                                              │  │
│ │     "date": "2024-01-15",                                   │  │
│ │     "feature_version": "v1_abc123",                         │  │
│ │     "prior_version": "v2_def456",                           │  │
│ │     "model_version": "selector_v1_fold3_ghi789",            │  │
│ │     "git_commit": "a1b2c3d",                                │  │
│ │     "elapsed_time_sec": 329                                 │  │
│ │ }                                                            │  │
│ │ write_provenance(provenance)                                │  │
│ │                                                              │  │
│ │ # Clean up temp files                                       │  │
│ │ remove_temp_files()                                         │  │
│ │                                                              │  │
│ │ # Release PID lock                                          │  │
│ │ release_lock()                                              │  │
│ │                                                              │  │
│ │ log.info("Nightly cycle completed successfully")            │  │
│ │ log.info(f"Total duration: 3m 29s")                         │  │
│ │ log.info("Orders queued for review")                        │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════
SUMMARY:
  Start:  6:00 PM ET
  End:    6:05 PM ET (wait for morning execution)
  Duration: 3 minutes 29 seconds

  Artifacts created:
    ✓ Canonical OHLCV (497 symbols)
    ✓ Universe (450 tradable)
    ✓ Features (450 × 28 columns)
    ✓ Priors (450 × 4 columns)
    ✓ Scores (450 symbols ranked)
    ✓ Target portfolio (30 positions)
    ✓ Orders (45 pending)
    ✓ Execution report (HTML + email)

  Next step: Human review + approval (9:25 AM deadline)
  Then: Pre-market execution (9:30 AM ET)
═══════════════════════════════════════════════════════════════════════
```

---

### 1.2 Morning Execution Workflow (9:30 AM Market Open)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MORNING EXECUTION WORKFLOW                            │
│                   (Pre-market order submission)                          │
└─────────────────────────────────────────────────────────────────────────┘

TIME: 9:25 AM ET (5 minutes before market open)
TRIGGER: Cron job OR manual approval signal
PRECONDITION: Orders approved OR auto-approve enabled

PHASE 1: PRE-FLIGHT CHECKS
═══════════════════════════════════════════════════════════════════════
┌───────────────────────────────────────────────────────────────────┐
│ 9:25:00 AM - Start execution workflow                             │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ $ python backend/scripts/run/run_execution.py \\            │  │
│ │     --orders artifacts/orders/pending_20240116.json \\      │  │
│ │     --mode live                                             │  │
│ │                                                              │  │
│ │ Modes:                                                       │  │
│ │   live: Submit to broker (IBKR, Alpaca, etc.)               │  │
│ │   paper: Paper trading simulation                           │  │
│ │   backtest: Historical simulation with fills                │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 9:25:01 AM - Approval check                                       │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ # Check for approval signal                                 │  │
│ │ approval_file = "artifacts/orders/APPROVED"                 │  │
│ │ rejection_file = "artifacts/orders/REJECTED"                │  │
│ │                                                              │  │
│ │ if exists(rejection_file):                                  │  │
│ │     log.warning("Orders REJECTED by human reviewer")        │  │
│ │     exit(0)  # No execution                                 │  │
│ │                                                              │  │
│ │ if not exists(approval_file):                               │  │
│ │     if config.execution.auto_approve:                       │  │
│ │         log.info("Auto-approve enabled, proceeding")        │  │
│ │     else:                                                    │  │
│ │         log.error("No approval found, aborting")            │  │
│ │         exit(1)                                             │  │
│ │                                                              │  │
│ │ log.info("Orders approved, proceeding with execution")      │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 9:25:02 AM - Broker connection & account verification             │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ from algaie.execution.broker import IBKRBroker              │  │
│ │                                                              │  │
│ │ # Connect to broker                                         │  │
│ │ broker = IBKRBroker(                                        │  │
│ │     host=config.broker.host,                                │  │
│ │     port=config.broker.port,                                │  │
│ │     client_id=config.broker.client_id                       │  │
│ │ )                                                            │  │
│ │ broker.connect()                                            │  │
│ │                                                              │  │
│ │ # Verify account                                            │  │
│ │ account_info = broker.get_account_info()                    │  │
│ │ log.info(f"Account: {account_info.account_id}")             │  │
│ │ log.info(f"Buying power: ${account_info.buying_power:,.2f}")│  │
│ │                                                              │  │
│ │ # Sanity checks                                             │  │
│ │ assert account_info.buying_power > total_order_value        │  │
│ │ assert account_info.status == "Active"                      │  │
│ │ assert not account_info.day_trading_restricted              │  │
│ │                                                              │  │
│ │ log.info("Broker connection established")                   │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 9:25:05 AM - Validate current positions vs expected               │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ # Fetch current positions from broker                       │  │
│ │ broker_positions = broker.get_positions()                   │  │
│ │                                                              │  │
│ │ # Load expected positions (from yesterday's target)         │  │
│ │ expected_positions = load_portfolio(                        │  │
│ │     "artifacts/portfolio/target_20240115.json"              │  │
│ │ )                                                            │  │
│ │                                                              │  │
│ │ # Reconciliation check                                      │  │
│ │ discrepancies = []                                          │  │
│ │ for symbol in set(broker_positions.keys() |                 │  │
│ │                   expected_positions.keys()):               │  │
│ │     broker_qty = broker_positions.get(symbol, 0)            │  │
│ │     expected_qty = expected_positions.get(symbol, 0)        │  │
│ │                                                              │  │
│ │     if broker_qty != expected_qty:                          │  │
│ │         discrepancies.append(                               │  │
│ │             f"{symbol}: broker={broker_qty}, "              │  │
│ │             f"expected={expected_qty}"                      │  │
│ │         )                                                    │  │
│ │                                                              │  │
│ │ if discrepancies:                                           │  │
│ │     log.warning(f"Position discrepancies: {discrepancies}") │  │
│ │     # Alert but continue (manual trades may have occurred)  │  │
│ │                                                              │  │
│ │ log.info("Position reconciliation complete")                │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

PHASE 2: ORDER EXECUTION
═══════════════════════════════════════════════════════════════════════
┌───────────────────────────────────────────────────────────────────┐
│ 9:25:10 AM - Submit orders to broker (phased execution)           │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ # Load pending orders                                       │  │
│ │ orders = load_orders("artifacts/orders/pending_20240116.json")│ │
│ │                                                              │  │
│ │ # Sort: SELL orders first (free up capital)                 │  │
│ │ sell_orders = [o for o in orders if o.side == "SELL"]       │  │
│ │ buy_orders = [o for o in orders if o.side == "BUY"]         │  │
│ │ sorted_orders = sell_orders + buy_orders                    │  │
│ │                                                              │  │
│ │ # Submit orders with throttling                             │  │
│ │ submitted = []                                              │  │
│ │ failed = []                                                 │  │
│ │                                                              │  │
│ │ for order in sorted_orders:                                 │  │
│ │     try:                                                     │  │
│ │         # Submit market order                               │  │
│ │         broker_order = broker.submit_order(                 │  │
│ │             symbol=order.symbol,                            │  │
│ │             side=order.side,                                │  │
│ │             qty=order.qty,                                  │  │
│ │             type="MKT",                                     │  │
│ │             time_in_force="DAY"                             │  │
│ │         )                                                    │  │
│ │                                                              │  │
│ │         submitted.append({                                  │  │
│ │             "order": order,                                 │  │
│ │             "broker_order_id": broker_order.order_id,       │  │
│ │             "timestamp": datetime.now()                     │  │
│ │         })                                                   │  │
│ │                                                              │  │
│ │         log.info(                                           │  │
│ │             f"Submitted: {order.side} {order.qty} "         │  │
│ │             f"{order.symbol} (ID: {broker_order.order_id})" │  │
│ │         )                                                    │  │
│ │                                                              │  │
│ │         # Throttle to avoid overwhelming broker             │  │
│ │         time.sleep(0.1)  # 100ms between orders             │  │
│ │                                                              │  │
│ │     except BrokerError as e:                                │  │
│ │         log.error(f"Failed to submit {order}: {e}")         │  │
│ │         failed.append({"order": order, "error": str(e)})    │  │
│ │                                                              │  │
│ │ Summary:                                                     │  │
│ │   Total orders: 45                                          │  │
│ │   Submitted: 44 (97.8%)                                     │  │
│ │   Failed: 1 (2.2%)                                          │  │
│ │     - NVDA: "Insufficient buying power" (order too large)   │  │
│ │                                                              │  │
│ │ Duration: ~5 seconds (45 orders × 100ms throttle)           │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 9:25:15 AM - Monitor order fills (wait up to 5 minutes)           │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ # Poll broker for order status                              │  │
│ │ start_time = time.time()                                    │  │
│ │ timeout = 300  # 5 minutes                                  │  │
│ │                                                              │  │
│ │ fills = []                                                  │  │
│ │ pending = submitted.copy()                                  │  │
│ │                                                              │  │
│ │ while pending and (time.time() - start_time) < timeout:     │  │
│ │     for order_info in pending[:]:  # Iterate over copy      │  │
│ │         status = broker.get_order_status(                   │  │
│ │             order_info["broker_order_id"]                   │  │
│ │         )                                                    │  │
│ │                                                              │  │
│ │         if status.state == "FILLED":                        │  │
│ │             fills.append({                                  │  │
│ │                 "order": order_info["order"],               │  │
│ │                 "fill_price": status.avg_fill_price,        │  │
│ │                 "fill_qty": status.filled_qty,              │  │
│ │                 "fill_time": status.fill_time               │  │
│ │             })                                              │  │
│ │             pending.remove(order_info)                      │  │
│ │             log.info(                                       │  │
│ │                 f"FILLED: {order_info['order'].symbol} "    │  │
│ │                 f"@ ${status.avg_fill_price:.2f}"           │  │
│ │             )                                               │  │
│ │                                                              │  │
│ │         elif status.state == "REJECTED":                    │  │
│ │             log.error(                                      │  │
│ │                 f"REJECTED: {order_info['order'].symbol} "  │  │
│ │                 f"- {status.reject_reason}"                 │  │
│ │             )                                               │  │
│ │             pending.remove(order_info)                      │  │
│ │                                                              │  │
│ │     # Wait 1 second before next poll                        │  │
│ │     if pending:                                             │  │
│ │         time.sleep(1)                                       │  │
│ │                                                              │  │
│ │ # Final status                                              │  │
│ │ fill_rate = len(fills) / len(submitted)                     │  │
│ │ log.info(f"Fill rate: {fill_rate:.1%} ({len(fills)}/{len(submitted)})")│
│ │                                                              │  │
│ │ if pending:                                                 │  │
│ │     log.warning(f"{len(pending)} orders still pending after timeout")│ │
│ │     # Cancel unfilled orders                                │  │
│ │     for order_info in pending:                              │  │
│ │         broker.cancel_order(order_info["broker_order_id"])  │  │
│ │                                                              │  │
│ │ Duration: ~15 seconds (most orders fill quickly at open)    │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

PHASE 3: RECONCILIATION & LOGGING
═══════════════════════════════════════════════════════════════════════
┌───────────────────────────────────────────────────────────────────┐
│ 9:25:30 AM - Update portfolio state                               │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ # Fetch final positions from broker                         │  │
│ │ final_positions = broker.get_positions()                    │  │
│ │                                                              │  │
│ │ # Compute portfolio metrics                                 │  │
│ │ portfolio_value = sum(                                      │  │
│ │     pos.qty * pos.market_price                              │  │
│ │     for pos in final_positions.values()                     │  │
│ │ )                                                            │  │
│ │ cash = broker.get_account_info().cash                       │  │
│ │ total_equity = portfolio_value + cash                       │  │
│ │                                                              │  │
│ │ log.info(f"Portfolio value: ${portfolio_value:,.2f}")       │  │
│ │ log.info(f"Cash: ${cash:,.2f}")                             │  │
│ │ log.info(f"Total equity: ${total_equity:,.2f}")             │  │
│ │                                                              │  │
│ │ # Write current state                                       │  │
│ │ write_portfolio_state(                                      │  │
│ │     positions=final_positions,                              │  │
│ │     cash=cash,                                              │  │
│ │     timestamp=datetime.now(),                               │  │
│ │     path="artifacts/portfolio/current_state.json"           │  │
│ │ )                                                            │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 9:25:31 AM - Log trade history & generate report                  │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ # Write fills to trade log (append-only)                    │  │
│ │ trade_log = load_trade_log()                                │  │
│ │ for fill in fills:                                          │  │
│ │     trade_log.append({                                      │  │
│ │         "date": "2024-01-16",                               │  │
│ │         "symbol": fill["order"].symbol,                     │  │
│ │         "side": fill["order"].side,                         │  │
│ │         "qty": fill["fill_qty"],                            │  │
│ │         "price": fill["fill_price"],                        │  │
│ │         "value": fill["fill_qty"] * fill["fill_price"],     │  │
│ │         "timestamp": fill["fill_time"]                      │  │
│ │     })                                                       │  │
│ │                                                              │  │
│ │ write_trade_log(trade_log)                                  │  │
│ │                                                              │  │
│ │ # Generate execution report                                 │  │
│ │ report = {                                                  │  │
│ │     "date": "2024-01-16",                                   │  │
│ │     "orders_submitted": len(submitted),                     │  │
│ │     "orders_filled": len(fills),                            │  │
│ │     "orders_failed": len(failed),                           │  │
│ │     "fill_rate": len(fills) / len(submitted),               │  │
│ │     "total_value_traded": sum(                              │  │
│ │         f["fill_qty"] * f["fill_price"] for f in fills      │  │
│ │     ),                                                       │  │
│ │     "portfolio_value": portfolio_value,                     │  │
│ │     "cash": cash,                                           │  │
│ │     "total_equity": total_equity                            │  │
│ │ }                                                            │  │
│ │                                                              │  │
│ │ write_execution_report(                                     │  │
│ │     report,                                                 │  │
│ │     path="artifacts/reports/execution_20240116.json"        │  │
│ │ )                                                            │  │
│ │                                                              │  │
│ │ # Email + Slack notification                                │  │
│ │ notify(                                                     │  │
│ │     subject="Execution Complete: 44/45 orders filled",      │  │
│ │     body=f"Portfolio value: ${portfolio_value:,.2f}"        │  │
│ │ )                                                            │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 9:25:32 AM - Cleanup & disconnect                                 │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ # Archive executed orders                                   │  │
│ │ archive_orders(                                             │  │
│ │     "artifacts/orders/pending_20240116.json",               │  │
│ │     "artifacts/orders/archive/executed_20240116.json"       │  │
│ │ )                                                            │  │
│ │                                                              │  │
│ │ # Disconnect from broker                                    │  │
│ │ broker.disconnect()                                         │  │
│ │                                                              │  │
│ │ log.info("Execution workflow completed")                    │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════
SUMMARY:
  Start:  9:25 AM ET
  End:    9:25:32 AM ET
  Duration: 32 seconds

  Orders: 45 submitted, 44 filled, 1 failed
  Fill rate: 97.8%
  Value traded: $94,150

  New portfolio value: $101,230
  Cash: $5,850
  Total equity: $107,080

  Status: SUCCESS (within normal parameters)
═══════════════════════════════════════════════════════════════════════
```

---

## 2. WEEKLY RETRAINING WORKFLOW

### 2.1 Complete Weekly Training Cycle (Sunday Evening)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    WEEKLY RETRAINING WORKFLOW                            │
│              (Update models with latest week of data)                    │
└─────────────────────────────────────────────────────────────────────────┘

TIME: Sunday 6:00 PM ET (weekly)
TRIGGER: Cron job (0 18 * * 0)  # Every Sunday 6 PM
DURATION: ~90 minutes total

PHASE 1: DATA PREPARATION
═══════════════════════════════════════════════════════════════════════
┌───────────────────────────────────────────────────────────────────┐
│ 6:00:00 PM - Start weekly training cycle                          │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ $ python backend/scripts/run/run_weekly_training.py \\      │  │
│ │     --end-date $(date +%Y-%m-%d) \\                         │  │
│ │     --train-window 504 \\  # 2 years                        │  │
│ │     --mode production                                        │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 6:00:05 PM - Validate data completeness                           │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ # Check canonical data coverage                             │  │
│ │ end_date = "2024-01-14"  # Last Friday                      │  │
│ │ start_date = trading_days_ago(end_date, 504)  # 2 years    │  │
│ │                                                              │  │
│ │ expected_symbols = load_universe_symbols()  # 500           │  │
│ │ expected_dates = get_trading_days(start_date, end_date)     │  │
│ │ # ~504 trading days                                         │  │
│ │                                                              │  │
│ │ # Validate coverage                                         │  │
│ │ for symbol in expected_symbols:                             │  │
│ │     canonical_path = f"artifacts/canonical/{symbol}.parquet"│  │
│ │     if not exists(canonical_path):                          │  │
│ │         log.warning(f"{symbol}: No canonical data")         │  │
│ │         continue                                            │  │
│ │                                                              │  │
│ │     df = pl.read_parquet(canonical_path)                    │  │
│ │     actual_dates = set(df["date"])                          │  │
│ │     missing_dates = set(expected_dates) - actual_dates      │  │
│ │                                                              │  │
│ │     if len(missing_dates) > 10:  # >2% missing              │  │
│ │         log.error(                                          │  │
│ │             f"{symbol}: {len(missing_dates)} missing dates" │  │
│ │         )                                                    │  │
│ │         # Exclude from training                             │  │
│ │                                                              │  │
│ │ Coverage summary:                                           │  │
│ │   495/500 symbols have complete data (99%)                  │  │
│ │   5 symbols excluded (insufficient coverage)                │  │
│ │                                                              │  │
│ │ Duration: ~30 seconds                                       │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 6:00:35 PM - Build training dataset                               │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ # Rebuild full pipeline for training window                 │  │
│ │ from algaie.data.features import build_training_dataset     │  │
│ │                                                              │  │
│ │ dataset = build_training_dataset(                           │  │
│ │     canonical_dir="artifacts/canonical/",                   │  │
│ │     start_date=start_date,                                  │  │
│ │     end_date=end_date,                                      │  │
│ │     symbols=valid_symbols,  # 495 symbols                   │  │
│ │     config=config                                           │  │
│ │ )                                                            │  │
│ │                                                              │  │
│ │ Pipeline steps:                                             │  │
│ │   1. Universe construction (daily for 504 days)             │  │
│ │   2. Market enrichment (breadth + covariates)               │  │
│ │   3. MarketFrame assembly                                   │  │
│ │   4. Feature engineering (raw + rank normalization)         │  │
│ │   5. Forward return targets (5-day horizon)                 │  │
│ │   6. Breadth filtering (remove weak breadth days)           │  │
│ │   7. Drop NaN rows (warmup period)                          │  │
│ │                                                              │  │
│ │ Output shape:                                               │  │
│ │   ~180,000 rows (450 symbols × 400 days after filtering)    │  │
│ │   32 columns (28 features + 4 targets)                      │  │
│ │                                                              │  │
│ │ Write: artifacts/training/dataset_20240114.parquet          │  │
│ │                                                              │  │
│ │ Duration: ~8 minutes (full historical rebuild)              │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

PHASE 2: CHRONOS TEACHER TRAINING (OPTIONAL - Monthly)
═══════════════════════════════════════════════════════════════════════
┌───────────────────────────────────────────────────────────────────┐
│ 6:08:35 PM - Check if Chronos retraining needed                   │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ # Chronos trained less frequently (monthly/quarterly)       │  │
│ │ last_training = get_last_chronos_training_date()            │  │
│ │ days_since = (end_date - last_training).days                │  │
│ │                                                              │  │
│ │ if days_since < config.chronos.retrain_interval_days:       │  │
│ │     log.info(                                               │  │
│ │         f"Chronos last trained {days_since} days ago, "     │  │
│ │         f"skipping (threshold: {config.chronos.retrain_interval_days})"│
│ │     )                                                        │  │
│ │     # Skip to selector training                             │  │
│ │ else:                                                        │  │
│ │     log.info("Chronos retraining triggered")                │  │
│ │     # Proceed with Chronos training below                   │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 6:08:36 PM - Build Chronos gold dataset (if retraining)           │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ from algaie.data.features import GoldDatasetBuilder         │  │
│ │                                                              │  │
│ │ builder = GoldDatasetBuilder(config.chronos)                │  │
│ │ gold_dataset = builder.build(                               │  │
│ │     canonical_dir="artifacts/canonical/",                   │  │
│ │     symbols=valid_symbols,                                  │  │
│ │     end_date=end_date                                       │  │
│ │ )                                                            │  │
│ │                                                              │  │
│ │ Dataset structure:                                          │  │
│ │   - Per-symbol time-series windows                          │  │
│ │   - Context: 512 trading days                               │  │
│ │   - Horizons: 1-21 trading days forward                     │  │
│ │   - Rolling windows (stride=21 days)                        │  │
│ │                                                              │  │
│ │ Total samples: 495 symbols × 23 windows = 11,385            │  │
│ │                                                              │  │
│ │ Write: artifacts/training/chronos_gold_20240114.parquet     │  │
│ │                                                              │  │
│ │ Duration: ~3 minutes                                        │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 6:11:36 PM - Train Chronos teacher (GPU required)                 │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ from algaie.models.foundation import train_chronos_teacher  │  │
│ │                                                              │  │
│ │ # Load base T5 model + LoRA config                          │  │
│ │ base_model = "amazon/chronos-t5-base"                       │  │
│ │ lora_config = LoRAConfig(                                   │  │
│ │     r=8,  # LoRA rank                                       │  │
│ │     lora_alpha=16,                                          │  │
│ │     target_modules=["q", "v"],  # Attention layers          │  │
│ │     lora_dropout=0.05                                       │  │
│ │ )                                                            │  │
│ │                                                              │  │
│ │ # Training configuration                                    │  │
│ │ training_args = {                                           │  │
│ │     "batch_size": 32,                                       │  │
│ │     "gradient_accumulation": 4,  # Effective batch=128      │  │
│ │     "learning_rate": 1e-4,                                  │  │
│ │     "weight_decay": 0.01,                                   │  │
│ │     "epochs": 10,                                           │  │
│ │     "warmup_steps": 100,                                    │  │
│ │     "fp16": True  # Mixed precision                         │  │
│ │ }                                                            │  │
│ │                                                              │  │
│ │ # Train                                                      │  │
│ │ trainer = train_chronos_teacher(                            │  │
│ │     dataset=gold_dataset,                                   │  │
│ │     base_model=base_model,                                  │  │
│ │     lora_config=lora_config,                                │  │
│ │     training_args=training_args,                            │  │
│ │     output_dir="artifacts/models/chronos_teacher_v2_lora/"  │  │
│ │ )                                                            │  │
│ │                                                              │  │
│ │ Training progress:                                          │  │
│ │   Epoch 1/10: loss=0.452, val_nll=0.521                     │  │
│ │   Epoch 2/10: loss=0.398, val_nll=0.487                     │  │
│ │   ...                                                        │  │
│ │   Epoch 10/10: loss=0.312, val_nll=0.405                    │  │
│ │                                                              │  │
│ │ Best checkpoint: epoch 9 (val_nll=0.403)                    │  │
│ │                                                              │  │
│ │ Duration: ~10 minutes (A100 GPU)                            │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

PHASE 3: SELECTOR TRAINING (ALWAYS)
═══════════════════════════════════════════════════════════════════════
┌───────────────────────────────────────────────────────────────────┐
│ 6:21:36 PM - Generate Chronos priors for training dates           │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ # Use latest Chronos checkpoint (newly trained or existing) │  │
│ │ chronos = load_chronos_teacher("artifacts/models/chronos_teacher_v2_lora/")│
│ │                                                              │  │
│ │ # Generate priors for all training dates                    │  │
│ │ priors_all = []                                             │  │
│ │ for date in training_dates:  # 400 dates                    │  │
│ │     contexts = prepare_contexts(                            │  │
│ │         canonical_dir,                                      │  │
│ │         symbols=valid_symbols,                              │  │
│ │         end_date=date,                                      │  │
│ │         context_length=512                                  │  │
│ │     )                                                        │  │
│ │                                                              │  │
│ │     with torch.no_grad():                                   │  │
│ │         distributions = chronos.predict_batch(contexts)     │  │
│ │         date_priors = extract_priors(distributions)         │  │
│ │         priors_all.extend(date_priors)                      │  │
│ │                                                              │  │
│ │ priors_df = pl.DataFrame(priors_all)                        │  │
│ │ # Shape: 180,000 rows × 5 cols (symbol, date, drift, vol, tail, prob)│
│ │                                                              │  │
│ │ Write: artifacts/training/chronos_priors_training.parquet   │  │
│ │                                                              │  │
│ │ Duration: ~5 minutes (400 dates × GPU inference)            │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 6:26:36 PM - Join features + priors for selector training         │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ # Combine feature dataset with Chronos priors               │  │
│ │ training_data = dataset.join(                               │  │
│ │     priors_df,                                              │  │
│ │     on=["symbol", "date"],                                  │  │
│ │     how="inner"                                             │  │
│ │ )                                                            │  │
│ │                                                              │  │
│ │ # Validate join (no rows lost)                              │  │
│ │ assert len(training_data) == len(dataset)                   │  │
│ │                                                              │  │
│ │ # Final shape: 180,000 rows × 32 columns                    │  │
│ │   - 28 feature columns                                      │  │
│ │   - 4 prior columns (from Chronos)                          │  │
│ │   - y_rank, y_trade (targets)                               │  │
│ │                                                              │  │
│ │ Duration: ~10 seconds (indexed join)                        │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 6:26:46 PM - Walk-forward cross-validation splits                 │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ from algaie.models.ranker import create_walk_forward_splits │  │
│ │                                                              │  │
│ │ splits = create_walk_forward_splits(                        │  │
│ │     data=training_data,                                     │  │
│ │     train_window_days=504,  # 2 years                       │  │
│ │     test_window_days=126,   # 6 months                      │  │
│ │     embargo_days=6,         # H_sel + 1                     │  │
│ │     n_splits=5              # 5 folds                       │  │
│ │ )                                                            │  │
│ │                                                              │  │
│ │ Split structure:                                            │  │
│ │ ┌────────────────────────────────────────────────────┐     │  │
│ │ │ Fold 1:                                            │     │  │
│ │ │   Train: 2020-01-01 → 2021-12-31 [EMBARGO 6d]     │     │  │
│ │ │   Test:  2022-01-10 → 2022-06-30                   │     │  │
│ │ │                                                     │     │  │
│ │ │ Fold 2:                                            │     │  │
│ │ │   Train: 2020-07-01 → 2022-06-30 [EMBARGO 6d]     │     │  │
│ │ │   Test:  2022-07-10 → 2022-12-31                   │     │  │
│ │ │                                                     │     │  │
│ │ │ Fold 3:                                            │     │  │
│ │ │   Train: 2021-01-01 → 2022-12-31 [EMBARGO 6d]     │     │  │
│ │ │   Test:  2023-01-10 → 2023-06-30                   │     │  │
│ │ │                                                     │     │  │
│ │ │ Fold 4:                                            │     │  │
│ │ │   Train: 2021-07-01 → 2023-06-30 [EMBARGO 6d]     │     │  │
│ │ │   Test:  2023-07-10 → 2023-12-31                   │     │  │
│ │ │                                                     │     │  │
│ │ │ Fold 5:                                            │     │  │
│ │ │   Train: 2022-01-01 → 2023-12-31 [EMBARGO 6d]     │     │  │
│ │ │   Test:  2024-01-10 → 2024-01-14  (latest)         │     │  │
│ │ └────────────────────────────────────────────────────┘     │  │
│ │                                                              │  │
│ │ Duration: ~5 seconds (date filtering)                       │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 6:26:51 PM - Train selector for each fold (sequential)            │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ from algaie.models.ranker import train_rank_transformer     │  │
│ │                                                              │  │
│ │ for fold_idx, (train_data, test_data) in enumerate(splits): │  │
│ │     log.info(f"Training fold {fold_idx+1}/5")               │  │
│ │                                                              │  │
│ │     # Fit preprocessor on training fold only                │  │
│ │     scaler = StandardScaler()                               │  │
│ │     X_train = train_data[feature_cols]                      │  │
│ │     scaler.fit(X_train)                                     │  │
│ │     X_train_scaled = scaler.transform(X_train)              │  │
│ │     X_test_scaled = scaler.transform(test_data[feature_cols])│ │
│ │                                                              │  │
│ │     # Training configuration                                │  │
│ │     model = RankTransformer(                                │  │
│ │         input_dim=32,                                       │  │
│ │         hidden_dim=128,                                     │  │
│ │         num_layers=6,                                       │  │
│ │         num_heads=4                                         │  │
│ │     )                                                        │  │
│ │                                                              │  │
│ │     training_args = {                                       │  │
│ │         "batch_size": 256,  # Per-date batching             │  │
│ │         "learning_rate": 3e-4,                              │  │
│ │         "epochs": 20,                                       │  │
│ │         "early_stopping_patience": 5,                       │  │
│ │         "loss_weights": {                                   │  │
│ │             "quantile": 0.5,                                │  │
│ │             "direction": 0.3,                               │  │
│ │             "risk": 0.2                                     │  │
│ │         }                                                    │  │
│ │     }                                                        │  │
│ │                                                              │  │
│ │     # Train                                                  │  │
│ │     trainer = train_rank_transformer(                       │  │
│ │         model=model,                                        │  │
│ │         train_data=(X_train_scaled, y_train),               │  │
│ │         val_data=(X_test_scaled, y_test),                   │  │
│ │         training_args=training_args                         │  │
│ │     )                                                        │  │
│ │                                                              │  │
│ │     # Evaluate on test fold                                 │  │
│ │     metrics = trainer.evaluate(X_test_scaled, y_test)       │  │
│ │     log.info(f"Fold {fold_idx+1} metrics:")                 │  │
│ │     log.info(f"  IC: {metrics['ic']:.3f}")                  │  │
│ │     log.info(f"  Sharpe: {metrics['sharpe']:.2f}")          │  │
│ │     log.info(f"  Hit rate: {metrics['hit_rate']:.1%}")      │  │
│ │                                                              │  │
│ │     # Save checkpoint                                       │  │
│ │     save_checkpoint(                                        │  │
│ │         model=model,                                        │  │
│ │         scaler=scaler,                                      │  │
│ │         path=f"artifacts/models/selector_v2_fold{fold_idx+1}.pt"│ │
│ │     )                                                        │  │
│ │                                                              │  │
│ │ Training results per fold:                                  │  │
│ │   Fold 1: IC=0.042, Sharpe=1.85, Hit=54.2%                  │  │
│ │   Fold 2: IC=0.038, Sharpe=1.72, Hit=53.8%                  │  │
│ │   Fold 3: IC=0.045, Sharpe=1.92, Hit=54.8%                  │  │
│ │   Fold 4: IC=0.041, Sharpe=1.78, Hit=54.1%                  │  │
│ │   Fold 5: IC=0.047, Sharpe=2.01, Hit=55.3% ← BEST           │  │
│ │                                                              │  │
│ │ Average: IC=0.043, Sharpe=1.86, Hit=54.4%                   │  │
│ │                                                              │  │
│ │ Duration: ~15 min per fold × 5 folds = 75 minutes           │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

PHASE 4: MODEL PROMOTION
═══════════════════════════════════════════════════════════════════════
┌───────────────────────────────────────────────────────────────────┐
│ 7:41:51 PM - Promote best model to production                     │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ # Select best fold (highest Sharpe on most recent test)     │  │
│ │ best_fold = 5  # Sharpe=2.01, most recent data              │  │
│ │                                                              │  │
│ │ # Promote to production                                     │  │
│ │ from algaie.core.artifacts import promote_to_prod           │  │
│ │                                                              │  │
│ │ promote_to_prod(                                            │  │
│ │     artifact_path=f"artifacts/models/selector_v2_fold{best_fold}.pt",│ │
│ │     artifact_type="selector_model"                          │  │
│ │ )                                                            │  │
│ │                                                              │  │
│ │ # Deprecate old production model                            │  │
│ │ deprecate_artifact("artifacts/models/selector_v1_fold3.pt") │  │
│ │                                                              │  │
│ │ # Update symlink                                            │  │
│ │ symlink(                                                    │  │
│ │     target=f"selector_v2_fold{best_fold}.pt",               │  │
│ │     link="artifacts/models/selector_latest.pt"              │  │
│ │ )                                                            │  │
│ │                                                              │  │
│ │ log.info(f"Promoted fold {best_fold} to production")        │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 7:41:52 PM - Generate training report & alerts                    │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ # Create comprehensive training report                      │  │
│ │ report = {                                                  │  │
│ │     "training_date": "2024-01-14",                          │  │
│ │     "training_window": {                                    │  │
│ │         "start": "2022-01-01",                              │  │
│ │         "end": "2024-01-14",                                │  │
│ │         "n_days": 504                                       │  │
│ │     },                                                       │  │
│ │     "dataset": {                                            │  │
│ │         "n_symbols": 495,                                   │  │
│ │         "n_dates": 400,                                     │  │
│ │         "n_rows": 180000                                    │  │
│ │     },                                                       │  │
│ │     "chronos": {                                            │  │
│ │         "retrained": True,                                  │  │
│ │         "final_val_nll": 0.403,                             │  │
│ │         "training_time_min": 10                             │  │
│ │     },                                                       │  │
│ │     "selector": {                                           │  │
│ │         "n_folds": 5,                                       │  │
│ │         "avg_ic": 0.043,                                    │  │
│ │         "avg_sharpe": 1.86,                                 │  │
│ │         "avg_hit_rate": 0.544,                              │  │
│ │         "best_fold": 5,                                     │  │
│ │         "training_time_min": 75                             │  │
│ │     },                                                       │  │
│ │     "total_time_min": 101                                   │  │
│ │ }                                                            │  │
│ │                                                              │  │
│ │ write_training_report(                                      │  │
│ │     report,                                                 │  │
│ │     path="artifacts/reports/training_20240114.json"         │  │
│ │ )                                                            │  │
│ │                                                              │  │
│ │ # Email + Slack notification                                │  │
│ │ notify(                                                     │  │
│ │     subject="Weekly Training Complete",                     │  │
│ │     body=(                                                  │  │
│ │         f"New model deployed (Fold {best_fold})\\n"         │  │
│ │         f"Sharpe: {report['selector']['avg_sharpe']:.2f}\\n"│  │
│ │         f"IC: {report['selector']['avg_ic']:.3f}"           │  │
│ │     )                                                        │  │
│ │ )                                                            │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════
SUMMARY:
  Start:  6:00 PM ET Sunday
  End:    7:42 PM ET Sunday
  Duration: 101 minutes (1h 41m)

  Components trained:
    ✓ Chronos teacher (new checkpoint v2)
    ✓ Selector (5 folds, fold 5 promoted)

  Performance:
    Selector IC: 0.043 (avg), 0.047 (best fold)
    Selector Sharpe: 1.86 (avg), 2.01 (best fold)
    Hit rate: 54.4% (avg), 55.3% (best fold)

  Models promoted to production:
    chronos_teacher_v2_lora/
    selector_v2_fold5.pt

  Next: Nightly inference will use new models starting Monday
═══════════════════════════════════════════════════════════════════════
```

---

## 3. LIVE TRADING INTEGRATION

### 3.1 Broker Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BROKER INTEGRATION ARCHITECTURE                       │
│                  (Abstract interface + concrete implementations)         │
└─────────────────────────────────────────────────────────────────────────┘

ABSTRACTION LAYER:
┌───────────────────────────────────────────────────────────────────┐
│ class Broker (Abstract Base Class):                               │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ # Connection management                                     │  │
│ │ def connect(self) -> None:                                  │  │
│ │     """Establish connection to broker"""                    │  │
│ │                                                              │  │
│ │ def disconnect(self) -> None:                               │  │
│ │     """Close broker connection"""                           │  │
│ │                                                              │  │
│ │ def is_connected(self) -> bool:                             │  │
│ │     """Check connection status"""                           │  │
│ │                                                              │  │
│ │ # Account information                                       │  │
│ │ def get_account_info(self) -> AccountInfo:                  │  │
│ │     """Fetch account balance, buying power, etc."""         │  │
│ │                                                              │  │
│ │ def get_positions(self) -> Dict[str, Position]:             │  │
│ │     """Get current positions"""                             │  │
│ │                                                              │  │
│ │ # Order execution                                           │  │
│ │ def submit_order(                                           │  │
│ │     self,                                                    │  │
│ │     symbol: str,                                            │  │
│ │     side: Literal["BUY", "SELL"],                           │  │
│ │     qty: int,                                               │  │
│ │     type: Literal["MKT", "LMT", "STP"],                     │  │
│ │     limit_price: Optional[float] = None,                    │  │
│ │     stop_price: Optional[float] = None,                     │  │
│ │     time_in_force: str = "DAY"                              │  │
│ │ ) -> BrokerOrder:                                           │  │
│ │     """Submit order to broker"""                            │  │
│ │                                                              │  │
│ │ def cancel_order(self, order_id: str) -> None:              │  │
│ │     """Cancel pending order"""                              │  │
│ │                                                              │  │
│ │ def get_order_status(self, order_id: str) -> OrderStatus:   │  │
│ │     """Query order status"""                                │  │
│ │                                                              │  │
│ │ # Market data                                               │  │
│ │ def get_quote(self, symbol: str) -> Quote:                  │  │
│ │     """Get current bid/ask/last price"""                    │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

CONCRETE IMPLEMENTATIONS:

┌───────────────────────────────────────────────────────────────────┐
│ 1. INTERACTIVE BROKERS (IBKR) - Production                        │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ class IBKRBroker(Broker):                                   │  │
│ │     def __init__(                                           │  │
│ │         self,                                                │  │
│ │         host: str = "127.0.0.1",                            │  │
│ │         port: int = 7497,  # TWS live                       │  │
│ │         client_id: int = 1                                  │  │
│ │     ):                                                       │  │
│ │         self.ib = IB()                                      │  │
│ │         self.host = host                                    │  │
│ │         self.port = port                                    │  │
│ │         self.client_id = client_id                          │  │
│ │                                                              │  │
│ │     def connect(self):                                      │  │
│ │         self.ib.connect(                                    │  │
│ │             self.host,                                      │  │
│ │             self.port,                                      │  │
│ │             clientId=self.client_id                         │  │
│ │         )                                                    │  │
│ │         log.info("IBKR connected")                          │  │
│ │                                                              │  │
│ │     def submit_order(self, ...):                            │  │
│ │         contract = Stock(symbol, "SMART", "USD")            │  │
│ │         order = MarketOrder(side, qty)                      │  │
│ │         trade = self.ib.placeOrder(contract, order)         │  │
│ │         return BrokerOrder(                                 │  │
│ │             order_id=str(trade.order.orderId),              │  │
│ │             status=trade.orderStatus.status                 │  │
│ │         )                                                    │  │
│ │                                                              │  │
│ │ Ports:                                                      │  │
│ │   - 7497: TWS live trading                                  │  │
│ │   - 7496: TWS paper trading                                 │  │
│ │   - 4001: IB Gateway live                                   │  │
│ │   - 4002: IB Gateway paper                                  │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 2. ALPACA - Alternative Production                                │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ class AlpacaBroker(Broker):                                 │  │
│ │     def __init__(                                           │  │
│ │         self,                                                │  │
│ │         api_key: str,                                       │  │
│ │         api_secret: str,                                    │  │
│ │         base_url: str = "https://api.alpaca.markets"        │  │
│ │     ):                                                       │  │
│ │         self.api = tradeapi.REST(                           │  │
│ │             api_key,                                        │  │
│ │             api_secret,                                     │  │
│ │             base_url                                        │  │
│ │         )                                                    │  │
│ │                                                              │  │
│ │     def submit_order(self, ...):                            │  │
│ │         order = self.api.submit_order(                      │  │
│ │             symbol=symbol,                                  │  │
│ │             qty=qty,                                        │  │
│ │             side=side.lower(),                              │  │
│ │             type="market",                                  │  │
│ │             time_in_force="day"                             │  │
│ │         )                                                    │  │
│ │         return BrokerOrder(                                 │  │
│ │             order_id=order.id,                              │  │
│ │             status=order.status                             │  │
│ │         )                                                    │  │
│ │                                                              │  │
│ │ Endpoints:                                                  │  │
│ │   - Live: https://api.alpaca.markets                        │  │
│ │   - Paper: https://paper-api.alpaca.markets                 │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│ 3. BACKTEST BROKER - Simulation (No real execution)               │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ class BacktestBroker(Broker):                               │  │
│ │     """Simulates fills for backtesting"""                   │  │
│ │                                                              │  │
│ │     def __init__(                                           │  │
│ │         self,                                                │  │
│ │         historical_data: pl.DataFrame,                      │  │
│ │         slippage_bps: float = 5.0,                          │  │
│ │         commission_per_share: float = 0.005                 │  │
│ │     ):                                                       │  │
│ │         self.data = historical_data                         │  │
│ │         self.slippage_bps = slippage_bps                    │  │
│ │         self.commission = commission_per_share              │  │
│ │         self.positions = {}                                 │  │
│ │         self.cash = 100_000.0  # Starting capital           │  │
│ │                                                              │  │
│ │     def submit_order(self, ...):                            │  │
│ │         # Simulate fill at next open + slippage             │  │
│ │         fill_price = self._get_fill_price(symbol)           │  │
│ │         cost = qty * fill_price + qty * self.commission     │  │
│ │                                                              │  │
│ │         if side == "BUY":                                   │  │
│ │             self.cash -= cost                               │  │
│ │             self.positions[symbol] = (                      │  │
│ │                 self.positions.get(symbol, 0) + qty         │  │
│ │             )                                                │  │
│ │         else:  # SELL                                       │  │
│ │             self.cash += (qty * fill_price - qty * self.commission)│ │
│ │             self.positions[symbol] -= qty                   │  │
│ │                                                              │  │
│ │         return BrokerOrder(                                 │  │
│ │             order_id=f"backtest_{uuid4()}",                 │  │
│ │             status="FILLED"                                 │  │
│ │         )                                                    │  │
│ │                                                              │  │
│ │ Used for: Historical simulation, strategy testing           │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

BROKER SELECTION (Configuration-driven):
┌───────────────────────────────────────────────────────────────────┐
│ # config/production.yaml                                           │
│ broker:                                                            │
│   type: "ibkr"  # or "alpaca", "backtest"                          │
│   host: "127.0.0.1"                                                │
│   port: 7497  # Live trading                                       │
│   client_id: 1                                                     │
│                                                                     │
│ # Factory pattern                                                  │
│ def create_broker(config: BrokerConfig) -> Broker:                │
│     if config.type == "ibkr":                                      │
│         return IBKRBroker(                                         │
│             host=config.host,                                      │
│             port=config.port,                                      │
│             client_id=config.client_id                             │
│         )                                                           │
│     elif config.type == "alpaca":                                  │
│         return AlpacaBroker(                                       │
│             api_key=config.api_key,                                │
│             api_secret=config.api_secret                           │
│         )                                                           │
│     elif config.type == "backtest":                                │
│         return BacktestBroker(...)                                 │
│     else:                                                           │
│         raise ValueError(f"Unknown broker type: {config.type}")    │
└───────────────────────────────────────────────────────────────────┘
```

---

## 4. MONITORING & ALERTING

### 4.1 System Health Monitoring

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SYSTEM HEALTH MONITORING ARCHITECTURE                 │
└─────────────────────────────────────────────────────────────────────────┘

MONITORING LAYERS:

LAYER 1: INFRASTRUCTURE MONITORING
┌───────────────────────────────────────────────────────────────────┐
│ Component: Server health & resource utilization                   │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ Metrics collected (1-minute intervals):                     │  │
│ │                                                              │  │
│ │ CPU:                                                        │  │
│ │   - Utilization (%)                                         │  │
│ │   - Load average (1/5/15 min)                               │  │
│ │   - Per-core usage                                          │  │
│ │                                                              │  │
│ │ Memory:                                                     │  │
│ │   - Used / Total (GB)                                       │  │
│ │   - Swap usage                                              │  │
│ │   - Available memory                                        │  │
│ │                                                              │  │
│ │ Disk:                                                       │  │
│ │   - /artifacts usage (% full)                               │  │
│ │   - I/O wait time                                           │  │
│ │   - Read/write throughput (MB/s)                            │  │
│ │                                                              │  │
│ │ Network:                                                    │  │
│ │   - Bytes in/out (MB/s)                                     │  │
│ │   - Packet loss (%)                                         │  │
│ │   - Connection count                                        │  │
│ │                                                              │  │
│ │ GPU (if applicable):                                        │  │
│ │   - Utilization (%)                                         │  │
│ │   - Memory used / total (GB)                                │  │
│ │   - Temperature (°C)                                        │  │
│ │   - Power draw (W)                                          │  │
│ │                                                              │  │
│ │ Implementation: Prometheus + Node Exporter                  │  │
│ └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│ Alerts:                                                            │
│   ⚠ WARNING:  CPU > 80% for 5 min                                 │
│   🚨 CRITICAL: CPU > 95% for 2 min                                 │
│   ⚠ WARNING:  Memory > 85%                                        │
│   🚨 CRITICAL: Memory > 95%                                        │
│   ⚠ WARNING:  Disk > 85% full                                     │
│   🚨 CRITICAL: Disk > 95% full                                     │
│   🚨 CRITICAL: GPU temperature > 85°C                              │
└───────────────────────────────────────────────────────────────────┘

LAYER 2: APPLICATION MONITORING
┌───────────────────────────────────────────────────────────────────┐
│ Component: Pipeline execution health                               │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ Metrics collected (per nightly/weekly run):                 │  │
│ │                                                              │  │
│ │ Pipeline Status:                                            │  │
│ │   - Stage completion (boolean per stage)                    │  │
│ │   - Total duration (seconds)                                │  │
│ │   - Stage durations (seconds)                               │  │
│ │   - Error count                                             │  │
│ │                                                              │  │
│ │ Data Quality:                                               │  │
│ │   - Symbols fetched / expected (coverage %)                 │  │
│ │   - Validation pass rate (%)                                │  │
│ │   - Quarantined symbols (count)                             │  │
│ │   - Feature completeness (% non-null)                       │  │
│ │                                                              │  │
│ │ Model Performance:                                          │  │
│ │   - Inference latency (ms)                                  │  │
│ │   - Scores distribution (quantiles)                         │  │
│ │   - Cross-sectional rank correlation (prev day)             │  │
│ │   - Model version hash                                      │  │
│ │                                                              │  │
│ │ Portfolio Metrics:                                          │  │
│ │   - Total equity ($)                                        │  │
│ │   - Daily P&L ($, %)                                        │  │
│ │   - Turnover (%)                                            │  │
│ │   - Position count                                          │  │
│ │   - Cash balance ($)                                        │  │
│ │                                                              │  │
│ │ Execution Metrics:                                          │  │
│ │   - Orders submitted                                        │  │
│ │   - Fill rate (%)                                           │  │
│ │   - Average slippage (bps)                                  │  │
│ │   - Rejected orders (count + reasons)                       │  │
│ │                                                              │  │
│ │ Implementation: Custom metrics logged to TimeSeries DB      │  │
│ │ (InfluxDB or Prometheus)                                    │  │
│ └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│ Alerts:                                                            │
│   ⚠ WARNING:  Pipeline duration > 5 min (150% of baseline)        │
│   🚨 CRITICAL: Pipeline failed (any stage error)                   │
│   ⚠ WARNING:  Data coverage < 95%                                 │
│   🚨 CRITICAL: Data coverage < 90%                                 │
│   ⚠ WARNING:  Fill rate < 95%                                     │
│   🚨 CRITICAL: Fill rate < 85%                                     │
│   🚨 CRITICAL: Daily P&L < -5% (circuit breaker)                   │
└───────────────────────────────────────────────────────────────────┘

LAYER 3: BUSINESS LOGIC MONITORING
┌───────────────────────────────────────────────────────────────────┐
│ Component: Strategy performance & risk metrics                     │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ Metrics collected (daily):                                  │  │
│ │                                                              │  │
│ │ Performance (rolling windows):                              │  │
│ │   - Sharpe ratio (1M, 3M, 1Y)                               │  │
│ │   - Max drawdown (1M, 3M, 1Y)                               │  │
│ │   - Win rate (%)                                            │  │
│ │   - Profit factor                                           │  │
│ │   - Information coefficient (IC)                            │  │
│ │                                                              │  │
│ │ Risk Metrics:                                               │  │
│ │   - Current risk posture (NORMAL/CAUTIOUS/DEFENSIVE)        │  │
│ │   - Portfolio beta                                          │  │
│ │   - Concentration (max position %)                          │  │
│ │   - Sector exposure (% per sector)                          │  │
│ │                                                              │  │
│ │ Model Drift:                                                │  │
│ │   - Score distribution shift (KL divergence)                │  │
│ │   - Feature drift (PSI - Population Stability Index)        │  │
│ │   - Prediction calibration error                            │  │
│ │                                                              │  │
│ │ Implementation: Custom analytics pipeline                   │  │
│ └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│ Alerts:                                                            │
│   ⚠ WARNING:  Sharpe (30d) < 1.0                                  │
│   🚨 CRITICAL: Sharpe (30d) < 0.5                                  │
│   ⚠ WARNING:  Max drawdown > 15%                                  │
│   🚨 CRITICAL: Max drawdown > 25%                                  │
│   ⚠ WARNING:  Model drift PSI > 0.1                               │
│   🚨 CRITICAL: Model drift PSI > 0.25 (requires retraining)        │
│   ⚠ WARNING:  Win rate (30d) < 50%                                │
└───────────────────────────────────────────────────────────────────┘

ALERTING CHANNELS & ESCALATION:
┌───────────────────────────────────────────────────────────────────┐
│ Severity: WARNING (⚠)                                              │
│   - Slack: #algaie-alerts                                         │
│   - Email: team@example.com                                       │
│   - Action: Review within 4 hours                                 │
│                                                                    │
│ Severity: CRITICAL (🚨)                                             │
│   - Slack: @channel in #algaie-alerts                             │
│   - Email: oncall@example.com                                     │
│   - SMS: Primary on-call engineer                                 │
│   - PagerDuty: Page on-call rotation                              │
│   - Action: Immediate response required                           │
│                                                                    │
│ Escalation Policy:                                                │
│   - No ACK within 15 min → Page secondary on-call                 │
│   - No ACK within 30 min → Page engineering manager               │
│   - Critical P&L loss → Auto-pause trading, notify CEO            │
└───────────────────────────────────────────────────────────────────┘
```

---

## 5. DISASTER RECOVERY & FAILOVER

### 5.1 Failure Scenarios & Recovery Procedures

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DISASTER RECOVERY PLAYBOOK                            │
└─────────────────────────────────────────────────────────────────────────┘

SCENARIO 1: Data Source Outage (API unavailable)
═══════════════════════════════════════════════════════════════════════
┌───────────────────────────────────────────────────────────────────┐
│ Symptoms:                                                          │
│   - HTTP 500/503 errors from Yahoo Finance                        │
│   - Timeout after 30 seconds                                      │
│   - Coverage < 90% after retries                                  │
│                                                                    │
│ Detection:                                                         │
│   - Automated: Data fetching errors logged                        │
│   - Alert: "Data coverage 87% (threshold: 95%)"                   │
│   - Time: 6:02 PM (during nightly cycle)                          │
│                                                                    │
│ Response Procedure:                                                │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ STEP 1: Automatic failover to backup provider               │  │
│ │   - Switch from Yahoo Finance → Polygon.io                  │  │
│ │   - Retry failed symbols with backup                        │  │
│ │   - Expected recovery: 95% coverage in 2 min                │  │
│ │                                                              │  │
│ │ STEP 2: If backup also fails → Use stale data               │  │
│ │   - Load yesterday's canonical data                         │  │
│ │   - Mark as stale in metadata                               │  │
│ │   - Continue pipeline with degraded data                    │  │
│ │   - Alert: "CRITICAL: Using stale data for 13% of symbols"  │  │
│ │                                                              │  │
│ │ STEP 3: If >20% stale → Abort nightly cycle                 │  │
│ │   - Skip inference & execution                              │  │
│ │   - Keep yesterday's positions                              │  │
│ │   - Alert: "CRITICAL: Nightly cycle aborted (data outage)"  │  │
│ │   - Manual intervention required                            │  │
│ └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│ Recovery Time Objective (RTO): 5 minutes                           │
│ Recovery Point Objective (RPO): 1 day (yesterday's data)           │
└───────────────────────────────────────────────────────────────────┘

SCENARIO 2: Model Inference Failure (GPU OOM/crash)
═══════════════════════════════════════════════════════════════════════
┌───────────────────────────────────────────────────────────────────┐
│ Symptoms:                                                          │
│   - torch.cuda.OutOfMemoryError exception                         │
│   - OR GPU driver crash (CUDA error)                              │
│   - Chronos/Selector inference aborted                            │
│                                                                    │
│ Detection:                                                         │
│   - Automated: Exception caught in inference loop                 │
│   - Alert: "CRITICAL: Model inference failed (GPU OOM)"           │
│   - Time: 6:05 PM (during Chronos priors generation)              │
│                                                                    │
│ Response Procedure:                                                │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ STEP 1: Reduce batch size & retry                           │  │
│ │   - Halve batch size (32 → 16)                              │  │
│ │   - Clear GPU cache (torch.cuda.empty_cache())              │  │
│ │   - Retry inference                                         │  │
│ │   - Expected: Success in 1 minute (slower but works)        │  │
│ │                                                              │  │
│ │ STEP 2: If still OOM → Fallback to CPU                      │  │
│ │   - Move model to CPU (slower, ~5 min vs 30 sec)            │  │
│ │   - Alert: "WARNING: Using CPU inference (slow)"            │  │
│ │   - Expected: Success in 5 minutes                          │  │
│ │                                                              │  │
│ │ STEP 3: If CPU also fails → Use yesterday's priors/scores   │  │
│ │   - Load artifacts/priors/chronos_priors_20240114.parquet   │  │
│ │   - Load artifacts/scores/scores_20240114.parquet           │  │
│ │   - Mark as stale in metadata                               │  │
│ │   - Continue with portfolio construction                    │  │
│ │   - Alert: "CRITICAL: Using stale model outputs"            │  │
│ │                                                              │  │
│ │ STEP 4: If >2 consecutive days of stale outputs → HALT      │  │
│ │   - Abort execution                                         │  │
│ │   - Manual intervention required                            │  │
│ │   - Do not submit orders                                    │  │
│ └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│ Recovery Time Objective (RTO): 6 minutes (CPU fallback)            │
│ Recovery Point Objective (RPO): 1 day (yesterday's outputs)        │
└───────────────────────────────────────────────────────────────────┘

SCENARIO 3: Catastrophic P&L Loss (Circuit Breaker)
═══════════════════════════════════════════════════════════════════════
┌───────────────────────────────────────────────────────────────────┐
│ Symptoms:                                                          │
│   - Daily P&L < -5% (triggered during market hours)               │
│   - OR Drawdown > 15% from recent high                            │
│   - Rapid unexpected portfolio devaluation                        │
│                                                                    │
│ Detection:                                                         │
│   - Real-time: Portfolio monitoring every 1 minute                │
│   - Alert: "🚨 CIRCUIT BREAKER TRIGGERED: P&L -5.2%"               │
│   - Time: Any time during market hours                            │
│                                                                    │
│ Response Procedure:                                                │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ STEP 1: IMMEDIATE HALT (AUTOMATED)                          │  │
│ │   - Cancel all pending orders                               │  │
│ │   - Block new order submission                              │  │
│ │   - Set trading_halted = True flag                          │  │
│ │   - Alert: Page CEO, CTO, on-call engineer                  │  │
│ │   - Notification: SMS + Phone call + Slack @channel         │  │
│ │                                                              │  │
│ │ STEP 2: ASSESS CAUSE (MANUAL, immediate)                    │  │
│ │   - Check market-wide movements (SPY, VIX)                  │  │
│ │   - If systemic crash → Normal (wait it out)                │  │
│ │   - If idiosyncratic → Investigate positions                │  │
│ │   - Review recent orders for errors                         │  │
│ │                                                              │  │
│ │ STEP 3: RISK MITIGATION (MANUAL decision)                   │  │
│ │   - Option A: Do nothing (hold through volatility)          │  │
│ │   - Option B: Reduce exposure (liquidate 50% of positions)  │  │
│ │   - Option C: Full liquidation (exit all positions)         │  │
│ │   - Decision by: CEO + CTO consensus                        │  │
│ │                                                              │  │
│ │ STEP 4: POST-MORTEM & RESUMPTION                            │  │
│ │   - After market close: Root cause analysis                 │  │
│ │   - Identify bug/strategy flaw/market event                 │  │
│ │   - Implement fixes if needed                               │  │
│ │   - Resume trading only after approval                      │  │
│ └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│ Recovery Time Objective (RTO): Immediate halt, resume next day minimum│
│ Recovery Point Objective (RPO): N/A (capital preservation priority)│
└───────────────────────────────────────────────────────────────────┘
```

### 5.2 Backup & Restore Procedures

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      BACKUP & RESTORE ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────────────┘

BACKUP STRATEGY:
┌───────────────────────────────────────────────────────────────────┐
│ 1. Artifacts (Daily)                                               │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ Target: /artifacts/ directory                               │  │
│ │ Frequency: Daily at 8 PM ET (after nightly cycle)           │  │
│ │ Method: Incremental snapshots (rsync)                       │  │
│ │ Retention:                                                   │  │
│ │   - Daily: Keep last 30 days                                │  │
│ │   - Weekly: Keep last 12 weeks                              │  │
│ │   - Monthly: Keep last 12 months                            │  │
│ │ Destination: AWS S3 (us-east-1, Glacier for old backups)    │  │
│ │ Compression: gzip                                           │  │
│ │ Encryption: AES-256 (server-side)                           │  │
│ │                                                              │  │
│ │ Backup size: ~50 GB per day (incremental)                   │  │
│ │ Storage cost: ~$15/month (S3 Standard + Glacier)            │  │
│ └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│ 2. Database (Hourly)                                               │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ Target: SQLite artifact_records.db                          │  │
│ │ Frequency: Hourly                                           │  │
│ │ Method: Point-in-time snapshots                             │  │
│ │ Retention: Last 168 hours (7 days)                          │  │
│ │ Destination: Local + S3                                     │  │
│ └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│ 3. Code & Configuration (Git)                                      │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ Target: Source code, config files                           │  │
│ │ Frequency: Every commit (version controlled)                │  │
│ │ Remote: GitHub (private repo)                               │  │
│ │ Branches:                                                    │  │
│ │   - main: Production code                                   │  │
│ │   - staging: Pre-production testing                         │  │
│ │   - dev: Development work                                   │  │
│ │ Tags: Release versions (v1.0.0, v1.1.0, ...)                │  │
│ └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│ 4. Model Checkpoints (Weekly)                                      │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ Target: artifacts/models/                                   │  │
│ │ Frequency: After weekly training (Sunday)                   │  │
│ │ Method: Full copy to S3                                     │  │
│ │ Retention: Last 12 weeks                                    │  │
│ │ Versioning: Enabled (keep all versions)                     │  │
│ │ Size: ~50 MB per checkpoint                                 │  │
│ └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘

RESTORE PROCEDURES:
┌───────────────────────────────────────────────────────────────────┐
│ Scenario: Complete system failure (hardware death)                 │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ STEP 1: Provision new server (15 min)                       │  │
│ │   - Launch EC2 instance (g4dn.xlarge with GPU)              │  │
│ │   - Ubuntu 22.04, 500GB SSD                                 │  │
│ │   - Install system dependencies                             │  │
│ │                                                              │  │
│ │ STEP 2: Restore code (5 min)                                │  │
│ │   $ git clone git@github.com:org/algaie.git                 │  │
│ │   $ cd algaie                                               │  │
│ │   $ git checkout v1.5.3  # Latest production tag            │  │
│ │   $ python -m venv .venv                                    │  │
│ │   $ source .venv/bin/activate                               │  │
│ │   $ pip install -r requirements.txt                         │  │
│ │                                                              │  │
│ │ STEP 3: Restore artifacts (30 min)                          │  │
│ │   $ aws s3 sync s3://algaie-backups/artifacts/ \\           │  │
│ │       /home/ubuntu/algaie/artifacts/ \\                     │  │
│ │       --exclude "*.tmp"                                     │  │
│ │   # Downloads ~50 GB                                        │  │
│ │                                                              │  │
│ │ STEP 4: Restore database (1 min)                            │  │
│ │   $ aws s3 cp \\                                            │  │
│ │       s3://algaie-backups/db/artifact_records.db \\         │  │
│ │       artifacts/registry/artifact_records.db                │  │
│ │                                                              │  │
│ │ STEP 5: Validate restore (5 min)                            │  │
│ │   $ python backend/scripts/verify/validate_artifacts.py     │  │
│ │   ✓ Canonical data: 500/500 symbols                         │  │
│ │   ✓ Models: selector_v1_fold5.pt found                      │  │
│ │   ✓ Latest scores: scores_20240114.parquet found            │  │
│ │                                                              │  │
│ │ STEP 6: Resume operations (immediate)                       │  │
│ │   $ python backend/scripts/run/run_nightly_cycle.py         │  │
│ │   # System back online                                      │  │
│ └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│ Total RTO: ~60 minutes (server provision to live)                  │
│ Total RPO: 24 hours (yesterday's backup)                           │
└───────────────────────────────────────────────────────────────────┘

DISASTER RECOVERY TESTING:
┌───────────────────────────────────────────────────────────────────┐
│ Quarterly DR drill (last Sunday of each quarter):                 │
│ ┌─────────────────────────────────────────────────────────────┐  │
│ │ 1. Simulate server failure (shutdown production)            │  │
│ │ 2. Provision DR server from scratch                         │  │
│ │ 3. Restore from latest backup                               │  │
│ │ 4. Run nightly cycle on DR server                           │  │
│ │ 5. Validate output matches expected                         │  │
│ │ 6. Document time to recovery                                │  │
│ │ 7. Identify & fix bottlenecks                               │  │
│ └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│ Success criteria:                                                  │
│   - RTO < 90 minutes                                              │
│   - RPO < 48 hours                                                │
│   - Zero data corruption                                          │
│   - All tests pass on restored system                             │
└───────────────────────────────────────────────────────────────────┘
```

---

## CONCLUSION

This Integration Workflows document provides comprehensive operational guidance for the Algaie trading system, covering:

1. **Nightly Production Cycle** - Complete 3.5-minute automated pipeline from market close to orders ready
2. **Morning Execution** - Pre-market order submission and fill monitoring
3. **Weekly Retraining** - 101-minute model update cycle with walk-forward validation
4. **Broker Integration** - Abstract interface with IBKR/Alpaca/Backtest implementations
5. **Monitoring & Alerting** - 3-layer health monitoring with escalation policies
6. **Disaster Recovery** - Failure scenarios, backup strategy, and restore procedures

**Operational Cadence**:
- **Daily**: Nightly inference (6 PM) + morning execution (9:25 AM)
- **Weekly**: Model retraining (Sunday 6 PM)
- **Monthly**: Chronos teacher retraining (if needed)
- **Quarterly**: DR drill & system health review

**Key Metrics**:
- Nightly cycle: 3.5 minutes (SLA: <5 min)
- Fill rate: 97.8% (SLA: >95%)
- Data coverage: 99% (SLA: >95%)
- System uptime: 99.9% (SLA: >99%)

**Cross-References**:
- Visual Diagrams for workflow visualizations
- System Analysis Parts 1-4 for component details
- Deep Dive Validation for data quality procedures

---

*End of Integration Workflows Document - Last Updated: 2026-02-13*
