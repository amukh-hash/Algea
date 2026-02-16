# ALGAIE SYSTEM VISUAL DIAGRAMS

*Comprehensive Visual Architecture & Data Flow Representations*

---

## OVERVIEW

This document provides detailed visual representations of the Algaie trading system architecture, data flows, component interactions, and operational workflows. Each diagram is designed to be self-contained and explanatory, showing both structural relationships and temporal sequences.

**Diagram Categories**:
1. System-Wide Architecture
2. Data Flow Pipelines
3. Component Interactions
4. State Machines & Workflows
5. Training vs Inference Comparison
6. Time-Series Causality & Windowing

---

## 1. SYSTEM-WIDE ARCHITECTURE

### 1.1 Complete System Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ALGAIE TRADING SYSTEM                                │
│                     (End-to-End ML Trading Pipeline)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
         ┌──────────▼──────────┐           ┌───────────▼──────────┐
         │   FOUNDATION LAYER   │           │   OPERATIONAL LAYER   │
         │   (Infrastructure)   │           │   (Runtime Systems)   │
         └──────────┬──────────┘           └───────────┬──────────┘
                    │                                   │
        ┌───────────┼───────────┐          ┌───────────┼───────────┐
        │           │           │          │           │           │
   ┌────▼───┐  ┌───▼────┐  ┌──▼───┐  ┌───▼────┐  ┌──▼────┐  ┌───▼────┐
   │ Config │  │ Paths  │  │ Cont │  │ Train  │  │ Infer │  │ Exec   │
   │ System │  │ System │  │ Sys  │  │ System │  │ System│  │ System │
   └────┬───┘  └───┬────┘  └──┬───┘  └───┬────┘  └──┬────┘  └───┬────┘
        │          │          │          │          │          │
        └──────────┴──────────┴──────────┴──────────┴──────────┘
                              │
                   ┌──────────┴──────────┐
                   │                     │
        ┌──────────▼──────────┐    ┌────▼─────────────────┐
        │     DATA LAYER       │    │    MODEL LAYER       │
        │  (Ingestion & Prep)  │    │  (ML Components)     │
        └──────────┬──────────┘    └────┬─────────────────┘
                   │                     │
     ┌─────────────┼─────────────┐      │
     │             │             │      │
┌────▼────┐  ┌────▼────┐  ┌─────▼──┐   │
│Canonical│  │Universe │  │ Market │   │
│  OHLCV  │  │ Builder │  │ Enrich │   │
└────┬────┘  └────┬────┘  └─────┬──┘   │
     │            │             │      │
     └────────────┴─────────────┘      │
                  │                    │
            ┌─────▼─────┐              │
            │ Feature   │              │
            │ Builder   │◄─────────────┘
            └─────┬─────┘
                  │
     ┌────────────┼────────────┐
     │            │            │
┌────▼─────┐ ┌───▼──────┐ ┌──▼────────┐
│ Chronos  │ │  Rank    │ │ Portfolio │
│ Teacher  │ │Transform │ │ Optimizer │
└────┬─────┘ └───┬──────┘ └──┬────────┘
     │           │           │
     └───────────┴───────────┘
                 │
         ┌───────▼───────┐
         │   Execution   │
         │   Engine      │
         └───────────────┘
```

**Layer Responsibilities**:

1. **Foundation Layer**: Configuration management, path resolution, contracts, calendar
2. **Data Layer**: Raw data ingestion, validation, universe construction, market enrichment
3. **Model Layer**: Feature engineering, ML model training/inference, portfolio optimization
4. **Operational Layer**: Training pipelines, inference workflows, order execution

---

### 1.2 Data Flow: Raw Data → Execution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     END-TO-END DATA FLOW PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────┘

STAGE 1: RAW DATA INGESTION
┌──────────────┐
│ External API │  (Yahoo Finance, Polygon.io, etc.)
└──────┬───────┘
       │ HTTP requests
       ▼
┌──────────────────────────────────────┐
│  Raw OHLCV Data                      │
│  ┌────────────────────────────┐     │
│  │ symbol  date        close  │     │
│  │ AAPL    2024-01-02  185.64 │     │
│  │ AAPL    2024-01-03  184.25 │     │
│  └────────────────────────────┘     │
└──────────────┬───────────────────────┘
               │
               ▼

STAGE 2: VALIDATION & CANONICALIZATION
┌───────────────────────────────────────┐
│  8-Stage Validation Pipeline          │
│  ├─ 1. Required columns               │
│  ├─ 2. Date format & sorting          │
│  ├─ 3. Non-finite values              │
│  ├─ 4. Price positivity               │
│  ├─ 5. Volume non-negativity          │
│  ├─ 6. Close column resolution        │
│  ├─ 7. Invalid returns (quarantine)   │
│  └─ 8. Date range coverage            │
└───────────────┬───────────────────────┘
                │ [PASS]
                ▼
┌────────────────────────────────────────┐
│  Canonical OHLCV (Parquet)             │
│  /artifacts/canonical/AAPL.parquet     │
│  ┌──────────────────────────────┐     │
│  │ Validated, sorted, typed      │     │
│  │ ~5MB per symbol, ~500 symbols │     │
│  └──────────────────────────────┘     │
└────────────────┬───────────────────────┘
                 │
                 ▼

STAGE 3: UNIVERSE CONSTRUCTION
┌─────────────────────────────────────────┐
│  UniverseBuilder (4-stage filter)       │
│  ┌────────────────────────────────┐    │
│  │ Observable → Tradable → Tier   │    │
│  │ 500 stocks → 450 → 3 tiers     │    │
│  └────────────────────────────────┘    │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────┐
│  Universe DataFrame                       │
│  ┌────────────────────────────────┐      │
│  │ symbol date   is_tradable tier │      │
│  │ AAPL   2024-01 True         0  │      │
│  │ MSFT   2024-01 True         0  │      │
│  └────────────────────────────────┘      │
└──────────────────┬───────────────────────┘
                   │
                   ▼

STAGE 4: MARKET ENRICHMENT (Parallel Streams)
┌────────────────────────┐   ┌─────────────────────┐
│  Breadth Indicators    │   │  Market Covariates  │
│  ┌──────────────────┐  │   │  ┌───────────────┐  │
│  │ AD Line          │  │   │  │ SPY ret       │  │
│  │ BPI (21-day)     │  │   │  │ QQQ ret       │  │
│  └──────────────────┘  │   │  │ RV21 change   │  │
└────────────┬───────────┘   │  └───────────────┘  │
             │               └──────────┬──────────┘
             │                          │
             └──────────┬───────────────┘
                        ▼
              ┌──────────────────────┐
              │  MarketFrame Builder │
              │  (Panel join)        │
              └──────────┬───────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  MarketFrame (Long-format panel)                    │
│  ┌───────────────────────────────────────────────┐  │
│  │ symbol date   close spy_ret rv21_chg ad_line │  │
│  │ AAPL   2024-01 185.6 0.015   0.02     0.35   │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼

STAGE 5: FEATURE ENGINEERING
┌──────────────────────────────────────────┐
│  SelectorFeatureBuilder (6 substages)    │
│  ┌────────────────────────────────┐     │
│  │ 1. Raw features (returns, vol) │     │
│  │ 2. Rank normalization [-1,+1] │     │
│  │ 3. Forward return targets      │     │
│  │ 4. Risk adjustment             │     │
│  │ 5. Breadth filtering           │     │
│  │ 6. Bounds validation           │     │
│  └────────────────────────────────┘     │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│  Features DataFrame                                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │ symbol date ret_1d_rank vol_rank y_rank y_trade │  │
│  │ AAPL   2024  0.45       0.23     0.67   1       │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼

STAGE 6A: TEACHER MODEL (Chronos - Weekly)
┌─────────────────────────────────────────┐
│  Chronos2Teacher                        │
│  ┌───────────────────────────────┐     │
│  │ T5 base + LoRA                │     │
│  │ Context: 512 days historical  │     │
│  │ Horizon: 1-21 days forward    │     │
│  │ Output: Quantile distribution │     │
│  └───────────────────────────────┘     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────┐
│  Priors (per symbol, per date)               │
│  ┌────────────────────────────────────┐     │
│  │ symbol date drift vol tail prob_up │     │
│  │ AAPL   2024 0.02  0.18 0.15  0.55  │     │
│  └────────────────────────────────────┘     │
└──────────────────┬───────────────────────────┘
                   │
                   ▼

STAGE 6B: SELECTOR MODEL (RankTransformer - Nightly)
┌──────────────────────────────────────────────┐
│  RankTransformer                             │
│  ┌────────────────────────────────────┐     │
│  │ Transformer encoder (6 layers)     │     │
│  │ Input: Features + Priors           │     │
│  │ Multi-task heads:                  │     │
│  │   - Quantile regression            │     │
│  │   - Direction classification       │     │
│  │   - Risk prediction                │     │
│  └────────────────────────────────────┘     │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Selector Scores (cross-sectional ranks)    │
│  ┌───────────────────────────────────┐     │
│  │ symbol date score rank percentile │     │
│  │ AAPL   2024 0.82  15   96.7%      │     │
│  │ MSFT   2024 0.75  28   93.8%      │     │
│  └───────────────────────────────────┘     │
└─────────────────┬───────────────────────────┘
                  │
                  ▼

STAGE 7: PORTFOLIO CONSTRUCTION
┌──────────────────────────────────────────┐
│  PortfolioOptimizer                      │
│  ┌────────────────────────────────┐     │
│  │ 1. Top N selection (N=20-50)   │     │
│  │ 2. HRP allocation              │     │
│  │ 3. Risk posture adjustment     │     │
│  │ 4. Position sizing             │     │
│  └────────────────────────────────┘     │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  Target Portfolio                            │
│  ┌────────────────────────────────────┐     │
│  │ symbol shares weight value         │     │
│  │ AAPL   50     0.05    $9,280       │     │
│  │ MSFT   25     0.05    $9,280       │     │
│  │ ...    ...    ...     ...          │     │
│  │ [CASH] -      0.00    $0           │     │
│  └────────────────────────────────────┘     │
└──────────────────┬───────────────────────────┘
                   │
                   ▼

STAGE 8: ORDER GENERATION & EXECUTION
┌─────────────────────────────────────────────┐
│  ExecutionEngine                            │
│  ┌───────────────────────────────────┐     │
│  │ Diff: Current → Target            │     │
│  │ Generate orders (MKT/LMT)         │     │
│  │ Simulate fills (backtest)         │     │
│  │ OR Submit to broker (live)        │     │
│  └───────────────────────────────────┘     │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────┐
│  Executed Trades                             │
│  ┌────────────────────────────────────┐     │
│  │ symbol side qty price timestamp    │     │
│  │ AAPL   BUY  10  185.6 09:31:05     │     │
│  │ MSFT   SELL 5   380.2 09:31:12     │     │
│  └────────────────────────────────────┘     │
└──────────────────────────────────────────────┘
```

**Timeline**:
- Stages 1-5: Daily at market close (6-7 PM ET)
- Stage 6A: Weekly (Sunday evening, ~10 min GPU)
- Stage 6B: Nightly (post-market, ~1 min)
- Stages 7-8: Pre-market next day (8:30-9:30 AM ET)

---

## 2. DATA FLOW PIPELINES

### 2.1 Canonical Data Validation Flow (8 Stages)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CANONICAL DATA VALIDATION PIPELINE                    │
│                         (validate_canonical_ohlcv)                       │
└─────────────────────────────────────────────────────────────────────────┘

INPUT: Raw DataFrame
┌─────────────────────────────────┐
│ ticker  date        close  vol  │
│ AAPL    2024-01-02  185.6  1.2M │
│ AAPL    2024-01-03  NaN    0.8M │  ← Non-finite value
│ AAPL    2024-01-05  -10.0  0.9M │  ← Negative price
└─────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 1: Required Columns Check                         │
│ ┌──────────────────────────────────────────────────┐    │
│ │ Expected: [symbol, date, open, high, low, ...]  │    │
│ │ Actual:   [ticker, date, close, vol]            │    │
│ │ Missing:  [open, high, low]  ← ERROR            │    │
│ └──────────────────────────────────────────────────┘    │
└──────────────────────────┬───────────────────────────────┘
                           │ [PASS if all present]
                           ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 2: Date Validation & Sorting                      │
│ ┌──────────────────────────────────────────────────┐    │
│ │ 1. Parse date column → datetime64                │    │
│ │ 2. Check timezone-awareness (must be naive)      │    │
│ │ 3. Check for null dates                          │    │
│ │ 4. Sort by date ascending                        │    │
│ │ 5. Verify monotonic increasing                   │    │
│ └──────────────────────────────────────────────────┘    │
└──────────────────────────┬───────────────────────────────┘
                           │ [PASS]
                           ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 3: Non-Finite Value Detection                     │
│ ┌──────────────────────────────────────────────────┐    │
│ │ Check columns: [open, high, low, close, volume]  │    │
│ │ Detect: NaN, +Inf, -Inf                          │    │
│ │                                                   │    │
│ │ Result: ValidationIssue(                         │    │
│ │   message="Non-finite values in 'close'",       │    │
│ │   rows=[2]  ← Index of NaN row                  │    │
│ │ )                                                │    │
│ └──────────────────────────────────────────────────┘    │
└──────────────────────────┬───────────────────────────────┘
                           │ [FAIL if any found] → Raise ValidationError
                           ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 4: Price Positivity Check                         │
│ ┌──────────────────────────────────────────────────┐    │
│ │ Columns: [open, high, low, close]                │    │
│ │ Constraint: ALL > 0                              │    │
│ │                                                   │    │
│ │ Row 3: close = -10.0  ← VIOLATION                │    │
│ │                                                   │    │
│ │ Result: ValidationIssue(                         │    │
│ │   message="Non-positive prices in 'close'",     │    │
│ │   rows=[3]                                       │    │
│ │ )                                                │    │
│ └──────────────────────────────────────────────────┘    │
└──────────────────────────┬───────────────────────────────┘
                           │ [FAIL] → Raise ValidationError
                           ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 5: Volume Non-Negativity Check                    │
│ ┌──────────────────────────────────────────────────┐    │
│ │ Column: volume                                    │    │
│ │ Constraint: ALL >= 0                             │    │
│ │                                                   │    │
│ │ Negative volumes → ValidationError               │    │
│ └──────────────────────────────────────────────────┘    │
└──────────────────────────┬───────────────────────────────┘
                           │ [PASS]
                           ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 6: Close Column Resolution                        │
│ ┌──────────────────────────────────────────────────┐    │
│ │ Check for multiple close columns:                │    │
│ │ - "close"                                         │    │
│ │ - "adj_close" or "adjusted_close"                │    │
│ │                                                   │    │
│ │ Priority: adj_close > close                      │    │
│ │                                                   │    │
│ │ If both exist:                                    │    │
│ │   Rename adj_close → close                       │    │
│ │   Drop original close                            │    │
│ └──────────────────────────────────────────────────┘    │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 7: Invalid Return Detection (QUARANTINE)          │
│ ┌──────────────────────────────────────────────────┐    │
│ │ Compute: ratio = close / close.shift(1)          │    │
│ │                                                   │    │
│ │ Invalid conditions:                              │    │
│ │   1. ratio <= 0  (zero/negative price)           │    │
│ │   2. (ratio - 1) <= -1  (>100% drop)            │    │
│ │                                                   │    │
│ │ Count: invalid_frac = n_invalid / n_total        │    │
│ │                                                   │    │
│ │ If invalid_frac > max_invalid_frac (0.02):       │    │
│ │   ┌────────────────────────────────────┐         │    │
│ │   │ QUARANTINE SYSTEM ACTIVATED        │         │    │
│ │   │                                     │         │    │
│ │   │ 1. Create report:                  │         │    │
│ │   │    - Symbol name                   │         │    │
│ │   │    - Invalid fraction              │         │    │
│ │   │    - Row details (date, price)     │         │    │
│ │   │                                     │         │    │
│ │   │ 2. Write to quarantine directory:  │         │    │
│ │   │    /artifacts/quarantine/AAPL.json │         │    │
│ │   │                                     │         │    │
│ │   │ 3. RAISE ValidationError           │         │    │
│ │   │    (Block canonical write)         │         │    │
│ │   └────────────────────────────────────┘         │    │
│ │                                                   │    │
│ │ Else: Continue (tolerate small errors)           │    │
│ └──────────────────────────────────────────────────┘    │
└──────────────────────────┬───────────────────────────────┘
                           │ [PASS]
                           ▼
┌──────────────────────────────────────────────────────────┐
│ STAGE 8: Date Range Coverage Check                      │
│ ┌──────────────────────────────────────────────────┐    │
│ │ If config specifies date range:                  │    │
│ │   start_date = "2020-01-01"                      │    │
│ │   end_date   = "2024-12-31"                      │    │
│ │                                                   │    │
│ │ Check:                                            │    │
│ │   df["date"].min() <= start_date                 │    │
│ │   df["date"].max() >= end_date                   │    │
│ │                                                   │    │
│ │ If violated → ValidationError                    │    │
│ └──────────────────────────────────────────────────┘    │
└──────────────────────────┬───────────────────────────────┘
                           │ [ALL STAGES PASSED]
                           ▼
┌─────────────────────────────────────────────────────────┐
│ OUTPUT: Validated Canonical DataFrame                   │
│ ┌─────────────────────────────────────────────────┐    │
│ │ symbol  date        open   high   low    close │    │
│ │ AAPL    2024-01-02  184.2  186.1  183.8  185.6 │    │
│ │ AAPL    2024-01-03  185.5  187.3  185.0  186.8 │    │
│ │ AAPL    2024-01-04  186.9  188.2  186.5  187.4 │    │
│ └─────────────────────────────────────────────────┘    │
│                                                          │
│ Ready for Parquet write to:                             │
│ /artifacts/canonical/AAPL.parquet                       │
└─────────────────────────────────────────────────────────┘
```

**Error Handling Behavior**:
- Stages 1-6: **Hard Fail** - Raise ValidationError immediately
- Stage 7: **Conditional** - Quarantine if threshold exceeded, else tolerate
- Stage 8: **Optional** - Only if date range configured

**Performance**: ~50ms per symbol for 1000-day dataset (Polars lazy eval)

---

### 2.2 Universe Construction Pipeline (4-Stage Filter Cascade)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     UNIVERSE CONSTRUCTION PIPELINE                       │
│                          (UniverseBuilder.build)                         │
└─────────────────────────────────────────────────────────────────────────┘

INPUT: Daily OHLCV for all symbols
┌────────────────────────────────────────────────────────────┐
│ symbol  date        close  volume  dollar_vol  hist_days  │
│ AAPL    2024-01-15  185.6  50.2M   9.31B       1253       │
│ MSFT    2024-01-15  380.4  22.1M   8.41B       1253       │
│ TSLA    2024-01-15  210.7  95.3M   20.1B       1125       │
│ PENNY   2024-01-15  0.05   1.2M    0.06M       45         │  ← Penny stock
└────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ FILTER STAGE 1: Observable Universe                         │
│ ┌──────────────────────────────────────────────────────┐    │
│ │ Criteria:                                             │    │
│ │   close > min_price (default: $1.00)                 │    │
│ │   hist_days >= min_history_days (default: 252)       │    │
│ │                                                       │    │
│ │ Purpose: Remove penny stocks & newly listed          │    │
│ │                                                       │    │
│ │ Input:  500 symbols                                  │    │
│ │ Output: 485 symbols (15 removed)                     │    │
│ │                                                       │    │
│ │ Removed symbols:                                      │    │
│ │   PENNY: close = $0.05 < $1.00                       │    │
│ │   NEWCO: hist_days = 45 < 252                        │    │
│ └──────────────────────────────────────────────────────┘    │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ FILTER STAGE 2: Tradable Universe                           │
│ ┌──────────────────────────────────────────────────────┐    │
│ │ Criteria:                                             │    │
│ │   is_observable == True  (from Stage 1)              │    │
│ │   dollar_vol >= min_dollar_vol (default: $5M)        │    │
│ │                                                       │    │
│ │ Purpose: Ensure sufficient liquidity for trading     │    │
│ │                                                       │    │
│ │ Input:  485 symbols (observable)                     │    │
│ │ Output: 450 symbols (35 removed)                     │    │
│ │                                                       │    │
│ │ Removed symbols:                                      │    │
│ │   ILLIQ: dollar_vol = $2.3M < $5M                    │    │
│ └──────────────────────────────────────────────────────┘    │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ FILTER STAGE 3: Tier Assignment                             │
│ ┌──────────────────────────────────────────────────────┐    │
│ │ Method: Dollar volume quantile bins                  │    │
│ │                                                       │    │
│ │ Breakpoints (configurable):                          │    │
│ │   Tier 0 (Mega): 67th+ percentile (top 33%)          │    │
│ │   Tier 1 (Mid):  33rd-67th percentile (middle 34%)   │    │
│ │   Tier 2 (Small): 0-33rd percentile (bottom 33%)     │    │
│ │                                                       │    │
│ │ Example on 2024-01-15 (450 tradable symbols):        │    │
│ │                                                       │    │
│ │   Dollar volume distribution:                        │    │
│ │   ┌───────────────────────────────────┐              │    │
│ │   │  P33 = $50M                       │              │    │
│ │   │  P67 = $500M                      │              │    │
│ │   └───────────────────────────────────┘              │    │
│ │                                                       │    │
│ │   Tier 0: TSLA ($20.1B), AAPL ($9.3B), ...          │    │
│ │           → 150 symbols                              │    │
│ │   Tier 1: MSFT ($8.4B), NVDA ($3.2B), ...           │    │
│ │           → 153 symbols                              │    │
│ │   Tier 2: Remaining symbols                          │    │
│ │           → 147 symbols                              │    │
│ │                                                       │    │
│ │ Assignment logic:                                     │    │
│ │   tier = pd.cut(                                     │    │
│ │       dollar_vol,                                    │    │
│ │       bins=[-np.inf, p33, p67, np.inf],             │    │
│ │       labels=[2, 1, 0]  # Reverse: 0 = highest      │    │
│ │   )                                                  │    │
│ └──────────────────────────────────────────────────────┘    │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ FILTER STAGE 4: Weight Normalization                        │
│ ┌──────────────────────────────────────────────────────┐    │
│ │ Purpose: Tier-weighted portfolio allocation          │    │
│ │                                                       │    │
│ │ Algorithm:                                            │    │
│ │   1. Invert tier (higher tier = more weight)         │    │
│ │      inv_tier = max_tier - tier + 1                  │    │
│ │      (Tier 0 → 3, Tier 1 → 2, Tier 2 → 1)           │    │
│ │                                                       │    │
│ │   2. Group by date, compute mean inv_tier            │    │
│ │      date_mean = inv_tier.mean() per date            │    │
│ │                                                       │    │
│ │   3. Normalize: weight = inv_tier / date_mean        │    │
│ │                                                       │    │
│ │ Example on 2024-01-15:                               │    │
│ │                                                       │    │
│ │   Tier 0 (150 stocks): inv_tier = 3                  │    │
│ │   Tier 1 (153 stocks): inv_tier = 2                  │    │
│ │   Tier 2 (147 stocks): inv_tier = 1                  │    │
│ │                                                       │    │
│ │   Total inv_tier = 150×3 + 153×2 + 147×1 = 903       │    │
│ │   Date mean = 903 / 450 = 2.007                      │    │
│ │                                                       │    │
│ │   Weights:                                            │    │
│ │     Tier 0: 3 / 2.007 = 1.495                        │    │
│ │     Tier 1: 2 / 2.007 = 0.997                        │    │
│ │     Tier 2: 1 / 2.007 = 0.498                        │    │
│ │                                                       │    │
│ │   Verification: Sum = 150×1.495 + 153×0.997 +        │    │
│ │                       147×0.498 = 450.0 ✓            │    │
│ └──────────────────────────────────────────────────────┘    │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT: Universe DataFrame                                      │
│ ┌─────────────────────────────────────────────────────────┐    │
│ │ symbol  date        is_observable  is_tradable  tier  weight│
│ │ AAPL    2024-01-15  True           True         0     1.495 │
│ │ MSFT    2024-01-15  True           True         1     0.997 │
│ │ TSLA    2024-01-15  True           True         0     1.495 │
│ │ ILLIQ   2024-01-15  True           False        NaN   NaN   │
│ │ PENNY   2024-01-15  False          False        NaN   NaN   │
│ └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│ Write to: /artifacts/eligibility/universe.parquet              │
└─────────────────────────────────────────────────────────────────┘
```

**Daily Re-tiering**:
- Universe is reconstructed EVERY trading day
- Symbols can move between tiers based on volume changes
- No "sticky" tier assignment (fully dynamic)

**Performance**: ~30 seconds for 500 symbols × 1000 days

---

### 2.3 Feature Engineering Pipeline (6 Substages)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  SELECTOR FEATURE ENGINEERING PIPELINE                   │
│                      (SelectorFeatureBuilder.build)                      │
└─────────────────────────────────────────────────────────────────────────┘

INPUT: MarketFrame (OHLCV + Covariates + Breadth)
┌──────────────────────────────────────────────────────────────────┐
│ symbol  date        close  volume  spy_ret  rv21_chg  ad_line   │
│ AAPL    2024-01-15  185.6  50.2M   0.015    0.02      0.35      │
│ MSFT    2024-01-15  380.4  22.1M   0.015    0.02      0.35      │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│ SUBSTAGE 1: Raw Feature Computation (Polars Lazy)               │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ Operations (per symbol, time-series):                  │      │
│ │                                                         │      │
│ │ 1. Log Returns:                                        │      │
│ │    log_return_1d = log(close / close.shift(1))        │      │
│ │    log_return_5d = log_return_1d.rolling_sum(5)       │      │
│ │    log_return_21d = log_return_1d.rolling_sum(21)     │      │
│ │                                                         │      │
│ │ 2. Volatility:                                         │      │
│ │    volatility_20d = log_return_1d.rolling_std(20)     │      │
│ │    volatility_60d = log_return_1d.rolling_std(60)     │      │
│ │                                                         │      │
│ │ 3. Relative Volume:                                    │      │
│ │    vol_ma_21d = volume.rolling_mean(21)               │      │
│ │    rel_volume = volume / vol_ma_21d                   │      │
│ │                                                         │      │
│ │ 4. Price Momentum:                                     │      │
│ │    rsi_14 = RSI(close, window=14)                     │      │
│ │    mom_ratio = close / close.shift(21)                │      │
│ │                                                         │      │
│ │ 5. Forward Returns (TARGETS):                         │      │
│ │    fwd_return_1d = log(close.shift(-1) / close)       │      │
│ │    fwd_return_5d = log(close.shift(-5) / close)       │      │
│ │    fwd_volatility = fwd_return_1d.rolling_std(21)     │      │
│ │                                                         │      │
│ │ Lazy Execution: All expressions built, not computed    │      │
│ └────────────────────────────────────────────────────────┘      │
└──────────────────────────┬───────────────────────────────────────┘
                           │ .collect()  ← Trigger computation
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ SUBSTAGE 2: Cross-Sectional Rank Normalization                  │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ Purpose: Convert features to [-1, +1] scale            │      │
│ │                                                         │      │
│ │ Formula (per date, cross-sectional):                   │      │
│ │   r = rank(feature) - 1  (0-indexed)                  │      │
│ │   N = count(symbols) on date                          │      │
│ │   x_norm = 2 × (r / (N - 1)) - 1                      │      │
│ │                                                         │      │
│ │ Example on 2024-01-15 (450 symbols):                  │      │
│ │                                                         │      │
│ │   log_return_5d values:                               │      │
│ │     AAPL:  0.025  → rank 300 → (2×300/449-1) = 0.337 │      │
│ │     MSFT:  0.018  → rank 250 → (2×250/449-1) = 0.114 │      │
│ │     TSLA: -0.042  → rank  50 → (2×50/449-1)  = -0.777│      │
│ │                                                         │      │
│ │ Normalized features:                                   │      │
│ │   - log_return_1d_rank                                │      │
│ │   - log_return_5d_rank                                │      │
│ │   - log_return_21d_rank                               │      │
│ │   - volatility_20d_rank                               │      │
│ │   - rel_volume_rank                                   │      │
│ │   - rsi_14_rank                                       │      │
│ │                                                         │      │
│ │ Properties:                                            │      │
│ │   - Robust to outliers (rank-based)                   │      │
│ │   - Stationary distribution                           │      │
│ │   - Cross-sectional comparability                     │      │
│ └────────────────────────────────────────────────────────┘      │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ SUBSTAGE 3: Target Label Construction                           │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ Two targets created:                                   │      │
│ │                                                         │      │
│ │ 1. y_rank (Regression Target):                        │      │
│ │    y_rank = rank_normalize(fwd_return_5d)             │      │
│ │    → Cross-sectional rank of forward returns          │      │
│ │                                                         │      │
│ │ 2. y_trade (Binary Classification):                   │      │
│ │    risk_adj = y_rank / (fwd_volatility + ε)           │      │
│ │    threshold = 70th percentile of risk_adj per date   │      │
│ │    y_trade = 1 if risk_adj >= threshold else 0        │      │
│ │                                                         │      │
│ │ Example on 2024-01-15:                                │      │
│ │                                                         │      │
│ │   AAPL:                                                │      │
│ │     fwd_return_5d = 0.032 → y_rank = 0.67             │      │
│ │     fwd_volatility = 0.018                            │      │
│ │     risk_adj = 0.67 / 0.018 = 37.2                    │      │
│ │     P70(risk_adj) = 25.0                              │      │
│ │     y_trade = 1  (37.2 >= 25.0) ✓                     │      │
│ │                                                         │      │
│ │   MSFT:                                                │      │
│ │     fwd_return_5d = 0.015 → y_rank = 0.25             │      │
│ │     fwd_volatility = 0.012                            │      │
│ │     risk_adj = 0.25 / 0.012 = 20.8                    │      │
│ │     y_trade = 0  (20.8 < 25.0)                        │      │
│ └────────────────────────────────────────────────────────┘      │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ SUBSTAGE 4: Breadth Filtering                                   │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ Purpose: Remove days with poor market breadth          │      │
│ │                                                         │      │
│ │ Criteria:                                              │      │
│ │   market_breadth_ad < -0.20  (bearish threshold)      │      │
│ │   OR                                                   │      │
│ │   market_breadth_bpi_21d < 0.35  (weak momentum)      │      │
│ │                                                         │      │
│ │ Action: Drop all rows for that date                   │      │
│ │                                                         │      │
│ │ Example:                                               │      │
│ │   2024-01-15: ad = -0.25, bpi = 0.30                  │      │
│ │   → FILTER OUT (both conditions violated)             │      │
│ │   → Remove all 450 symbol rows for this date          │      │
│ │                                                         │      │
│ │ Typical retention: ~60-70% of trading days             │      │
│ └────────────────────────────────────────────────────────┘      │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ SUBSTAGE 5: Feature Bounds Validation                           │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ Check all normalized features in valid range:          │      │
│ │   -1 <= feature_rank <= +1                            │      │
│ │                                                         │      │
│ │ Also check:                                            │      │
│ │   volatility >= 0                                     │      │
│ │   rel_volume >= 0                                     │      │
│ │   0 <= rsi_14 <= 100                                  │      │
│ │                                                         │      │
│ │ If violations found → Raise ValidationError           │      │
│ └────────────────────────────────────────────────────────┘      │
└──────────────────────────┬───────────────────────────────────────┘
                           │ [PASS]
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ SUBSTAGE 6: Final Assembly & Schema Validation                  │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ 1. Select final feature columns:                       │      │
│ │    - symbol, date                                      │      │
│ │    - *_rank features (normalized)                     │      │
│ │    - Market covariates (spy_ret, rv21_chg, etc.)      │      │
│ │    - Breadth indicators (ad_line, bpi_21d)            │      │
│ │    - Targets (y_rank, y_trade)                        │      │
│ │                                                         │      │
│ │ 2. Validate schema contract:                          │      │
│ │    check_schema_contract(df, "selector_features_v1")  │      │
│ │                                                         │      │
│ │ 3. Drop rows with NaN in features (warmup period)     │      │
│ │                                                         │      │
│ │ 4. Sort by [date, symbol]                             │      │
│ └────────────────────────────────────────────────────────┘      │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ OUTPUT: Feature DataFrame                                        │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ Columns (28 total):                                    │      │
│ │   - symbol, date                                       │      │
│ │   - log_return_1d_rank, log_return_5d_rank, ...       │      │
│ │   - volatility_20d_rank, rel_volume_rank, ...         │      │
│ │   - spy_ret, qqq_ret, rv21_chg, ief_ret               │      │
│ │   - market_breadth_ad, market_breadth_bpi_21d         │      │
│ │   - y_rank, y_trade                                   │      │
│ │                                                         │      │
│ │ Shape: ~180,000 rows (450 symbols × 400 days)          │      │
│ │        after breadth filtering & warmup drop           │      │
│ │                                                         │      │
│ │ Memory: ~15MB (Polars efficient dtypes)                │      │
│ └────────────────────────────────────────────────────────┘      │
│                                                                  │
│ Write to: /artifacts/features/selector_features.parquet         │
└──────────────────────────────────────────────────────────────────┘
```

**Performance Optimization**:
- Polars lazy evaluation: Build full expression tree, optimize, execute once
- ~2 minutes for 500 symbols × 1000 days (vs ~20 min with Pandas)
- 10x memory reduction vs Pandas equivalent

---

## 3. COMPONENT INTERACTIONS

### 3.1 Model Training Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING ECOSYSTEM                              │
│                  (Two-Stage Teacher-Student Pipeline)                    │
└─────────────────────────────────────────────────────────────────────────┘

STAGE 1: CHRONOS TEACHER TRAINING (Weekly, Sunday evening)
═══════════════════════════════════════════════════════════

┌──────────────────┐
│ Canonical OHLCV  │  Read from /artifacts/canonical/*.parquet
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│ GoldDatasetBuilder                                      │
│ ┌─────────────────────────────────────────────────┐    │
│ │ Per symbol, create time-series windows:         │    │
│ │                                                  │    │
│ │ Context window:  512 days historical            │    │
│ │ Horizon window:  1-21 days forward               │    │
│ │                                                  │    │
│ │ Example for AAPL on 2024-01-15:                 │    │
│ │   context  = [2022-06-01 ... 2024-01-15]        │    │
│ │   horizons = [2024-01-16 ... 2024-02-13]        │    │
│ │                                                  │    │
│ │ Returns format:                                  │    │
│ │   y_t = log(close_t / close_{t-1})              │    │
│ └─────────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Chronos2Teacher Training Loop                           │
│ ┌────────────────────────────────────────────────────┐  │
│ │ Model: T5 base (220M params)                       │  │
│ │ LoRA rank: 8 (reduces trainable to ~2M params)    │  │
│ │                                                     │  │
│ │ Forward pass:                                      │  │
│ │   1. Encode context (512 days) → hidden states    │  │
│ │   2. Quantile head outputs location-scale params  │  │
│ │      μ_t, σ_t, T_t for each horizon t            │  │
│ │                                                     │  │
│ │ Loss: Negative Log-Likelihood (NLL)               │  │
│ │   NLL = -Σ log p(y_t | μ_t, σ_t, T_t)            │  │
│ │                                                     │  │
│ │ Optimizer: AdamW (lr=1e-4, weight_decay=0.01)     │  │
│ │ Batch size: 32 (gradient accumulation 4 → eff=128)│  │
│ │ Epochs: 10                                         │  │
│ │                                                     │  │
│ │ Training time: ~10 minutes (A100 GPU, 500 symbols)│  │
│ └────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Chronos2Teacher Checkpoint                              │
│ /artifacts/models/chronos_teacher_v1_20240115.pt       │
│ ┌────────────────────────────────────────────────────┐  │
│ │ Weights: LoRA adapters only (~8MB)                 │  │
│ │ Base model: HuggingFace T5-base (downloaded once)  │  │
│ └────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       │  Inference mode (nightly)
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Priors Generation (Chronos Inference)                   │
│ ┌────────────────────────────────────────────────────┐  │
│ │ For each symbol on inference date:                 │  │
│ │                                                     │  │
│ │ Input: 512-day historical context                  │  │
│ │ Output: Quantile distribution for next 5 days      │  │
│ │                                                     │  │
│ │ Priors extracted:                                  │  │
│ │   drift = mean of distribution                    │  │
│ │   vol = std of distribution                       │  │
│ │   tail_risk = |P95 - P05| / P50                   │  │
│ │   prob_up = P(return > 0)                         │  │
│ │                                                     │  │
│ │ Example for AAPL on 2024-01-15:                   │  │
│ │   drift = 0.02%                                   │  │
│ │   vol = 1.8%                                      │  │
│ │   tail_risk = 0.15                                │  │
│ │   prob_up = 0.55                                  │  │
│ └────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       └──────────────┐
                                      │
STAGE 2: SELECTOR TRAINING (Weekly, after Chronos)        │
═══════════════════════════════════════════════════        │
                                                           │
┌──────────────────┐                                      │
│ Feature Builder  │  (Section 2.3 pipeline)              │
└────────┬─────────┘                                      │
         │                                                 │
         ▼                                                 │
┌──────────────────────────────────────┐                  │
│ Features DataFrame                   │                  │
│ - Normalized ranks [-1, +1]          │                  │
│ - Market covariates                  │                  │
│ - Targets (y_rank, y_trade)          │                  │
└────────┬─────────────────────────────┘                  │
         │                                                 │
         │  Join on (symbol, date)                        │
         ◄─────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ Combined Training Dataset                                │
│ ┌────────────────────────────────────────────────────┐  │
│ │ Columns:                                            │  │
│ │   - Features (log_return_*_rank, vol_rank, ...)    │  │
│ │   - Priors (drift, vol, tail_risk, prob_up)        │  │
│ │   - Targets (y_rank, y_trade)                      │  │
│ │                                                     │  │
│ │ Shape: ~180K rows × 32 columns                     │  │
│ └────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Walk-Forward Train/Test Split                           │
│ ┌────────────────────────────────────────────────────┐  │
│ │ Walk-forward validation:                           │  │
│ │   Train window:  2 years (504 trading days)        │  │
│ │   Test window:   6 months (126 trading days)       │  │
│ │   Step size:     126 days (non-overlapping test)   │  │
│ │                                                     │  │
│ │ Example splits:                                    │  │
│ │   Fold 1:                                          │  │
│ │     Train: 2020-01-01 → 2021-12-31                 │  │
│ │     Test:  2022-01-01 → 2022-06-30                 │  │
│ │   Fold 2:                                          │  │
│ │     Train: 2020-07-01 → 2022-06-30                 │  │
│ │     Test:  2022-07-01 → 2022-12-31                 │  │
│ │   ...                                              │  │
│ └────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ RankTransformer Training Loop                           │
│ ┌────────────────────────────────────────────────────┐  │
│ │ Model Architecture:                                 │  │
│ │   - Input embedding (32 features → 128 dim)        │  │
│ │   - Transformer encoder (6 layers, 4 heads)        │  │
│ │   - Multi-task heads:                              │  │
│ │       * Quantile head (9 quantiles)                │  │
│ │       * Direction head (binary classification)     │  │
│ │       * Risk head (volatility prediction)          │  │
│ │                                                     │  │
│ │ Loss (weighted combination):                       │  │
│ │   L_total = α·L_quantile + β·L_direction + γ·L_risk│  │
│ │   α=0.5, β=0.3, γ=0.2                              │  │
│ │                                                     │  │
│ │ Optimizer: AdamW (lr=3e-4, cosine schedule)        │  │
│ │ Batch size: 256 (per date, all symbols)            │  │
│ │ Epochs: 20                                         │  │
│ │                                                     │  │
│ │ Training time: ~15 minutes (GPU, per fold)         │  │
│ └────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ RankTransformer Checkpoint                              │
│ /artifacts/models/selector_v1_fold3_20240115.pt        │
│ ┌────────────────────────────────────────────────────┐  │
│ │ Weights: Full model (~5MB)                         │  │
│ │ Scaler state: Feature normalization params         │  │
│ └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

**Training Schedule**:
- **Sunday 6 PM**: Chronos teacher training (~10 min)
- **Sunday 6:15 PM**: Generate priors for all historical dates (~5 min)
- **Sunday 6:20 PM**: Selector training (~1.5 hours for all folds)
- **Sunday 8 PM**: Models ready for week ahead

---

### 3.2 Inference Workflow Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      NIGHTLY INFERENCE WORKFLOW                          │
│                    (Daily at 6 PM ET after market close)                 │
└─────────────────────────────────────────────────────────────────────────┘

T=0: Market Close (4:00 PM ET)
│
├─ Buffer for late trades settlement (2 hours)
│
▼

T=1: Data Ingestion Starts (6:00 PM ET)
┌──────────────────────────────────────────────────────────┐
│ Step 1: Fetch Latest OHLCV                              │
│ ┌────────────────────────────────────────────────────┐  │
│ │ Data source: Yahoo Finance API / Polygon.io        │  │
│ │ Symbols: 500 (from universe)                       │  │
│ │ Date: Today (2024-01-15)                           │  │
│ │                                                     │  │
│ │ Parallel requests (20 concurrent):                 │  │
│ │   fetch(AAPL) ─┐                                   │  │
│ │   fetch(MSFT) ─┤                                   │  │
│ │   fetch(GOOGL)─┤→ ThreadPoolExecutor              │  │
│ │   ...          │                                   │  │
│ │                └→ ~2 minutes total                 │  │
│ └────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
T=2: Validation & Canonicalization (6:02 PM)
┌──────────────────────────────────────────────────────────┐
│ Step 2: 8-Stage Validation Pipeline                     │
│ (See Section 2.1 for full diagram)                      │
│ ┌────────────────────────────────────────────────────┐  │
│ │ Per-symbol validation (parallel):                  │  │
│ │   - 497 PASS → Write to canonical/*.parquet        │  │
│ │   - 3 FAIL → Quarantine + alert                    │  │
│ │                                                     │  │
│ │ Time: ~30 seconds (Polars parallel writes)         │  │
│ └────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
T=3: Universe Update (6:02:30 PM)
┌──────────────────────────────────────────────────────────┐
│ Step 3: Rebuild Universe for Today                      │
│ (See Section 2.2 for full pipeline)                     │
│ ┌────────────────────────────────────────────────────┐  │
│ │ Input: All canonical OHLCV (historical + today)    │  │
│ │ Output: Updated universe with new tiers            │  │
│ │                                                     │  │
│ │ Today's universe (2024-01-15):                     │  │
│ │   Tradable: 450 symbols                            │  │
│ │   Tier 0: 150, Tier 1: 153, Tier 2: 147           │  │
│ │                                                     │  │
│ │ Changes from yesterday:                            │  │
│ │   - NVDA: Tier 1 → Tier 0 (volume surge)          │  │
│ │   - AMD: Tier 0 → Tier 1 (volume decline)         │  │
│ │                                                     │  │
│ │ Time: ~5 seconds                                   │  │
│ └────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
T=4: Market Enrichment (6:02:35 PM)
┌──────────────────────────────────────────────────────────┐
│ Step 4A: Compute Breadth Indicators (Today)             │
│ ┌────────────────────────────────────────────────────┐  │
│ │ Input: All 450 tradable symbols' OHLCV             │  │
│ │                                                     │  │
│ │ AD Line update:                                    │  │
│ │   direction = sign(close - close_yesterday)        │  │
│ │   advancers = 280, decliners = 170                 │  │
│ │   ad = (280 - 170) / 450 = 0.24                    │  │
│ │                                                     │  │
│ │ BPI-21 update:                                     │  │
│ │   Rolling window 2023-12-15 → 2024-01-15          │  │
│ │   Up days: 14 / 21 = 0.67                         │  │
│ │                                                     │  │
│ │ Time: ~2 seconds (vectorized)                      │  │
│ └────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│ Step 4B: Update Market Covariates (Today)               │
│ ┌────────────────────────────────────────────────────┐  │
│ │ Fetch index data:                                  │  │
│ │   SPY: +1.2% → spy_ret = 0.012                     │  │
│ │   QQQ: +1.8% → qqq_ret = 0.018                     │  │
│ │   IWM: +0.9% → iwm_ret = 0.009                     │  │
│ │   IEF: -0.1% → ief_ret = -0.001                    │  │
│ │                                                     │  │
│ │ Update RV21:                                       │  │
│ │   Rolling std of SPY returns (21 days)             │  │
│ │   rv21_level = 0.15 (15% annualized)               │  │
│ │   rv21_chg_1d = (0.15 - 0.148) / 0.148 = 0.014    │  │
│ │                                                     │  │
│ │ Time: ~1 second                                    │  │
│ └────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
T=5: MarketFrame Assembly (6:02:38 PM)
┌──────────────────────────────────────────────────────────┐
│ Step 5: Join All Data Sources                           │
│ ┌────────────────────────────────────────────────────┐  │
│ │ Join keys: (symbol, date)                          │  │
│ │                                                     │  │
│ │ canonical_ohlcv                                    │  │
│ │   .join(universe, on=["symbol", "date"])           │  │
│ │   .join(breadth, on="date")                        │  │
│ │   .join(covariates, on="date")                     │  │
│ │                                                     │  │
│ │ Output: Complete market frame for today            │  │
│ │   450 symbols × 1 date = 450 rows                  │  │
│ │                                                     │  │
│ │ Time: <1 second (indexed joins)                    │  │
│ └────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
T=6: Feature Engineering (6:02:39 PM)
┌──────────────────────────────────────────────────────────┐
│ Step 6: Build Selector Features (Today Only)            │
│ (See Section 2.3 for full pipeline)                     │
│ ┌────────────────────────────────────────────────────┐  │
│ │ Raw features computed (need historical context):   │  │
│ │   - log_return_5d: last 5 days                     │  │
│ │   - volatility_20d: last 20 days                   │  │
│ │   - rel_volume: volume / MA(21)                    │  │
│ │                                                     │  │
│ │ Cross-sectional ranks (today's 450 symbols):       │  │
│ │   Each feature → rank → normalize to [-1, +1]      │  │
│ │                                                     │  │
│ │ Output: 450 rows × 28 feature columns              │  │
│ │                                                     │  │
│ │ Time: ~5 seconds (Polars lazy + collect)           │  │
│ └────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
T=7: Chronos Priors Generation (6:02:44 PM)
┌──────────────────────────────────────────────────────────┐
│ Step 7: Run Chronos Teacher Inference                   │
│ ┌────────────────────────────────────────────────────┐  │
│ │ Load checkpoint:                                    │  │
│ │   chronos_teacher_v1_20240114.pt                   │  │
│ │   (Trained last Sunday)                            │  │
│ │                                                     │  │
│ │ Per-symbol inference (batched):                    │  │
│ │   Input: 512-day context ending today              │  │
│ │   Output: Quantile dist for next 5 days            │  │
│ │                                                     │  │
│ │ Batch processing:                                  │  │
│ │   450 symbols / batch_size(32) = 15 batches        │  │
│ │   GPU inference: ~30 seconds total                 │  │
│ │                                                     │  │
│ │ Extract priors (per symbol):                       │  │
│ │   AAPL:  drift=0.02, vol=1.8, tail=0.15, p_up=0.55│  │
│ │   MSFT:  drift=0.01, vol=1.2, tail=0.12, p_up=0.52│  │
│ │   ...                                              │  │
│ │                                                     │  │
│ │ Time: ~30 seconds (GPU bottleneck)                 │  │
│ └────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
T=8: Selector Inference (6:03:14 PM)
┌──────────────────────────────────────────────────────────┐
│ Step 8: Run RankTransformer Inference                   │
│ ┌────────────────────────────────────────────────────┐  │
│ │ Load checkpoint:                                    │  │
│ │   selector_v1_fold3_20240114.pt                    │  │
│ │   (Trained last Sunday, most recent fold)          │  │
│ │                                                     │  │
│ │ Input assembly (per symbol):                       │  │
│ │   features (28 cols) + priors (4 cols) = 32 total │  │
│ │                                                     │  │
│ │ Model forward pass:                                │  │
│ │   Batch: All 450 symbols together                  │  │
│ │   Transformer encoder → Multi-task heads           │  │
│ │   Outputs:                                         │  │
│ │     - quantiles (9 values)                         │  │
│ │     - prob_up (scalar)                             │  │
│ │     - predicted_vol (scalar)                       │  │
│ │                                                     │  │
│ │ Score blending:                                    │  │
│ │   base_score = median(quantiles)                   │  │
│ │   regime_adj = f(market_breadth, rv21_chg)         │  │
│ │   final_score = base_score × regime_adj            │  │
│ │                                                     │  │
│ │ Cross-sectional ranking:                           │  │
│ │   rank(final_score) → percentiles                  │  │
│ │                                                     │  │
│ │ Time: ~2 seconds (GPU)                             │  │
│ └────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
T=9: Portfolio Construction (6:03:16 PM)
┌──────────────────────────────────────────────────────────┐
│ Step 9: Build Target Portfolio                          │
│ ┌────────────────────────────────────────────────────┐  │
│ │ Top-N selection:                                    │  │
│ │   N = 30 (configurable, risk-dependent)            │  │
│ │   Select symbols with score >= P93.3               │  │
│ │                                                     │  │
│ │ Selected symbols (sorted by score):                │  │
│ │   1. NVDA  (score 0.92, rank 1)                    │  │
│ │   2. AAPL  (score 0.89, rank 2)                    │  │
│ │   3. MSFT  (score 0.87, rank 3)                    │  │
│ │   ...                                              │  │
│ │   30. AMD   (score 0.72, rank 30)                  │  │
│ │                                                     │  │
│ │ HRP Allocation:                                    │  │
│ │   Compute covariance matrix (60-day rolling)       │  │
│ │   Hierarchical clustering on correlation           │  │
│ │   Inverse-volatility weights                       │  │
│ │                                                     │  │
│ │ Risk posture adjustment:                           │  │
│ │   Current posture: NORMAL                          │  │
│ │   → No adjustment (100% allocation)                │  │
│ │                                                     │  │
│ │   If CAUTIOUS: 70% allocation, 30% cash            │  │
│ │   If DEFENSIVE: 40% allocation, 60% cash           │  │
│ │                                                     │  │
│ │ Position sizing (account value $100K):             │  │
│ │   NVDA: 0.045 × $100K / $500.2 = 9 shares          │  │
│ │   AAPL: 0.042 × $100K / $185.6 = 22 shares         │  │
│ │   ...                                              │  │
│ │                                                     │  │
│ │ Time: ~3 seconds                                   │  │
│ └────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
T=10: Order Generation (6:03:19 PM)
┌──────────────────────────────────────────────────────────┐
│ Step 10: Compute Rebalancing Orders                     │
│ ┌────────────────────────────────────────────────────┐  │
│ │ Current portfolio (from yesterday):                │  │
│ │   AAPL: 20 shares                                  │  │
│ │   MSFT: 10 shares                                  │  │
│ │   GOOGL: 5 shares  ← Not in today's target        │  │
│ │   ...                                              │  │
│ │                                                     │  │
│ │ Target portfolio (from Step 9):                    │  │
│ │   NVDA: 9 shares   ← New position                 │  │
│ │   AAPL: 22 shares  ← Increase                     │  │
│ │   MSFT: 12 shares  ← Increase                     │  │
│ │   ...                                              │  │
│ │                                                     │  │
│ │ Orders generated (diff):                           │  │
│ │   BUY  NVDA  9 shares  @ MKT                       │  │
│ │   BUY  AAPL  2 shares  @ MKT                       │  │
│ │   BUY  MSFT  2 shares  @ MKT                       │  │
│ │   SELL GOOGL 5 shares  @ MKT                       │  │
│ │   ...                                              │  │
│ │                                                     │  │
│ │ Total: 45 orders (15 buys, 30 sells)               │  │
│ │                                                     │  │
│ │ Time: <1 second                                    │  │
│ └────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
T=11: Pre-Market Queue (6:03:20 PM → Next Day 9:30 AM)
┌──────────────────────────────────────────────────────────┐
│ Step 11: Queue Orders for Pre-Market Execution          │
│ ┌────────────────────────────────────────────────────┐  │
│ │ Write orders to queue:                             │  │
│ │   /artifacts/orders/pending_20240116.json          │  │
│ │                                                     │  │
│ │ Human review window: 6 PM → 9:30 AM (15.5 hours)   │  │
│ │   - Review order list                              │  │
│ │   - Check portfolio allocation                     │  │
│ │   - Approve or modify                              │  │
│ │                                                     │  │
│ │ Next morning (9:30 AM ET):                         │  │
│ │   Execution engine submits orders to broker        │  │
│ │   (Alpaca API, Interactive Brokers, etc.)          │  │
│ │                                                     │  │
│ │ Or in backtest mode:                               │  │
│ │   Simulate fills at next_open price                │  │
│ │   Apply slippage (configurable bps)                │  │
│ └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘

TOTAL PIPELINE TIME: ~3.5 minutes (dominated by Chronos GPU inference)
```

**Concurrency Optimizations**:
- Steps 1-2: Parallel per-symbol (20 concurrent)
- Step 4A-4B: Parallel computation (breadth + covariates)
- Step 7: Batched GPU inference (32 symbols/batch)
- Step 8: Single-batch inference (all symbols together)

---

## 4. STATE MACHINES & WORKFLOWS

### 4.1 Risk Posture State Machine

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       RISK POSTURE STATE MACHINE                         │
│                   (Dynamic Portfolio Exposure Adjustment)                │
└─────────────────────────────────────────────────────────────────────────┘

STATES (3):
  ┌─────────────┐
  │   NORMAL    │  Full exposure (100% allocation)
  └─────────────┘
  ┌─────────────┐
  │  CAUTIOUS   │  Reduced exposure (70% allocation, 30% cash)
  └─────────────┘
  ┌─────────────┐
  │  DEFENSIVE  │  Minimal exposure (40% allocation, 60% cash)
  └─────────────┘

TRANSITION LOGIC:
┌────────────────────────────────────────────────────────────────────┐
│ Evaluated DAILY after market enrichment data available            │
└────────────────────────────────────────────────────────────────────┘

Input Signals:
┌──────────────────────────────────────────────────────────┐
│ 1. Market Breadth Indicators                             │
│    - market_breadth_ad (Advance/Decline)                 │
│    - market_breadth_bpi_21d (Buying Pressure)            │
│                                                           │
│ 2. Realized Volatility                                   │
│    - rv21_level (21-day rolling SPY volatility)          │
│    - rv21_chg_1d (1-day change in volatility)            │
│                                                           │
│ 3. Crash Override Signals                                │
│    - max_drawdown_5d (max 5-day SPY drawdown)            │
│    - vix_spike (VIX daily change)                        │
└──────────────────────────────────────────────────────────┘

STATE DIAGRAM:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│         ┌──────────────┐                                                │
│    ┌───►│   NORMAL     │◄────┐                                          │
│    │    │ (100% alloc) │     │                                          │
│    │    └──────┬───────┘     │                                          │
│    │           │             │                                          │
│    │  [Recovery]        [Stabilization]                                │
│    │    rv21 < 15%      breadth > 0.20                                 │
│    │    breadth > 0.15  bpi > 0.50                                     │
│    │           │             │                                          │
│    │           ▼             │                                          │
│    │    ┌──────────────┐    │                                          │
│    │    │  CAUTIOUS    │◄───┘                                          │
│    │    │ (70% alloc)  │                                               │
│    │    └──────┬───────┘                                               │
│    │           │                                                        │
│    │  [Warning Signs]                                                  │
│    │    rv21 > 20%                                                     │
│    │    OR breadth < 0.0                                               │
│    │    OR bpi < 0.40                                                  │
│    │           │                                                        │
│    │           ▼                                                        │
│    │    ┌──────────────┐                                               │
│    └────│  DEFENSIVE   │                                               │
│         │ (40% alloc)  │                                               │
│         └──────────────┘                                               │
│                ▲                                                        │
│                │                                                        │
│         [Crash Override]                                               │
│          max_dd_5d < -10%                                              │
│          OR vix_spike > 30%                                            │
│          (Direct transition from any state)                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

TRANSITION TABLE:
┌────────────┬─────────────────────────────┬──────────────┐
│ From State │ Condition                   │ To State     │
├────────────┼─────────────────────────────┼──────────────┤
│ NORMAL     │ rv21 > 20%                  │ CAUTIOUS     │
│ NORMAL     │ breadth < 0.0               │ CAUTIOUS     │
│ NORMAL     │ bpi < 0.40                  │ CAUTIOUS     │
│ NORMAL     │ max_dd_5d < -10%            │ DEFENSIVE    │
├────────────┼─────────────────────────────┼──────────────┤
│ CAUTIOUS   │ rv21 > 25%                  │ DEFENSIVE    │
│ CAUTIOUS   │ breadth < -0.15             │ DEFENSIVE    │
│ CAUTIOUS   │ max_dd_5d < -10%            │ DEFENSIVE    │
│ CAUTIOUS   │ rv21 < 15% AND breadth>0.15 │ NORMAL       │
├────────────┼─────────────────────────────┼──────────────┤
│ DEFENSIVE  │ breadth > 0.20 AND bpi>0.50 │ CAUTIOUS     │
│ DEFENSIVE  │ (hold for min 5 days)       │ CAUTIOUS     │
└────────────┴─────────────────────────────┴──────────────┘

EXAMPLE SCENARIO: 2024 Market Turbulence
┌──────────────────────────────────────────────────────────────────┐
│ Date       │ State     │ Signals          │ Transition         │
├────────────┼───────────┼──────────────────┼────────────────────┤
│ 2024-01-10 │ NORMAL    │ rv21=14%, ad=0.25│ Hold NORMAL        │
│ 2024-01-11 │ NORMAL    │ rv21=16%, ad=0.10│ Hold NORMAL        │
│ 2024-01-12 │ NORMAL    │ rv21=22%, ad=-0.05│ → CAUTIOUS (vol) │
│ 2024-01-15 │ CAUTIOUS  │ rv21=24%, ad=-0.10│ Hold CAUTIOUS     │
│ 2024-01-16 │ CAUTIOUS  │ rv21=28%, ad=-0.20│ → DEFENSIVE       │
│ 2024-01-17 │ DEFENSIVE │ rv21=30%, ad=-0.25│ Hold DEFENSIVE    │
│ 2024-01-18 │ DEFENSIVE │ rv21=26%, ad=-0.15│ Hold (min 5 days) │
│ 2024-01-22 │ DEFENSIVE │ rv21=20%, ad=0.15 │ Hold (min 5 days) │
│ 2024-01-23 │ DEFENSIVE │ rv21=18%, ad=0.22 │ → CAUTIOUS (recover)│
│ 2024-01-25 │ CAUTIOUS  │ rv21=15%, ad=0.28, bpi=0.55│ → NORMAL │
└──────────────────────────────────────────────────────────────────┘

PORTFOLIO IMPACT:
┌────────────┬───────────────┬──────────────────────────────────┐
│ State      │ Allocation    │ Example ($100K account)          │
├────────────┼───────────────┼──────────────────────────────────┤
│ NORMAL     │ 100%          │ $100K in stocks, $0 cash         │
│ CAUTIOUS   │ 70%           │ $70K in stocks, $30K cash        │
│ DEFENSIVE  │ 40%           │ $40K in stocks, $60K cash        │
└────────────┴───────────────┴──────────────────────────────────┘

IMPLEMENTATION:
┌──────────────────────────────────────────────────────────────────┐
│ def get_risk_posture(market_data: pd.DataFrame) -> str:         │
│     latest = market_data.iloc[-1]                                │
│                                                                   │
│     # Crash override (immediate DEFENSIVE)                       │
│     if latest["max_drawdown_5d"] < -0.10:                       │
│         return "DEFENSIVE"                                       │
│                                                                   │
│     rv21 = latest["rv21_level"]                                 │
│     breadth = latest["market_breadth_ad"]                       │
│     bpi = latest["market_breadth_bpi_21d"]                      │
│                                                                   │
│     # State transitions                                          │
│     if current_state == "NORMAL":                                │
│         if rv21 > 0.20 or breadth < 0.0 or bpi < 0.40:          │
│             return "CAUTIOUS"                                    │
│                                                                   │
│     elif current_state == "CAUTIOUS":                            │
│         if rv21 > 0.25 or breadth < -0.15:                      │
│             return "DEFENSIVE"                                   │
│         if rv21 < 0.15 and breadth > 0.15:                      │
│             return "NORMAL"                                      │
│                                                                   │
│     elif current_state == "DEFENSIVE":                           │
│         if days_in_defensive >= 5:                               │
│             if breadth > 0.20 and bpi > 0.50:                   │
│                 return "CAUTIOUS"                                │
│                                                                   │
│     return current_state  # No transition                        │
└──────────────────────────────────────────────────────────────────┘
```

**Key Properties**:
- **Asymmetric Transitions**: Easier to reduce risk (1 day) than increase (5 day minimum in DEFENSIVE)
- **Crash Override**: Immediate DEFENSIVE on severe drawdown (no gradual steps)
- **Hysteresis**: Different thresholds for up/down transitions (prevents flapping)

---

### 4.2 Artifact Lifecycle & Versioning Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   ARTIFACT VERSIONING & LIFECYCLE                        │
│                 (Reproducible, Immutable, Traceable)                     │
└─────────────────────────────────────────────────────────────────────────┘

ARTIFACT TYPES & STORAGE HIERARCHY:
┌──────────────────────────────────────────────────────────────────┐
│ /artifacts/                                                       │
│ ├── canonical/                  [Raw validated OHLCV]            │
│ │   ├── AAPL.parquet                                             │
│ │   ├── MSFT.parquet                                             │
│ │   └── ...                                                      │
│ │                                                                 │
│ ├── eligibility/                [Universe definitions]           │
│ │   ├── universe_20240115.parquet                                │
│ │   └── universe_latest.parquet  ← Symlink                       │
│ │                                                                 │
│ ├── market/                     [Breadth + Covariates]           │
│ │   ├── breadth_20240115.parquet                                 │
│ │   ├── covariates_20240115.parquet                              │
│ │   └── market_frame_20240115.parquet                            │
│ │                                                                 │
│ ├── features/                   [Engineered features]            │
│ │   ├── selector_features_v1_abc123.parquet                      │
│ │   └── schema_contract_v1.json                                  │
│ │                                                                 │
│ ├── priors/                     [Chronos teacher outputs]        │
│ │   ├── chronos_priors_v2_20240115.parquet                       │
│ │   └── prior_schema_v2.json                                     │
│ │                                                                 │
│ ├── models/                     [Trained checkpoints]            │
│ │   ├── chronos_teacher_v1_lora/                                 │
│ │   │   ├── adapter_config.json                                  │
│ │   │   ├── adapter_model.bin  (~8MB LoRA weights)               │
│ │   │   └── training_args.json                                   │
│ │   ├── selector_v1_fold3_def456.pt                              │
│ │   ├── scaler_v1_def456.pkl                                     │
│ │   └── calibrator_v1_def456.pkl                                 │
│ │                                                                 │
│ ├── scores/                     [Daily inference outputs]        │
│ │   ├── 2024-01-15_scores.parquet                                │
│ │   └── leaderboard_latest.parquet                               │
│ │                                                                 │
│ └── registry/                   [Metadata & provenance]          │
│     ├── artifact_records.db  (SQLite)                            │
│     ├── version_hashes.json                                      │
│     └── lineage_graph.json                                       │
└──────────────────────────────────────────────────────────────────┘

VERSIONING SCHEME:
┌──────────────────────────────────────────────────────────────────┐
│ Artifact Naming Convention:                                      │
│   <artifact_type>_<version>_<hash>.<ext>                        │
│                                                                   │
│ Examples:                                                         │
│   selector_features_v1_abc123.parquet                            │
│   ├─┬─────────────┬──┬──┬─────┬──┬──────                        │
│   │ │             │  │  │     │  └─ Extension                    │
│   │ │             │  │  │     └──── 12-char SHA256 truncation    │
│   │             │  │  └────────── Semantic version (v1, v2, ...) │
│   │             └──┴──────────── Artifact type                   │
│   └────────────────────────────── Name prefix                    │
│                                                                   │
│ Hash computation (stable):                                       │
│   payload = {                                                    │
│       "feature_config": {...},                                   │
│       "dependencies": ["canonical_v1", "market_v1"],            │
│       "code_version": "git:abc123",                             │
│       "created_at": "2024-01-15T18:00:00Z"                      │
│   }                                                              │
│   hash = SHA256(json.dumps(payload, sort_keys=True))[:12]       │
└──────────────────────────────────────────────────────────────────┘

ARTIFACT LIFECYCLE STATES:
┌──────────────────────────────────────────────────────────────────┐
│                                                                   │
│   ┌──────────┐                                                   │
│   │  DRAFT   │  Created, not validated                           │
│   └─────┬────┘                                                   │
│         │                                                         │
│         │ validate_artifact()                                    │
│         ▼                                                         │
│   ┌──────────┐                                                   │
│   │ VALIDATED│  Passed all schema/semantic checks                │
│   └─────┬────┘                                                   │
│         │                                                         │
│         │ register_artifact()                                    │
│         ▼                                                         │
│   ┌──────────┐                                                   │
│   │REGISTERED│  Entered into artifact registry DB                │
│   └─────┬────┘                                                   │
│         │                                                         │
│         │ promote_to_prod()                                      │
│         ▼                                                         │
│   ┌──────────┐                                                   │
│   │  PROD    │  Active in production pipelines                   │
│   └─────┬────┘                                                   │
│         │                                                         │
│         │ deprecate_artifact() [after new version promoted]      │
│         ▼                                                         │
│   ┌──────────┐                                                   │
│   │DEPRECATED│  Superseded, kept for lineage                     │
│   └─────┬────┘                                                   │
│         │                                                         │
│         │ archive_artifact() [after 90 days]                     │
│         ▼                                                         │
│   ┌──────────┐                                                   │
│   │ ARCHIVED │  Moved to cold storage, metadata retained         │
│   └──────────┘                                                   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘

DEPENDENCY GRAPH EXAMPLE:
┌──────────────────────────────────────────────────────────────────┐
│ Date: 2024-01-15 Nightly Run                                     │
│                                                                   │
│                     Canonical OHLCV                               │
│                     (500 symbols)                                 │
│                           │                                       │
│           ┌───────────────┼───────────────┐                       │
│           │               │               │                       │
│           ▼               ▼               ▼                       │
│      Universe      Market Breadth   Market Covariates            │
│     (450 trad)       (AD, BPI)      (SPY, QQQ, RV21)             │
│           │               │               │                       │
│           └───────────────┴───────────────┘                       │
│                           │                                       │
│                           ▼                                       │
│                     MarketFrame                                   │
│                     (joined panel)                                │
│                           │                                       │
│                           ▼                                       │
│                  Feature Builder                                  │
│                selector_features_v1_abc123                        │
│                           │                                       │
│           ┌───────────────┴───────────────┐                       │
│           │                               │                       │
│           ▼                               ▼                       │
│    Chronos Teacher                 Selector Training              │
│  chronos_priors_v2_def456         selector_v1_fold3_ghi789       │
│           │                               │                       │
│           └───────────────┬───────────────┘                       │
│                           │                                       │
│                           ▼                                       │
│                  Selector Inference                               │
│                  scores_20240115.parquet                          │
│                           │                                       │
│                  ┌────────┴────────┐                              │
│                  │ Provenance:     │                              │
│                  │ - features: abc123                             │
│                  │ - priors: def456                               │
│                  │ - model: ghi789                                │
│                  └─────────────────┘                              │
└──────────────────────────────────────────────────────────────────┘

ARTIFACT RECORD (SQLite schema):
┌──────────────────────────────────────────────────────────────────┐
│ CREATE TABLE artifact_records (                                  │
│     id INTEGER PRIMARY KEY,                                      │
│     artifact_path TEXT NOT NULL UNIQUE,                          │
│     artifact_type TEXT NOT NULL,                                 │
│     version TEXT NOT NULL,                                       │
│     hash TEXT NOT NULL,                                          │
│     state TEXT NOT NULL,  -- DRAFT/VALIDATED/REGISTERED/PROD/... │
│     created_at TIMESTAMP NOT NULL,                               │
│     validated_at TIMESTAMP,                                      │
│     promoted_at TIMESTAMP,                                       │
│     deprecated_at TIMESTAMP,                                     │
│     config_json TEXT,  -- Full config used to create             │
│     dependencies_json TEXT,  -- Parent artifact hashes           │
│     metadata_json TEXT  -- Additional provenance                 │
│ );                                                               │
│                                                                   │
│ Example row:                                                      │
│ {                                                                 │
│     "artifact_path": "features/selector_features_v1_abc123.parquet",│
│     "artifact_type": "selector_features",                        │
│     "version": "v1",                                             │
│     "hash": "abc123",                                            │
│     "state": "PROD",                                             │
│     "created_at": "2024-01-15T18:05:00Z",                        │
│     "promoted_at": "2024-01-15T18:30:00Z",                       │
│     "config_json": "{\"raw_features\": [...], \"norm_method\": \"rank\"}",│
│     "dependencies_json": "{\"canonical\": \"base_v1\", \"market\": \"v2_xyz\"}",│
│     "metadata_json": "{\"git_commit\": \"abc123\", \"user\": \"system\"}"│
│ }                                                                 │
└──────────────────────────────────────────────────────────────────┘

PROMOTION WORKFLOW:
┌──────────────────────────────────────────────────────────────────┐
│ Step 1: Create new artifact                                      │
│   feature_builder = SelectorFeatureBuilder(config_v2)            │
│   df = feature_builder.build(market_frame)                       │
│   path = save_artifact(df, state="DRAFT")                        │
│   → features/selector_features_v2_xyz789.parquet                 │
│                                                                   │
│ Step 2: Validation                                               │
│   issues = validate_artifact(path, contract="selector_v2")       │
│   if issues:                                                      │
│       raise ValidationError(issues)                              │
│   update_state(path, "VALIDATED")                                │
│                                                                   │
│ Step 3: Registration                                             │
│   record = ArtifactRecord(                                       │
│       path=path,                                                 │
│       config=config_v2.to_dict(),                                │
│       dependencies={"market": "v2_xyz"}                          │
│   )                                                              │
│   registry.insert(record)                                        │
│   update_state(path, "REGISTERED")                               │
│                                                                   │
│ Step 4: Promotion to PROD                                        │
│   # Deprecate old PROD version                                   │
│   old_prod = registry.get_prod("selector_features")             │
│   update_state(old_prod.path, "DEPRECATED")                      │
│                                                                   │
│   # Promote new version                                          │
│   update_state(path, "PROD")                                     │
│   update_symlink("features/selector_features_latest.parquet", path)│
│                                                                   │
│ Step 5: Downstream cascade                                       │
│   # All downstream artifacts (priors, scores) now stale          │
│   # Trigger rebuild or mark for rebuild                          │
│   mark_stale(["priors", "scores"], reason="features updated")    │
└──────────────────────────────────────────────────────────────────┘
```

**Immutability Guarantee**:
- Once an artifact is REGISTERED, its content is immutable
- Hash ensures bit-level reproducibility
- Modifications create NEW artifacts with new hashes
- Old artifacts remain in registry for lineage tracing

---

## 5. TRAINING VS INFERENCE COMPARISON

### 5.1 Side-by-Side Workflow Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│              TRAINING MODE vs INFERENCE MODE COMPARISON                  │
└─────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────┬───────────────────────────────────┐
│         TRAINING MODE                 │       INFERENCE MODE              │
│        (Weekly/Monthly)               │         (Nightly)                 │
├───────────────────────────────────────┼───────────────────────────────────┤
│                                       │                                   │
│ OBJECTIVE:                            │ OBJECTIVE:                        │
│ ├─ Learn parameters from historical  │ ├─ Generate predictions for       │
│ │  data                               │ │  tomorrow's trading              │
│ └─ Validate generalization            │ └─ Score current universe         │
│                                       │                                   │
│ DATA SCOPE:                           │ DATA SCOPE:                       │
│ ├─ Full historical dataset            │ ├─ Latest window only             │
│ │  (2020-01-01 → 2024-12-31)          │ │  (context: last 512 days)       │
│ │  ~1000 trading days                 │ │  (inference: today only)        │
│ └─ All symbols in universe            │ └─ Current tradable universe      │
│    (500 symbols)                      │    (450 symbols today)            │
│                                       │                                   │
│ SPLITS:                               │ SPLITS:                           │
│ ┌─────────────────────────────────┐   │ ┌─────────────────────────────┐   │
│ │ Walk-Forward Cross-Validation   │   │ │ No splits                   │   │
│ │                                 │   │ │ (use full trained model)    │   │
│ │ Fold 1:                         │   │ └─────────────────────────────┘   │
│ │   Train: 2020-01 → 2021-12      │   │                                   │
│ │   Test:  2022-01 → 2022-06      │   │                                   │
│ │ Fold 2:                         │   │                                   │
│ │   Train: 2020-07 → 2022-06      │   │                                   │
│ │   Test:  2022-07 → 2022-12      │   │                                   │
│ │ Fold 3:                         │   │                                   │
│ │   Train: 2021-01 → 2022-12      │   │                                   │
│ │   Test:  2023-01 → 2023-06      │   │                                   │
│ │ ...                             │   │                                   │
│ └─────────────────────────────────┘   │                                   │
│                                       │                                   │
│ FEATURE ENGINEERING:                  │ FEATURE ENGINEERING:              │
│ ├─ Compute ALL features for all dates│ ├─ Compute features for today only│
│ │  (warm-up handled by dropping NaNs) │ │  (use cached historical stats)  │
│ ├─ Cross-sectional ranks per date     │ ├─ Cross-sectional ranks for      │
│ │  (450 symbols/date × 800 dates)     │ │  today's 450 symbols            │
│ └─ Target labels: y_rank, y_trade     │ └─ No target labels (predicting   │
│    (use fwd_return_5d)                │    future, unknown)               │
│                                       │                                   │
│ CHRONOS TEACHER:                      │ CHRONOS TEACHER:                  │
│ ┌─────────────────────────────────┐   │ ┌─────────────────────────────┐   │
│ │ Training Loop:                  │   │ │ Inference Pass:             │   │
│ │                                 │   │ │                             │   │
│ │ for fold in folds:              │   │ │ Load checkpoint:            │   │
│ │   for batch in dataset:         │   │ │   chronos_teacher_v1.pt     │   │
│ │     context = batch[:512]       │   │ │                             │   │
│ │     horizons = batch[512:533]   │   │ │ Per symbol:                 │   │
│ │     outputs = model(context)    │   │ │   context = last_512_days   │   │
│ │     loss = NLL(outputs, horizons)│  │ │   priors = model(context)   │   │
│ │     loss.backward()             │   │ │                             │   │
│ │     optimizer.step()            │   │ │ Output: drift, vol,         │   │
│ │                                 │   │ │   tail_risk, prob_up        │   │
│ │ Save checkpoint every epoch     │   │ │   (4 scalars per symbol)    │   │
│ └─────────────────────────────────┘   │ └─────────────────────────────┘   │
│                                       │                                   │
│ Duration: ~10 min (GPU, 10 epochs)    │ Duration: ~30 sec (GPU, 450 sym)  │
│                                       │                                   │
│ SELECTOR MODEL:                       │ SELECTOR MODEL:                   │
│ ┌─────────────────────────────────┐   │ ┌─────────────────────────────┐   │
│ │ Training Loop:                  │   │ │ Inference Pass:             │   │
│ │                                 │   │ │                             │   │
│ │ for fold in folds:              │   │ │ Load checkpoint:            │   │
│ │   train_data, test_data = split │   │ │   selector_v1_fold3.pt      │   │
│ │                                 │   │ │                             │   │
│ │   for epoch in range(20):       │   │ │ Input assembly:             │   │
│ │     for date_batch in train:    │   │ │   features (28 cols) +      │   │
│ │       X = features + priors     │   │ │   priors (4 cols) = 32      │   │
│ │       y_true = y_rank, y_trade  │   │ │                             │   │
│ │       outputs = model(X)        │   │ │ Forward pass (batched):     │   │
│ │       loss = ranking_loss(      │   │ │   scores = model(X)         │   │
│ │           outputs, y_true)      │   │ │                             │   │
│ │       loss.backward()           │   │ │ Rank scores:                │   │
│ │       optimizer.step()          │   │ │   percentiles = rank(scores)│   │
│ │                                 │   │ │                             │   │
│ │   # Evaluate on test fold       │   │ │ Output: score + percentile  │   │
│ │   metrics = evaluate(test_data) │   │ │   (450 rows × 2 cols)       │   │
│ │                                 │   │ └─────────────────────────────┘   │
│ │ Save best checkpoint per fold   │   │                                   │
│ └─────────────────────────────────┘   │ Duration: ~2 sec (GPU)            │
│                                       │                                   │
│ Duration: ~15 min per fold × 5 folds  │                                   │
│         = ~75 min total               │                                   │
│                                       │                                   │
│ VALIDATION:                           │ VALIDATION:                       │
│ ├─ Out-of-sample test performance     │ ├─ Schema contract check          │
│ │  (IC, Sharpe, hit rate)             │ │  (ensure feature alignment)     │
│ ├─ Overfitting checks                 │ ├─ Sanity checks (no NaNs/Infs)   │
│ │  (train vs test gap)                │ ├─ Distribution drift detection   │
│ └─ Ablation studies                   │ │  (compare to historical)        │
│    (feature importance)               │ └─ Model version verification     │
│                                       │    (correct checkpoint loaded)    │
│                                       │                                   │
│ OUTPUT ARTIFACTS:                     │ OUTPUT ARTIFACTS:                 │
│ ├─ Model checkpoints (5 folds)        │ ├─ Scores parquet                 │
│ │  selector_v1_fold{1-5}.pt           │ │  scores_20240115.parquet        │
│ ├─ Scaler fitted on train data        │ ├─ Leaderboard (top N)            │
│ │  scaler_v1.pkl                      │ │  leaderboard_latest.parquet     │
│ ├─ Calibrator for probability         │ └─ Provenance metadata            │
│ │  calibrator_v1.pkl                  │    (feature/prior/model versions) │
│ ├─ Training metrics & logs            │                                   │
│ │  training_metrics_v1.json           │                                   │
│ └─ Validation reports                 │                                   │
│    validation_report_v1.html          │                                   │
│                                       │                                   │
│ FREQUENCY:                            │ FREQUENCY:                        │
│ ├─ Chronos: Monthly/Quarterly         │ ├─ Nightly (every trading day)    │
│ └─ Selector: Weekly                   │ │  at 6 PM ET after market close  │
│    (or when features change)          │ └─ Takes ~3.5 min total           │
│                                       │                                   │
│ COMPUTATIONAL COST:                   │ COMPUTATIONAL COST:               │
│ ├─ GPU required (A100/V100)           │ ├─ GPU recommended (can use CPU)  │
│ ├─ ~10 min Chronos + ~75 min Selector │ ├─ ~30 sec Chronos + ~2 sec Selector│
│ └─ Total: ~85 min                     │ └─ Total: ~32 sec                 │
│                                       │                                   │
└───────────────────────────────────────┴───────────────────────────────────┘
```

**Key Differences**:
1. **Targets**: Training has ground truth labels (fwd returns), inference predicts unknown future
2. **Splits**: Training uses walk-forward CV, inference uses full trained model
3. **Scope**: Training processes historical panel, inference processes single date
4. **Validation**: Training focuses on generalization, inference on distribution consistency
5. **Frequency**: Training weekly/monthly, inference nightly

---

### 5.2 Data Leakage Prevention Mechanisms

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   DATA LEAKAGE PREVENTION ARCHITECTURE                   │
└─────────────────────────────────────────────────────────────────────────┘

PRINCIPLE: Information at time t can only use data ≤ t

ENFORCEMENT LAYERS:

LAYER 1: Trading Calendar Enforcement
┌──────────────────────────────────────────────────────────────────┐
│ ✓ CORRECT: Trading-day aware forward returns                     │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ def compute_forward_return(df, horizon=5):             │      │
│ │     # Use trading calendar, not calendar days          │      │
│ │     calendar = get_nyse_calendar()                     │      │
│ │     fwd_dates = df["date"].map(                        │      │
│ │         lambda d: calendar.shift_trading_days(d, horizon)│    │
│ │     )                                                   │      │
│ │     # Join on exact shifted dates                      │      │
│ │     fwd = df.set_index("date")["close"].reindex(fwd_dates)│  │
│ │     return log(fwd / df["close"])                      │      │
│ └────────────────────────────────────────────────────────┘      │
│                                                                   │
│ ✗ INCORRECT: Naive shift (leaks on holidays/weekends)            │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ # BAD: Uses calendar days, not trading days            │      │
│ │ fwd_return = log(df["close"].shift(-5) / df["close"])  │      │
│ │ # Problem: shift(-5) can be 3-7 trading days depending │      │
│ │ # on weekends/holidays → inconsistent horizons         │      │
│ └────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘

LAYER 2: Purged Walk-Forward Splits
┌──────────────────────────────────────────────────────────────────┐
│ Standard K-Fold (LEAKY):                                          │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ Train: ████████████                                    │      │
│ │ Test:              ████ ← Gap needed!                  │      │
│ │        ↑           ↑                                    │      │
│ │        │           └─ Test observation at t             │      │
│ │        └─ Training observation at t-5                   │      │
│ │           used fwd_return_5d (looks to t)              │      │
│ │           → Leakage!                                    │      │
│ └────────────────────────────────────────────────────────┘      │
│                                                                   │
│ Purged Walk-Forward (CORRECT):                                   │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ Fold 1:                                                │      │
│ │   Train: ████████████ [embargo] ────                   │      │
│ │   Test:                         ────███                │      │
│ │                      ↑          ↑                       │      │
│ │                      │          └─ Test starts here     │      │
│ │                      └─ Last train observation          │      │
│ │                         (no fwd return computed         │      │
│ │                          past this point)               │      │
│ │                                                         │      │
│ │ Embargo period: H_sel + 1 trading days                 │      │
│ │   (If H_sel = 5, embargo = 6 days)                     │      │
│ │                                                         │      │
│ │ Fold 2:                                                │      │
│ │   Train: ████████████████████ [embargo] ──             │      │
│ │   Test:                               ──███            │      │
│ │                                                         │      │
│ │ Implementation:                                        │      │
│ │   train_end = fold_split_date                          │      │
│ │   test_start = shift_trading_days(                     │      │
│ │       train_end, H_sel + 1                             │      │
│ │   )                                                    │      │
│ └────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘

LAYER 3: Feature Computation Causality
┌──────────────────────────────────────────────────────────────────┐
│ ✓ CORRECT: Lagged features only                                  │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ # Rolling statistics look backward                     │      │
│ │ volatility_20d = (                                     │      │
│ │     df.sort("date")                                    │      │
│ │     .with_columns(                                     │      │
│ │         pl.col("log_return")                           │      │
│ │         .rolling_std(window_size=20)                   │      │
│ │         .over("symbol")                                │      │
│ │         .alias("volatility_20d")                       │      │
│ │     )                                                   │      │
│ │ )                                                      │      │
│ │ # At date t, volatility_20d uses [t-19, ..., t]        │      │
│ └────────────────────────────────────────────────────────┘      │
│                                                                   │
│ ✗ INCORRECT: Forward-looking features                            │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ # BAD: Rolling mean looks forward                      │      │
│ │ next_week_avg_return = (                               │      │
│ │     df["log_return"]                                   │      │
│ │     .rolling_mean(window_size=5, center=True)          │      │
│ │     # center=True uses ±2 days → leaks future          │      │
│ │ )                                                      │      │
│ └────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘

LAYER 4: Model Fitting Isolation
┌──────────────────────────────────────────────────────────────────┐
│ ✓ CORRECT: Fit on train, transform on test                       │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ # Scaler fitted ONLY on training fold                  │      │
│ │ scaler = StandardScaler()                              │      │
│ │ scaler.fit(X_train)  # Learn mean, std from train      │      │
│ │                                                         │      │
│ │ X_train_scaled = scaler.transform(X_train)             │      │
│ │ X_test_scaled = scaler.transform(X_test)               │      │
│ │ # Test uses train statistics (no leakage)              │      │
│ └────────────────────────────────────────────────────────┘      │
│                                                                   │
│ ✗ INCORRECT: Fit on full dataset                                 │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ # BAD: Scaler sees test data statistics                │      │
│ │ scaler.fit(X_full)  # Includes test fold → leakage     │      │
│ │ X_train_scaled = scaler.transform(X_train)             │      │
│ │ X_test_scaled = scaler.transform(X_test)               │      │
│ └────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘

LAYER 5: Target Construction
┌──────────────────────────────────────────────────────────────────┐
│ ✓ CORRECT: Trading-day aware shift with alignment check          │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ def build_targets(df, horizon=5):                      │      │
│ │     calendar = get_nyse_calendar()                     │      │
│ │                                                         │      │
│ │     # Compute future date for each row                 │      │
│ │     df = df.with_columns(                              │      │
│ │         pl.col("date").map_elements(                   │      │
│ │             lambda d: calendar.shift_trading_days(d, horizon),│
│ │             return_dtype=pl.Date                       │      │
│ │         ).alias("target_date")                         │      │
│ │     )                                                   │      │
│ │                                                         │      │
│ │     # Self-join on (symbol, target_date)               │      │
│ │     targets = df.join(                                 │      │
│ │         df.select(["symbol", "date", "close"]),        │      │
│ │         left_on=["symbol", "target_date"],             │      │
│ │         right_on=["symbol", "date"],                   │      │
│ │         how="left"                                     │      │
│ │     ).with_columns(                                    │      │
│ │         (pl.col("close_right") / pl.col("close")).log()│      │
│ │         .alias("fwd_return_5d")                        │      │
│ │     )                                                   │      │
│ │                                                         │      │
│ │     # Drop rows where target unavailable (end of data) │      │
│ │     return targets.filter(pl.col("fwd_return_5d").is_not_null())│
│ └────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘

VERIFICATION CHECKLIST:
┌──────────────────────────────────────────────────────────────────┐
│ Before promoting features/models to production:                  │
│                                                                   │
│ ☐ All forward returns use trading calendar (not .shift())        │
│ ☐ Walk-forward splits have embargo period ≥ H_sel + 1            │
│ ☐ Scalers/preprocessors fitted only on training folds            │
│ ☐ No rolling operations with center=True                         │
│ ☐ Market breadth uses only symbols tradable at time t            │
│ ☐ Universe eligibility updated daily (no look-ahead)             │
│ ☐ Model checkpoints saved per fold (not across folds)            │
│ ☐ Validation metrics computed only on purged test sets           │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. TIME-SERIES CAUSALITY & WINDOWING

### 6.1 Context Window & Horizon Alignment

```
┌─────────────────────────────────────────────────────────────────────────┐
│                TIME-SERIES WINDOWS & CAUSALITY DIAGRAM                   │
│                     (Chronos Teacher Example)                            │
└─────────────────────────────────────────────────────────────────────────┘

TIMELINE VISUALIZATION (AAPL, 2024-01-15 inference):
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│  Historical Context Window          │  Forecast Horizon                  │
│  (512 trading days)                 │  (21 trading days)                 │
│                                     │                                    │
│  2022-06-01 ─────────────────────▶ 2024-01-15 ─────────▶ 2024-02-13    │
│  ├────────────────────────────────┤ ▲ ├──────────────────────┤          │
│  │                                │ │ │                      │          │
│  │  Context: 512 days of OHLCV    │ │ │  Horizon: 21 days    │          │
│  │  - Daily returns               │ │ │  - Predict distribution         │
│  │  - Volatility                  │ │ │  - Not single point!            │
│  │  - Volume                      │ │ │                      │          │
│  │                                │ │ │                      │          │
│  └────────────────────────────────┘ │ └──────────────────────┘          │
│                                     │                                    │
│                               TODAY (inference date)                     │
│                               └─ Causality boundary                      │
│                                  (no future information)                 │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘

CONTEXT WINDOW COMPOSITION (512 days ending 2024-01-15):
┌──────────────────────────────────────────────────────────────────┐
│ Date Range: 2022-06-01 → 2024-01-15                              │
│                                                                   │
│ Features extracted (per day):                                    │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ Sequence position: t ∈ [1, 512]                        │     │
│ │                                                          │     │
│ │ t=1   (2022-06-01):                                     │     │
│ │   log_return = log(close_t / close_{t-1}) = 0.015      │     │
│ │   high_low_spread = (high - low) / close = 0.025       │     │
│ │   volume_change = log(volume_t / volume_{t-1})         │     │
│ │                                                          │     │
│ │ t=2   (2022-06-02):                                     │     │
│ │   log_return = -0.008                                   │     │
│ │   ...                                                    │     │
│ │                                                          │     │
│ │ t=512 (2024-01-15):                                     │     │
│ │   log_return = 0.012                                    │     │
│ │   high_low_spread = 0.018                               │     │
│ │   volume_change = 0.05                                  │     │
│ └─────────────────────────────────────────────────────────┘     │
│                                                                   │
│ Shape: [512, 5] (512 days × 5 features)                          │
│ Normalization: Standardized (mean=0, std=1) over context         │
└──────────────────────────────────────────────────────────────────┘

HORIZON WINDOW STRUCTURE (21 days forward):
┌──────────────────────────────────────────────────────────────────┐
│ During Training: Ground truth available                           │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ horizon_idx: h ∈ [1, 21]                               │     │
│ │                                                          │     │
│ │ h=1  (2024-01-16): y_1  = log(close_{t+1} / close_t)   │     │
│ │ h=2  (2024-01-17): y_2  = log(close_{t+2} / close_t)   │     │
│ │ h=3  (2024-01-18): y_3  = log(close_{t+3} / close_t)   │     │
│ │ ...                                                      │     │
│ │ h=21 (2024-02-13): y_21 = log(close_{t+21} / close_t)  │     │
│ │                                                          │     │
│ │ Training Target: [y_1, y_2, ..., y_21]                 │     │
│ │ Loss: NLL over quantile distributions Q(y_h)            │     │
│ └─────────────────────────────────────────────────────────┘     │
│                                                                   │
│ During Inference: Future unknown (predict distribution)          │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ Model outputs: Quantile parameters per horizon h        │     │
│ │                                                          │     │
│ │ For each h ∈ [1, 21]:                                   │     │
│ │   μ_h     = location (median prediction)                │     │
│ │   σ_h     = scale (volatility)
│ │   σ_h     = scale (volatility)                          │     │
│ │   T_h     = temperature (tail heaviness)                │     │
│ │                                                          │     │
│ │ Quantile distribution: Q_τ(y_h) = μ_h + (T_h·σ_h)·z_τ  │     │
│ │   where z_τ is standard quantile at level τ            │     │
│ │                                                          │     │
│ │ Extracted priors (aggregate over horizons 1-5):        │     │
│ │   drift = mean(μ_1, ..., μ_5)                          │     │
│ │   vol   = mean(σ_1, ..., σ_5)                          │     │
│ │   tail_risk = |Q_0.95 - Q_0.05| / Q_0.50               │     │
│ │   prob_up = P(y_5 > 0)                                 │     │
│ └─────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘

CAUSALITY ENFORCEMENT VISUALIZATION:
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│  Past (available)              Now (decision point)    Future (predict)  │
│  ◄─────────────────────────────▶ │ ◄───────────────────────────────────▶│
│                                   │                                      │
│  ┌──────────────────────────┐    │    ┌──────────────────────────┐     │
│  │ Context features         │    │    │ Horizon targets           │     │
│  │ (512 days)               │    │    │ (21 days)                 │     │
│  │                          │    │    │                           │     │
│  │ - OHLCV returns          │    │    │ Training: y_1...y_21      │     │
│  │ - Volatility             │    │    │ Inference: Q(y_1)...Q(y_21)│    │
│  │ - Volume                 │    │    │                           │     │
│  └──────────────────────────┘    │    └──────────────────────────┘     │
│           │                       │              ▲                       │
│           └───── Input to model ──┘              │                       │
│                                                   │                       │
│                                            Model predicts                │
│                                            (no future info used)         │
│                                                                           │
│  ALLOWED information flow:  Past → Model → Future predictions            │
│  FORBIDDEN: Future → Model (would be leakage)                            │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

**Key Invariants**:
1. Context window always ends at `t` (today/inference date)
2. Horizon window starts at `t+1` (tomorrow)
3. No overlap between context and horizon (strict causality)
4. All horizons measured in trading days (NYSE calendar)

---

### 6.2 Selector Model Context Window (Different from Chronos)

```
┌─────────────────────────────────────────────────────────────────────────┐
│             SELECTOR (RankTransformer) CONTEXT WINDOW                    │
│                    (Cross-sectional, not time-series)                    │
└─────────────────────────────────────────────────────────────────────────┘

TIMELINE (2024-01-15 inference):
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│  Historical Features            │  Target Horizon                        │
│  (1 day lookback for most)      │  (5 trading days)                      │
│                                  │                                        │
│  2024-01-10 ────▶ 2024-01-15    │  2024-01-15 ────▶ 2024-01-22          │
│  ├─────────────────────────────┤ ▲ ├─────────────────────────────┤      │
│  │                             │ │ │                             │      │
│  │ Features computed:          │ │ │ Target: fwd_return_5d       │      │
│  │ - log_return_1d (today)     │ │ │ y_rank = rank(fwd_return_5d)│      │
│  │ - log_return_5d (5-day)     │ │ │ y_trade = risk_adj >= P70   │      │
│  │ - volatility_20d (20-day)   │ │ │                             │      │
│  │ - rel_volume (21-day MA)    │ │ │                             │      │
│  │                             │ │ │                             │      │
│  └─────────────────────────────┘ │ └─────────────────────────────┘      │
│                                  │                                        │
│                            TODAY (inference date)                         │
│                            └─ All features computed as of this date       │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘

CROSS-SECTIONAL CONTEXT (450 symbols on 2024-01-15):
┌──────────────────────────────────────────────────────────────────┐
│ Unlike Chronos (time-series), Selector sees cross-section:       │
│                                                                   │
│ Input shape: [N_symbols, N_features]                             │
│            = [450, 32]                                            │
│                                                                   │
│ Features per symbol (all as of TODAY):                           │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ symbol: AAPL                                            │     │
│ │                                                          │     │
│ │ Price features (normalized ranks):                      │     │
│ │   log_return_1d_rank  = 0.45   (rank 203/450)          │     │
│ │   log_return_5d_rank  = 0.67   (rank 301/450)          │     │
│ │   log_return_21d_rank = 0.52   (rank 234/450)          │     │
│ │   volatility_20d_rank = 0.23   (rank 104/450)          │     │
│ │   rel_volume_rank     = 0.15   (rank 68/450)           │     │
│ │   rsi_14_rank         = 0.60   (rank 270/450)          │     │
│ │                                                          │     │
│ │ Market covariates (same for all symbols):               │     │
│ │   spy_ret     = 0.015  (SPY up 1.5% today)              │     │
│ │   qqq_ret     = 0.018  (QQQ up 1.8%)                    │     │
│ │   iwm_ret     = 0.009  (IWM up 0.9%)                    │     │
│ │   ief_ret     = -0.001 (IEF down 0.1%)                  │     │
│ │   rv21_chg_1d = 0.014  (Volatility increasing)          │     │
│ │                                                          │     │
│ │ Breadth indicators (same for all symbols):              │     │
│ │   market_breadth_ad     = 0.24  (Bullish)               │     │
│ │   market_breadth_bpi_21d= 0.67  (Strong momentum)       │     │
│ │                                                          │     │
│ │ Chronos priors (per-symbol from teacher):               │     │
│ │   teacher_drift_20d    = 0.02   (Slight uptrend)        │     │
│ │   teacher_vol_20d      = 1.8%   (Low volatility)        │     │
│ │   teacher_tail_risk_20d= 0.15   (Moderate tail risk)    │     │
│ │   teacher_prob_up_20d  = 0.55   (55% prob of gain)      │     │
│ └─────────────────────────────────────────────────────────┘     │
│                                                                   │
│ Key difference from Chronos:                                     │
│   - Chronos: Single symbol, 512 days time-series                 │
│   - Selector: All symbols, 1 day cross-section                   │
│                                                                   │
│ Ranking constraint:                                              │
│   Model scores all 450 symbols together, then ranks              │
│   Top K (e.g., K=30) selected for portfolio                      │
└──────────────────────────────────────────────────────────────────┘

NO TIME-SERIES DIMENSION IN SELECTOR:
┌──────────────────────────────────────────────────────────────────┐
│ ✓ CORRECT: Features are snapshots as of inference date           │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ # Each feature is a scalar (or rank) at time t          │     │
│ │ features = {                                            │     │
│ │     "log_return_1d_rank": 0.45,  # Today's rank         │     │
│ │     "volatility_20d_rank": 0.23  # Rank of 20-day vol   │     │
│ │ }                                                        │     │
│ │ # No sequence dimension                                 │     │
│ └─────────────────────────────────────────────────────────┘     │
│                                                                   │
│ ✗ INCORRECT: Providing historical sequence to Selector           │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ # BAD: Selector is not a time-series model              │     │
│ │ features = {                                            │     │
│ │     "log_return_seq": [0.01, -0.02, 0.03, ...],  # NO!  │     │
│ │     "volatility_seq": [0.15, 0.18, 0.16, ...]    # NO!  │     │
│ │ }                                                        │     │
│ │ # Selector expects cross-section, not time-series       │     │
│ └─────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

**Architecture Alignment**:
- **Chronos**: Time-series encoder (T5) → Sequence understanding
- **Selector**: Cross-sectional ranker (Transformer) → Relative comparison
- **Different modalities**, different input shapes, different objectives

---

### 6.3 Multi-Horizon Target Alignment

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MULTI-HORIZON TARGET CONSTRUCTION                    │
│                  (Training selector with 5-day horizon)                  │
└─────────────────────────────────────────────────────────────────────────┘

SELECTOR TARGET HORIZON (H_sel = 5 trading days):
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                           │
│ Training observation: 2024-01-15                                         │
│                                                                           │
│  Features (known at t)      │  Target (known at t+H_sel)                 │
│  ◄─────────────────────────▶│  ◄─────────────────────────────────────▶  │
│                              │                                            │
│  2024-01-15 (t)             │  2024-01-15 (t) → 2024-01-22 (t+5)        │
│  ┌──────────────────────┐   │  ┌──────────────────────────────────┐     │
│  │ Symbol: AAPL         │   │  │ close_t = $185.60                │     │
│  │                      │   │  │ close_{t+5} = $192.30            │     │
│  │ log_return_1d_rank   │   │  │                                   │     │
│  │ log_return_5d_rank   │   │  │ fwd_return_5d =                  │     │
│  │ volatility_rank      │   │  │   log(192.30 / 185.60) = 0.0356  │     │
│  │ ...                  │   │  │   (3.56% gain)                   │     │
│  │ teacher_drift        │   │  │                                   │     │
│  │ teacher_vol          │   │  │ Cross-sectional rank:            │     │
│  └──────────────────────┘   │  │   y_rank = 0.67 (rank 301/450)   │     │
│                              │  │                                   │     │
│                              │  │ Risk adjustment:                 │     │
│                              │  │   fwd_volatility = 0.018         │     │
│                              │  │   risk_adj = 0.67/0.018 = 37.2   │     │
│                              │  │   P70(risk_adj) = 25.0           │     │
│                              │  │   y_trade = 1 (37.2 >= 25.0) ✓   │     │
│                              │  └──────────────────────────────────┘     │
│                              │                                            │
└──────────────────────────────────────────────────────────────────────────┘

TRADING CALENDAR ALIGNMENT (CRITICAL):
┌──────────────────────────────────────────────────────────────────┐
│ H_sel = 5 TRADING days, not calendar days                        │
│                                                                   │
│ Example: 2024-01-15 (Monday) + 5 trading days = ?                │
│                                                                   │
│ Naive calendar calculation (WRONG):                              │
│   2024-01-15 + 5 calendar days = 2024-01-20 (Saturday)           │
│   → Market closed, no price data!                                │
│                                                                   │
│ Correct trading day calculation:                                 │
│   Start: 2024-01-15 (Mon)                                        │
│   +1: 2024-01-16 (Tue)                                           │
│   +2: 2024-01-17 (Wed)                                           │
│   +3: 2024-01-18 (Thu)                                           │
│   +4: 2024-01-19 (Fri)                                           │
│   +5: 2024-01-22 (Mon, after weekend)                            │
│   → Correct target date!                                         │
│                                                                   │
│ With holiday: 2024-01-12 (Fri) + 5 trading days = ?              │
│   Start: 2024-01-12 (Fri)                                        │
│   +1: 2024-01-16 (Tue, Mon was MLK Day)                          │
│   +2: 2024-01-17 (Wed)                                           │
│   +3: 2024-01-18 (Thu)                                           │
│   +4: 2024-01-19 (Fri)                                           │
│   +5: 2024-01-22 (Mon)                                           │
│   → 10 calendar days, 5 trading days                             │
└──────────────────────────────────────────────────────────────────┘

CHRONOS MULTI-HORIZON vs SELECTOR SINGLE-HORIZON:
┌──────────────────────────────────────────────────────────────────┐
│ Chronos Teacher (distributional, multiple horizons):             │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ Horizons: h ∈ [1, 2, 3, ..., 21] trading days          │     │
│ │                                                          │     │
│ │ For each horizon h, predict:                            │     │
│ │   Q_τ(y_h) for τ ∈ [0.1, 0.2, ..., 0.9]                │     │
│ │   (9 quantiles per horizon)                             │     │
│ │                                                          │     │
│ │ Total outputs: 21 horizons × 3 params = 63 values       │     │
│ │   (μ_h, σ_h, T_h for each h)                            │     │
│ │                                                          │     │
│ │ Aggregated into priors:                                 │     │
│ │   drift ← mean(μ_1, ..., μ_5)                           │     │
│ │   vol   ← mean(σ_1, ..., σ_5)                           │     │
│ │   tail  ← quantile spread                               │     │
│ └─────────────────────────────────────────────────────────┘     │
│                                                                   │
│ Selector (point prediction, single horizon):                     │
│ ┌─────────────────────────────────────────────────────────┐     │
│ │ Horizon: H_sel = 5 trading days (fixed)                 │     │
│ │                                                          │     │
│ │ Target: y_rank (cross-sectional rank of fwd_return_5d)  │     │
│ │                                                          │     │
│ │ No multi-horizon prediction                             │     │
│ │ (uses Chronos priors for longer-term context)           │     │
│ └─────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

**Design Rationale**:
- **Chronos**: Captures full distribution over multiple horizons (uncertainty quantification)
- **Selector**: Optimizes for specific rebalancing horizon (actionable ranking)
- **Separation of concerns**: Chronos = regime awareness, Selector = tactical timing

---

### 6.4 Warmup Period & Edge Cases

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    WARMUP PERIOD HANDLING DIAGRAM                        │
└─────────────────────────────────────────────────────────────────────────┘

PROBLEM: Rolling features need historical data
┌──────────────────────────────────────────────────────────────────┐
│ Feature: volatility_20d (20-day rolling std of returns)          │
│                                                                   │
│ Earliest valid computation:                                      │
│   Date: 2024-01-01 (first date in dataset)                       │
│   Need: 20 prior days of returns                                 │
│   Available: None (start of data)                                │
│   Result: NaN (not enough data)                                  │
│                                                                   │
│ Timeline:                                                         │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ 2024-01-01  volatility_20d = NaN  (0 days available)   │      │
│ │ 2024-01-02  volatility_20d = NaN  (1 day available)    │      │
│ │ ...                                                     │      │
│ │ 2024-01-19  volatility_20d = NaN  (18 days available)  │      │
│ │ 2024-01-22  volatility_20d = 0.025 ✓ (20 days available)│     │
│ │ 2024-01-23  volatility_20d = 0.027 ✓ (21 days available)│     │
│ └────────────────────────────────────────────────────────┘      │
│                                                                   │
│ Warmup period: 20 trading days                                   │
│ First valid date: 2024-01-22                                     │
└──────────────────────────────────────────────────────────────────┘

MAXIMUM WARMUP CALCULATION:
┌──────────────────────────────────────────────────────────────────┐
│ Features with different lookback windows:                        │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ log_return_1d      → 1 day lookback                    │      │
│ │ log_return_5d      → 5 day lookback                    │      │
│ │ log_return_21d     → 21 day lookback                   │      │
│ │ volatility_20d     → 20 day lookback                   │      │
│ │ volatility_60d     → 60 day lookback ← LONGEST         │      │
│ │ rel_volume (MA21)  → 21 day lookback                   │      │
│ │ rsi_14             → 14 day lookback + 1 for diff      │      │
│ └────────────────────────────────────────────────────────┘      │
│                                                                   │
│ Max warmup = max(60, 21+1, 20, 21, 5, 1) = 60 trading days      │
│                                                                   │
│ Plus: Forward returns need +5 trading days at end                │
│   (to compute fwd_return_5d for last date)                       │
│                                                                   │
│ Effective data range:                                            │
│   Raw data: 2020-01-01 → 2024-12-31 (1000 trading days)         │
│   Warmup loss: 60 days at start                                  │
│   Forward loss: 5 days at end                                    │
│   Usable: 2020-03-20 → 2024-12-20 (935 training days)           │
└──────────────────────────────────────────────────────────────────┘

HANDLING STRATEGY (Drop NaN rows):
┌──────────────────────────────────────────────────────────────────┐
│ Pipeline approach:                                                │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ 1. Compute all features (including NaNs)               │      │
│ │    df = compute_features(ohlcv)                        │      │
│ │    # Shape: 1000 rows (some have NaNs)                 │      │
│ │                                                         │      │
│ │ 2. Drop rows with ANY NaN in feature columns           │      │
│ │    feature_cols = [                                    │      │
│ │        "log_return_1d_rank",                           │      │
│ │        "volatility_20d_rank",                          │      │
│ │        ...,                                            │      │
│ │        "y_rank", "y_trade"  # Include targets          │      │
│ │    ]                                                    │      │
│ │    df_clean = df.dropna(subset=feature_cols)           │      │
│ │    # Shape: 935 rows (65 warmup + forward dropped)     │      │
│ │                                                         │      │
│ │ 3. Proceed with training/inference on clean data       │      │
│ └────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘

EDGE CASE: Newly listed symbols
┌──────────────────────────────────────────────────────────────────┐
│ Symbol: NEWCO (IPO date: 2024-06-01)                             │
│                                                                   │
│ Problem: Not enough history for features                         │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ Date: 2024-06-15 (15 trading days since IPO)           │      │
│ │                                                         │      │
│ │ volatility_60d: NaN (need 60 days, have 15)            │      │
│ │ log_return_21d: NaN (need 21 days, have 15)            │      │
│ │                                                         │      │
│ │ Result: Excluded from universe that day                │      │
│ └────────────────────────────────────────────────────────┘      │
│                                                                   │
│ Eligibility filter handles this:                                 │
│   min_history_days = 252 (1 year of trading)                     │
│   → NEWCO excluded until 2025-06-01                              │
│                                                                   │
│ Prevents training on immature symbols                            │
└──────────────────────────────────────────────────────────────────┘

EDGE CASE: Missing data (corporate actions)
┌──────────────────────────────────────────────────────────────────┐
│ Symbol: XYZ (trading halted 2024-03-10 to 2024-03-15)            │
│                                                                   │
│ Gap in data:                                                      │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ 2024-03-08  close = $50.00                             │      │
│ │ 2024-03-09  close = $48.50  (last before halt)         │      │
│ │ 2024-03-10  MISSING (halted)                           │      │
│ │ 2024-03-11  MISSING                                    │      │
│ │ ...                                                     │      │
│ │ 2024-03-15  MISSING                                    │      │
│ │ 2024-03-16  close = $52.00  (resumed)                  │      │
│ └────────────────────────────────────────────────────────┘      │
│                                                                   │
│ Rolling features on 2024-03-16:                                  │
│   volatility_20d: Computed over available 15 days → NaN          │
│   → Row excluded from training/inference                         │
│                                                                   │
│ Symbol re-eligible after warmup period:                          │
│   2024-03-16 + 60 trading days = 2024-06-10                      │
│   → XYZ re-enters universe on 2024-06-10                         │
└──────────────────────────────────────────────────────────────────┘
```

**Robustness**:
- Warmup dropping ensures all features have valid lookback
- Missing data handled gracefully (exclude symbol temporarily)
- No imputation (avoids introducing bias)
- Universe filters catch edge cases before feature engineering

---

## 7. SUMMARY & KEY TAKEAWAYS

### 7.1 Critical Path Dependencies

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SYSTEM CRITICAL PATH ANALYSIS                         │
└─────────────────────────────────────────────────────────────────────────┘

DEPENDENCY LAYERS (Bottom-up):
┌──────────────────────────────────────────────────────────────────┐
│                                                                   │
│ Layer 1: FOUNDATION (Infrastructure)                             │
│ ├─ Trading Calendar (NYSE)                 [Singleton, cached]   │
│ ├─ PipelineConfig                           [Immutable]          │
│ ├─ Artifact Paths                           [Static mapping]     │
│ └─ Schema Contracts                         [Versioned]          │
│     │                                                             │
│     └──▶ ALL downstream systems depend on these                  │
│                                                                   │
│                                                                   │
│ Layer 2: DATA INGESTION                                          │
│ ├─ Canonical OHLCV                          [Daily, ~5 min]      │
│ │   Dependencies: None (raw API fetch)                           │
│ │   Bottleneck: API rate limits (20 concurrent)                  │
│ │                                                                 │
│ └─ Universe Builder                         [Daily, ~30 sec]     │
│     Dependencies: Canonical OHLCV                                │
│     Bottleneck: Dollar volume computation                        │
│                                                                   │
│                                                                   │
│ Layer 3: MARKET ENRICHMENT (Parallel streams)                    │
│ ├─ Breadth Indicators                       [Daily, ~10 sec]     │
│ │   Dependencies: Canonical OHLCV + Universe                     │
│ │   Bottleneck: Vectorized computation (fast)                    │
│ │                                                                 │
│ ├─ Market Covariates                        [Daily, ~5 sec]      │
│ │   Dependencies: Index data (SPY, QQQ, etc.)                    │
│ │   Bottleneck: API fetch (4 symbols)                            │
│ │                                                                 │
│ └─ MarketFrame Assembly                     [Daily, ~20 sec]     │
│     Dependencies: OHLCV + Universe + Breadth + Covariates        │
│     Bottleneck: Multi-way join (indexed, fast)                   │
│                                                                   │
│                                                                   │
│ Layer 4: FEATURE ENGINEERING                                     │
│ └─ Selector Features                        [Daily, ~2 min]      │
│     Dependencies: MarketFrame                                    │
│     Bottleneck: Cross-sectional ranking (450 symbols)            │
│                                                                   │
│                                                                   │
│ Layer 5: MODEL INFERENCE (Serial chain)                          │
│ ├─ Chronos Teacher Priors                   [Daily, ~30 sec GPU] │
│ │   Dependencies: Canonical OHLCV (512-day windows)              │
│ │   Bottleneck: GPU inference (batched, 15 batches)              │
│ │   ↓                                                             │
│ └─ Selector Inference                       [Daily, ~2 sec GPU]  │
│     Dependencies: Features + Priors                              │
│     Bottleneck: Model forward pass (single batch)                │
│                                                                   │
│                                                                   │
│ Layer 6: PORTFOLIO CONSTRUCTION                                  │
│ ├─ HRP Allocation                           [Daily, ~3 sec]      │
│ │   Dependencies: Scores + Covariance matrix                     │
│ │   Bottleneck: Hierarchical clustering (30 symbols)             │
│ │   ↓                                                             │
│ └─ Order Generation                         [Daily, <1 sec]      │
│     Dependencies: Target portfolio + Current positions           │
│     Bottleneck: None (simple diff)                               │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘

CRITICAL PATH TIMELINE (Nightly):
┌──────────────────────────────────────────────────────────────────┐
│ T=0:00    Market close (4:00 PM ET)                              │
│ T+2:00    Data ingestion starts (6:00 PM ET)                     │
│   ├─ T+2:00 to T+2:02  Fetch OHLCV (parallel)                    │
│   ├─ T+2:02 to T+2:02  Validate & canonicalize                   │
│   ├─ T+2:02 to T+2:03  Universe update                           │
│   ├─ T+2:03 to T+2:03  Breadth + Covariates (parallel)           │
│   ├─ T+2:03 to T+2:03  MarketFrame assembly                      │
│   ├─ T+2:03 to T+2:05  Feature engineering                       │
│   ├─ T+2:05 to T+2:05  Chronos priors (GPU) ← BOTTLENECK         │
│   ├─ T+2:05 to T+2:05  Selector inference (GPU)                  │
│   ├─ T+2:05 to T+2:06  Portfolio construction                    │
│   └─ T+2:06 to T+2:06  Order generation                          │
│ T+2:06    Orders queued for next-day execution                   │
│                                                                   │
│ TOTAL ELAPSED: 3 minutes 36 seconds                              │
│ BOTTLENECK: Chronos GPU inference (30 sec of 216 sec total)      │
│ PARALLELISM: Breadth + Covariates (saves ~3 sec)                 │
└──────────────────────────────────────────────────────────────────┘
```

**Optimization Opportunities**:
1. **Chronos batching**: Increase batch size (32→64) if GPU memory allows (~10% speedup)
2. **Feature caching**: Cache rolling statistics, recompute only latest day (~50% feature time reduction)
3. **Async I/O**: Parallelize OHLCV writes during validation (~15 sec savings)

---

### 7.2 Failure Modes & Recovery

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       SYSTEM FAILURE MODES & RECOVERY                    │
└─────────────────────────────────────────────────────────────────────────┘

FAILURE MODE 1: Data Ingestion Failure
┌──────────────────────────────────────────────────────────────────┐
│ Symptom: API returns HTTP 429 (rate limit) or 500 (server error) │
│                                                                   │
│ Impact: Missing data for subset of symbols                       │
│                                                                   │
│ Detection:                                                        │
│   - HTTP status code check                                       │
│   - Expected vs actual symbol count (500 expected, 487 fetched)  │
│                                                                   │
│ Recovery Strategy:                                                │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ 1. Exponential backoff retry (3 attempts)              │      │
│ │    wait_time = 2^attempt × random(0.5, 1.5) seconds   │      │
│ │                                                         │      │
│ │ 2. If still failing: Use stale data from yesterday     │      │
│ │    - Load yesterday's canonical data for failed symbols│      │
│ │    - Mark as "stale" in metadata                       │      │
│ │    - Alert: "Using stale data for AAPL, MSFT (+11)"   │      │
│ │                                                         │      │
│ │ 3. Fallback provider: Switch to backup API             │      │
│ │    Primary: Yahoo Finance                              │      │
│ │    Backup: Polygon.io / Alpha Vantage                  │      │
│ │                                                         │      │
│ │ 4. If all fail: Exclude symbols from today's universe  │      │
│ │    - Mark as "data_unavailable"                        │      │
│ │    - Reduce universe size (450 → 437 tradable)         │      │
│ │    - Continue with reduced universe                    │      │
│ └────────────────────────────────────────────────────────┘      │
│                                                                   │
│ Alerting: Email + Slack notification                             │
└──────────────────────────────────────────────────────────────────┘

FAILURE MODE 2: Validation Failure (Quarantine)
┌──────────────────────────────────────────────────────────────────┐
│ Symptom: Stock split not adjusted, creates invalid returns       │
│                                                                   │
│ Example: NVDA 10-for-1 split on 2024-06-07                       │
│   2024-06-06: close = $1200.00                                   │
│   2024-06-07: close = $120.00 (unadjusted)                       │
│   Return = -90% ← INVALID                                        │
│                                                                   │
│ Detection:                                                        │
│   Stage 7 validation: (ratio - 1) <= -1  (>100% drop)            │
│   invalid_frac = 1/1000 = 0.001 < threshold (0.02)               │
│   → Tolerated (single bad row)                                   │
│                                                                   │
│ If multiple bad rows: invalid_frac > 0.02                        │
│   → QUARANTINE triggered                                         │
│                                                                   │
│ Recovery Strategy:                                                │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ 1. Write quarantine report:                            │      │
│ │    /artifacts/quarantine/NVDA_20240607.json            │      │
│ │    {                                                    │      │
│ │      "symbol": "NVDA",                                 │      │
│ │      "date": "2024-06-07",                             │      │
│ │      "issue": "Invalid return: -90.0%",                │      │
│ │      "recommended_action": "Check split adjustment"    │      │
│ │    }                                                    │      │
│ │                                                         │      │
│ │ 2. Exclude symbol from canonical write                 │      │
│ │    (NVDA not available for today's inference)          │      │
│ │                                                         │      │
│ │ 3. Automated fix attempt:                              │      │
│ │    - Fetch adjusted prices from alternate provider     │      │
│ │    - Re-validate                                       │      │
│ │    - If pass: Write to canonical                       │      │
│ │                                                         │      │
│ │ 4. Manual review queue:                                │      │
│ │    - Notify data team                                  │      │
│ │    - Review split/dividend adjustments                 │      │
│ │    - Backfill corrected data                           │      │
│ └────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘

FAILURE MODE 3: Model Inference Failure
┌──────────────────────────────────────────────────────────────────┐
│ Symptom: GPU out of memory (OOM) during Chronos inference        │
│                                                                   │
│ Cause: Batch size too large for available GPU memory             │
│   Batch size: 32 symbols                                         │
│   GPU memory: 16GB (allocated: 15.8GB) → OOM                     │
│                                                                   │
│ Detection:                                                        │
│   torch.cuda.OutOfMemoryError exception                          │
│                                                                   │
│ Recovery Strategy:                                                │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ 1. Reduce batch size dynamically:                      │      │
│ │    try:                                                │      │
│ │        batch_size = 32                                 │      │
│ │        priors = chronos.infer(batch_size=32)           │      │
│ │    except torch.cuda.OutOfMemoryError:                 │      │
│ │        torch.cuda.empty_cache()                        │      │
│ │        batch_size = 16  # Halve batch size             │      │
│ │        priors = chronos.infer(batch_size=16)           │      │
│ │                                                         │      │
│ │ 2. If still failing: Fall back to CPU                  │      │
│ │    - Slower (~5min vs 30sec) but reliable              │      │
│ │    - Alert: "Using CPU inference (slow)"               │      │
│ │                                                         │      │
│ │ 3. If CPU also fails: Use yesterday's priors           │      │
│ │    - Load priors from previous day                     │      │
│ │    - Mark as "stale" in metadata                       │      │
│ │    - Alert: "CRITICAL: Using stale priors"             │      │
│ └────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘

FAILURE MODE 4: Portfolio Construction Singularity
┌──────────────────────────────────────────────────────────────────┐
│ Symptom: HRP allocation fails due to singular covariance matrix  │
│                                                                   │
│ Cause: Selected stocks are perfectly correlated                  │
│   (e.g., all tech stocks in same sector during crisis)           │
│   Covariance matrix rank < N → Not invertible                    │
│                                                                   │
│ Detection:                                                        │
│   np.linalg.LinAlgError: Singular matrix                         │
│                                                                   │
│ Recovery Strategy:                                                │
│ ┌────────────────────────────────────────────────────────┐      │
│ │ 1. Add regularization:                                 │      │
│ │    cov_regularized = cov + λ·I                         │      │
│ │    where λ = 1e-5 (small diagonal loading)             │      │
│ │                                                         │      │
│ │ 2. If still singular: Fall back to equal weights       │      │
│ │    weights = [1/N, 1/N, ..., 1/N]                      │      │
│ │    (Simple but robust)                                 │      │
│ │                                                         │      │
│ │ 3. Alert: "Using equal weights (HRP failed)"           │      │
│ └────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘
```

---

### 7.3 Performance Benchmarks

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SYSTEM PERFORMANCE BENCHMARKS                     │
│                       (Baseline: 500 symbols, 1000 days)                 │
└─────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────┬──────────────┬──────────────┬──────────┐
│ Component                      │ Time (Polars)│ Time (Pandas)│ Speedup  │
├────────────────────────────────┼──────────────┼──────────────┼──────────┤
│ Canonical validation (8-stage) │ 30 sec       │ 3 min        │ 6.0x     │
│ Universe construction          │ 5 sec        │ 45 sec       │ 9.0x     │
│ Breadth computation (vectorized)│10 sec       │ 3 min 7 sec  │ 18.7x    │
│ Feature engineering (rank norm)│ 2 min        │ 20 min       │ 10.0x    │
│ Total data pipeline            │ 3 min        │ 27 min       │ 9.0x     │
├────────────────────────────────┼──────────────┼──────────────┼──────────┤
│ Chronos inference (GPU)        │ 30 sec       │ 30 sec       │ 1.0x     │
│ Chronos inference (CPU)        │ 5 min        │ 5 min        │ 1.0x     │
│ Selector inference (GPU)       │ 2 sec        │ 2 sec        │ 1.0x     │
├────────────────────────────────┼──────────────┼──────────────┼──────────┤
│ HRP allocation (30 stocks)     │ 3 sec        │ 3 sec        │ 1.0x     │
│ Order generation               │ <1 sec       │ <1 sec       │ 1.0x     │
└────────────────────────────────┴──────────────┴──────────────┴──────────┘

MEMORY USAGE:
┌────────────────────────────────┬──────────────┬──────────────┐
│ Dataset                        │ Polars       │ Pandas       │
├────────────────────────────────┼──────────────┼──────────────┤
│ Canonical OHLCV (500×1000)     │ 25 MB        │ 180 MB       │
│ Features (450×800×28 cols)     │ 15 MB        │ 95 MB        │
│ Total working set              │ 40 MB        │ 275 MB       │
├────────────────────────────────┼──────────────┼──────────────┤
│ Reduction                      │ 6.9x smaller │              │
└────────────────────────────────┴──────────────┴──────────────┘

SCALABILITY (Linear scaling observed):
┌────────────────────┬──────────┬──────────┬──────────┬──────────┐
│ Universe Size      │ 100 sym  │ 500 sym  │ 1000 sym │ 2000 sym │
├────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Feature eng (Polars)│ 25 sec   │ 2 min    │ 4 min    │ 8 min    │
│ Chronos priors (GPU)│ 8 sec    │ 30 sec   │ 60 sec   │ 120 sec  │
│ Selector inference │ 1 sec    │ 2 sec    │ 4 sec    │ 8 sec    │
│ Total pipeline     │ 40 sec   │ 3.5 min  │ 7 min    │ 14 min   │
└────────────────────┴──────────┴──────────┴──────────┴──────────┘

BOTTLENECK ANALYSIS (500 symbols):
┌────────────────────────────────────────────────────────────────┐
│ Component               Time    % of Total  Bottleneck Type    │
├────────────────────────────────────────────────────────────────┤
│ Data ingestion          2 min   55%         I/O (API rate)     │
│ Feature engineering     2 min   55%         CPU (ranking)      │
│ Chronos inference       30 sec  14%         GPU (batching)     │
│ Selector inference      2 sec   1%          GPU (single batch) │
│ Portfolio construction  3 sec   1%          CPU (clustering)   │
│ Other (joins, writes)   20 sec  9%          I/O (disk)         │
├────────────────────────────────────────────────────────────────┤
│ TOTAL                   3.6 min 100%                           │
└────────────────────────────────────────────────────────────────┘
```

**Performance Takeaways**:
1. **Polars essential**: 6-18x speedup for data operations (90% cost reduction)
2. **GPU critical**: Chronos 10x faster on GPU vs CPU (30sec vs 5min)
3. **Linear scaling**: Doubling universe size doubles pipeline time (predictable)
4. **Parallelism gains**: Breadth/covariates parallel saves ~5 sec (16% reduction)

---

## CONCLUSION

This document provides comprehensive visual representations of the Algaie trading system architecture, covering:

1. **System-Wide Architecture** - Component hierarchy and end-to-end data flow
2. **Data Flow Pipelines** - Validation, universe construction, feature engineering
3. **Component Interactions** - Training and inference workflows
4. **State Machines** - Risk posture transitions, artifact lifecycle
5. **Training vs Inference** - Comparison of operational modes
6. **Time-Series Causality** - Context windows, horizon alignment, warmup handling
7. **Performance Analysis** - Benchmarks, bottlenecks, failure modes

**Document Statistics**:
- Total lines: ~2,200
- ASCII diagrams: 35
- Code examples: 20
- Workflow visualizations: 12

**Cross-References**:
- System Analysis Parts 1-4 for detailed code walkthroughs
- Deep Dive Validation for validation system specifics
- Integration Workflows (next document) for end-to-end operational sequences

---

*End of Visual Diagrams Document - Last Updated: 2026-02-13*
