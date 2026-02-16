# ALGAIE SYSTEM ANALYSIS - PART 4
## Feature Engineering & Model Architecture (Exhaustive Detail)

*Continuation from SYSTEM_ANALYSIS_PART3.md*

---

# PART 4: FEATURE ENGINEERING & MODEL ARCHITECTURE

## 4.1 Selector Feature Engineering Pipeline

### Purpose
Transform raw OHLCV data and universe metadata into normalized, cross-sectional features suitable for ranking model training. The pipeline ensures **point-in-time safety** (no look-ahead bias) and **cross-sectional comparability** through rank-based normalization.

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                SELECTOR FEATURE ENGINEERING PIPELINE                     │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  INPUT: Raw Market Data                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  OHLCV: [date, symbol, close, volume] (500 symbols × 1000 days)│   │
│  │  Universe: [date, symbol, is_tradable, tier, weight]           │   │
│  └─────────────────────────────────┬───────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │       STAGE 1: RAW FEATURE COMPUTATION                          │   │
│  │                                                                  │   │
│  │  Computed features (per symbol, time-series):                   │   │
│  │                                                                  │   │
│  │  1. log_return_1d = log(close_t / close_{t-1})                 │   │
│  │     Purpose: 1-day momentum                                     │   │
│  │     Range: typically [-0.1, 0.1] (~±10%)                        │   │
│  │                                                                  │   │
│  │  2. log_return_5d = Σ(log_return_1d over 5 days)               │   │
│  │     Purpose: Short-term trend                                   │   │
│  │     Range: typically [-0.2, 0.2] (~±20%)                        │   │
│  │                                                                  │   │
│  │  3. log_return_20d = Σ(log_return_1d over 20 days)             │   │
│  │     Purpose: Medium-term trend                                  │   │
│  │     Range: typically [-0.3, 0.3] (~±30%)                        │   │
│  │                                                                  │   │
│  │  4. volatility_20d = std(log_return_1d[t-19:t])                │   │
│  │     Purpose: Risk measure                                       │   │
│  │     Range: typically [0.01, 0.08] (1-8% daily vol)             │   │
│  │                                                                  │   │
│  │  5. relative_volume_20d = volume_t / median(volume[t-20:t-1])  │   │
│  │     Purpose: Volume anomaly detection                           │   │
│  │     Range: [0, ∞), typically [0.5, 2.0]                         │   │
│  │                                                                  │   │
│  │  6. target_return_raw = log(close_{t+5} / close_t)             │   │
│  │     Purpose: Forward return label (5-day default)               │   │
│  │     ⚠️  Uses shift(-5) - look-ahead for labels only            │   │
│  │                                                                  │   │
│  │  7. fwd_volatility = std(log_return_1d[t:t+5])                 │   │
│  │     Purpose: Forward risk for risk-adjusted ranking             │   │
│  │     ⚠️  Uses shift(-5) - look-ahead for labels only            │   │
│  └─────────────────────────────────┬───────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │       STAGE 2: UNIVERSE JOIN & FILTER                            │   │
│  │                                                                  │   │
│  │  Join features with universe on [date, symbol]:                 │   │
│  │    features.join(universe, how="inner")                         │   │
│  │                                                                  │   │
│  │  Filter to tradable universe:                                   │   │
│  │    .filter(is_tradable == True)                                 │   │
│  │                                                                  │   │
│  │  Example before/after:                                          │   │
│  │    Before: 8000 symbols × 1000 days = 8M rows                  │   │
│  │    After: 500 symbols × 1000 days = 500K rows                  │   │
│  │    Reduction: 93.75%                                            │   │
│  └─────────────────────────────────┬───────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │       STAGE 3: BREADTH FILTERING                                 │   │
│  │                                                                  │   │
│  │  Remove days with insufficient cross-sectional breadth:         │   │
│  │                                                                  │   │
│  │  breadth_per_day = count(symbol) GROUP BY date                  │   │
│  │  valid_days = days WHERE breadth >= min_breadth_train (200)    │   │
│  │                                                                  │   │
│  │  Rationale:                                                     │   │
│  │    - Cross-sectional ranking requires sufficient samples        │   │
│  │    - <200 symbols → rank statistics unreliable                  │   │
│  │    - Protects against market holidays, data gaps               │   │
│  │                                                                  │   │
│  │  Example:                                                       │   │
│  │  ┌──────────┬──────────┬────────┐                              │   │
│  │  │   date   │ breadth  │ valid? │                              │   │
│  │  ├──────────┼──────────┼────────┤                              │   │
│  │  │2025-01-15│   487    │  Yes   │                              │   │
│  │  │2025-01-16│   492    │  Yes   │                              │   │
│  │  │2025-01-17│   156    │  NO!   │ ← Market disruption           │   │
│  │  │2025-01-18│   501    │  Yes   │                              │   │
│  │  └──────────┴──────────┴────────┘                              │   │
│  │                                                                  │   │
│  │  Drop: 2025-01-17 (all rows for this date)                     │   │
│  └─────────────────────────────────┬───────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │       STAGE 4: CROSS-SECTIONAL RANK NORMALIZATION                │   │
│  │                                                                  │   │
│  │  Transform raw features to [-1, 1] via rank percentile:         │   │
│  │                                                                  │   │
│  │  For each feature f and each date d:                            │   │
│  │    1. Rank symbols by feature value (ordinal ranking)           │   │
│  │       rank ∈ [0, N-1] where N = symbols on date d              │   │
│  │                                                                  │   │
│  │    2. Normalize to [-1, 1]:                                     │   │
│  │       x_norm = 2 × (rank / (N - 1)) - 1                        │   │
│  │                                                                  │   │
│  │  Properties:                                                    │   │
│  │    - Lowest rank (0) → -1.0 (worst)                             │   │
│  │    - Median rank (N/2) → 0.0 (neutral)                          │   │
│  │    - Highest rank (N-1) → +1.0 (best)                           │   │
│  │    - Uniform distribution by construction                       │   │
│  │    - Outlier-robust (ranks insensitive to extreme values)      │   │
│  │                                                                  │   │
│  │  Special handling for volatility:                               │   │
│  │    vol_signal = -volatility_20d  (flip sign)                   │   │
│  │    Rationale: Lower vol = better (safer asset)                 │   │
│  │                                                                  │   │
│  │  Output features:                                               │   │
│  │    x_lr1:    normalized 1-day return                            │   │
│  │    x_lr5:    normalized 5-day return                            │   │
│  │    x_lr20:   normalized 20-day return                           │   │
│  │    x_vol:    normalized inverse volatility                      │   │
│  │    x_relvol: normalized relative volume                         │   │
│  └─────────────────────────────────┬───────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │       STAGE 5: TARGET COMPUTATION                                │   │
│  │                                                                  │   │
│  │  Label 1: y_rank (raw forward return)                           │   │
│  │    y_rank = target_return_raw                                   │   │
│  │    Used for: Quantile loss, return prediction                   │   │
│  │                                                                  │   │
│  │  Label 2: y_trade (binary trade signal)                         │   │
│  │    Step 1: Risk-adjusted return                                 │   │
│  │      risk_adj = y_rank / (fwd_volatility + ε)                  │   │
│  │      (or use historical vol if fwd_vol missing)                 │   │
│  │                                                                  │   │
│  │    Step 2: Rank percentile                                      │   │
│  │      rank_pct = rank(risk_adj) / (N - 1)  per date             │   │
│  │                                                                  │   │
│  │    Step 3: Binary threshold                                     │   │
│  │      y_trade = 1 if rank_pct >= 0.70 else 0                    │   │
│  │                                                                  │   │
│  │  Interpretation:                                                │   │
│  │    y_trade = 1: Top 30% risk-adjusted performers               │   │
│  │    y_trade = 0: Bottom 70% (avoid/short)                        │   │
│  │                                                                  │   │
│  │  Why risk-adjusted?                                             │   │
│  │    - Sharpe ratio analog (return / risk)                        │   │
│  │    - Prefer stable gains over volatile returns                  │   │
│  │    - Matches real portfolio construction                        │   │
│  └─────────────────────────────────┬───────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │       STAGE 6: VALIDATION                                        │   │
│  │                                                                  │   │
│  │  Schema Check:                                                  │   │
│  │    assert_schema(df, SELECTOR_FEATURES_V2_REQUIRED_COLS)        │   │
│  │    Required: [date, symbol, x_lr1, x_lr5, x_lr20, x_vol,       │   │
│  │              x_relvol, y_rank, tier, weight]                    │   │
│  │                                                                  │   │
│  │  Bounds Check:                                                  │   │
│  │    For all x_* features:                                        │   │
│  │      assert -1.001 <= min(x) and max(x) <= 1.001               │   │
│  │                                                                  │   │
│  │  Fail-fast: Raise ValueError if any violation                   │   │
│  └─────────────────────────────────┬───────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  OUTPUT: Normalized Feature DataFrame                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Shape: ~500K rows × 10+ columns                                │   │
│  │  Schema: [date, symbol, x_lr1, x_lr5, x_lr20, x_vol,           │   │
│  │           x_relvol, y_rank, y_trade, tier, weight]             │   │
│  │                                                                  │   │
│  │  Properties:                                                    │   │
│  │    - All x_* features ∈ [-1, 1]                                 │   │
│  │    - No look-ahead bias (features at t use data ≤ t)           │   │
│  │    - Cross-sectionally comparable (ranks within date)           │   │
│  │    - ML-ready (normalized, no missing values in features)      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Implementation Deep-Dive

#### 4.1.1 Raw Feature Computation

**Polars Lazy Evaluation**:
```python
def _compute_raw_features(self, ohlcv: pl.LazyFrame) -> pl.LazyFrame:
    """
    Lazy computation enables query optimization.

    Benefits:
    1. Polars can fuse operations (fewer passes through data)
    2. Memory efficient (only materialize when .collect())
    3. Parallel execution across cores
    """
    cfg = self.config
    return (
        ohlcv.sort(["symbol", "date"])
        .with_columns(
            # Log return: log(P_t / P_{t-1})
            (pl.col("close") / pl.col("close").shift(1).over("symbol"))
            .log()
            .alias("log_return_1d")
        )
        .with_columns([
            # Rolling sum for multi-day returns
            pl.col("log_return_1d").rolling_sum(window_size=5).over("symbol").alias("log_return_5d"),
            pl.col("log_return_1d").rolling_sum(window_size=20).over("symbol").alias("log_return_20d"),

            # Volatility: rolling standard deviation
            pl.col("log_return_1d").rolling_std(window_size=cfg.volatility_window).over("symbol").alias("volatility_20d"),

            # Relative volume: current vol / median(past 20 days vol)
            (pl.col("volume") / (
                pl.col("volume").shift(1).rolling_median(window_size=cfg.relative_vol_window).over("symbol") + 1.0
            )).alias("relative_volume_20d"),

            # FORWARD-LOOKING (labels only):
            # Target return: log(P_{t+H} / P_t)
            (pl.col("close").shift(-cfg.horizon_days).over("symbol") / pl.col("close")).log().alias("target_return_raw"),

            # Forward volatility: std(returns[t:t+H])
            pl.col("log_return_1d").rolling_std(window_size=cfg.horizon_days).shift(-cfg.horizon_days).over("symbol").alias("fwd_volatility"),
        ])
    )
```

**Detailed Feature Semantics**:

```
FEATURE: log_return_1d
Formula: log(close_t / close_{t-1})
Why log returns?
  - Additive (can sum over time)
  - Symmetry: +10% and -10% move have same magnitude
  - Statistical properties (closer to normal distribution)

Example (AAPL):
┌──────────┬────────┬────────────┬───────────────┐
│   date   │ close  │   ratio    │ log_return_1d │
├──────────┼────────┼────────────┼───────────────┤
│2025-01-13│ 150.00 │     -      │      -        │
│2025-01-14│ 151.50 │  1.010000  │  +0.009950    │  (+1.00%)
│2025-01-15│ 149.75 │  0.988448  │  -0.011618    │  (-1.16%)
│2025-01-16│ 152.25 │  1.016694  │  +0.016548    │  (+1.67%)
└──────────┴────────┴────────────┴───────────────┘

FEATURE: log_return_5d
Formula: Σ(log_return_1d over 5 days)
Why sum of log returns = log of compounded return:
  log(P_5 / P_0) = log(P_5/P_4 × P_4/P_3 × ... × P_1/P_0)
                 = log(P_5/P_4) + log(P_4/P_3) + ... + log(P_1/P_0)

Example (TSLA):
┌──────────┬────────┬────────────┬────────────────┐
│   date   │ close  │log_ret_1d  │ log_return_5d  │
├──────────┼────────┼────────────┼────────────────┤
│2025-01-13│ 240.00 │  +0.0080   │      -         │
│2025-01-14│ 242.50 │  +0.0103   │      -         │
│2025-01-15│ 238.00 │  -0.0187   │      -         │
│2025-01-16│ 245.20 │  +0.0299   │      -         │
│2025-01-17│ 248.50 │  +0.0133   │  +0.0428       │  (5-day sum)
│2025-01-20│ 251.30 │  +0.0112   │  +0.0460       │  (rolling)
└──────────┴────────┴────────────┴────────────────┘

Interpretation:
  log_return_5d = +0.0428 → 5-day return ≈ 4.28%
  Positive = uptrend, Negative = downtrend

FEATURE: volatility_20d
Formula: std(log_return_1d[t-19:t])
Why 20 days?
  - ~1 month of trading days
  - Balances responsiveness vs stability
  - Industry standard (common volatility window)

Example (QQQ):
┌──────────┬────────────┬──────────────┐
│   date   │log_ret_1d  │ volatility_20d│
├──────────┼────────────┼──────────────┤
│2025-01-13│  +0.0042   │   0.0152     │  (1.52% daily vol)
│2025-01-14│  -0.0051   │   0.0148     │
│2025-01-15│  +0.0089   │   0.0155     │  ← Spike (high return)
│2025-01-16│  -0.0023   │   0.0157     │  ← Vol increases
│2025-01-17│  +0.0067   │   0.0154     │  ← Normalizing
└──────────┴────────────┴──────────────┘

Annualized volatility ≈ 0.0152 × √252 ≈ 24.1%

FEATURE: relative_volume_20d
Formula: volume_t / median(volume[t-20:t-1])
Why relative vs absolute?
  - Cross-sectionally comparable
  - Detects anomalies (>2.0 = unusual activity)
  - Robust to stock splits (median normalizes)

Example (NVDA):
┌──────────┬────────┬───────────────┬──────────────────┐
│   date   │ volume │ median(20d)   │ relative_vol_20d │
├──────────┼────────┼───────────────┼──────────────────┤
│2025-01-13│  45.2M │    42.5M      │      1.06        │  (normal)
│2025-01-14│  48.7M │    42.8M      │      1.14        │
│2025-01-15│ 125.3M │    43.1M      │      2.91        │  (SPIKE!)
│2025-01-16│  52.1M │    44.2M      │      1.18        │
└──────────┴────────┴───────────────┴──────────────────┘

Interpretation:
  2.91x median volume → Major event (earnings, news)
```

#### 4.1.2 Rank Normalization Algorithm

**Mathematical Foundation**:

```
Given N symbols on date d with feature values f_1, f_2, ..., f_N:

Step 1: Ordinal Ranking
  Sort symbols by f (ascending)
  Assign rank r ∈ {0, 1, ..., N-1}

  Ties: Use "ordinal" method (arbitrary order)

  Example (5 symbols, log_return_1d):
    AAPL: -0.012  →  rank = 0 (worst)
    MSFT: -0.005  →  rank = 1
    TSLA: +0.003  →  rank = 2
    QQQ:  +0.008  →  rank = 3
    SPY:  +0.015  →  rank = 4 (best)

Step 2: Normalize to [-1, 1]
  x_norm = 2 × (r / (N - 1)) - 1

  Proof of bounds:
    r = 0:     x_norm = 2 × (0 / 4) - 1 = -1.0
    r = N/2:   x_norm = 2 × (2 / 4) - 1 =  0.0
    r = N-1:   x_norm = 2 × (4 / 4) - 1 = +1.0

  Example application:
    AAPL: rank=0 → 2×(0/4)-1 = -1.0
    MSFT: rank=1 → 2×(1/4)-1 = -0.5
    TSLA: rank=2 → 2×(2/4)-1 =  0.0
    QQQ:  rank=3 → 2×(3/4)-1 = +0.5
    SPY:  rank=4 → 2×(4/4)-1 = +1.0
```

**Implementation**:
```python
def _apply_rank_normalization(self, df: pl.DataFrame) -> pl.DataFrame:
    # Volatility: lower is better (flip sign)
    df = df.with_columns((-pl.col("volatility_20d")).alias("vol_signal"))

    norm_features = ["log_return_1d", "log_return_5d", "log_return_20d", "vol_signal", "relative_volume_20d"]
    out_names = ["x_lr1", "x_lr5", "x_lr20", "x_vol", "x_relvol"]

    exprs = []
    for feat, out_name in zip(norm_features, out_names):
        # Rank within date (ordinal: 1, 2, 3, ...)
        r = pl.struct([feat, "symbol"]).rank("ordinal").over("date") - 1  # → [0, N-1]

        # Count symbols per date
        n_t = pl.col("symbol").count().over("date")

        # Normalize to [-1, 1]
        norm = (2.0 * (r / (n_t - 1)) - 1.0).alias(out_name)
        exprs.append(norm)

    return df.with_columns(exprs)
```

**Why Ranks Over Z-Scores**:

```
Z-Score Normalization (Standard):
  x_norm = (x - mean(x)) / std(x)

Problems:
  1. Outlier sensitivity
     One extreme value (x = 100σ) distorts all others

  2. Non-uniform distribution
     Heavy tails → most values near 0

  3. Unbounded range
     x_norm ∈ (-∞, +∞) → need clipping for neural nets

Rank Normalization (Algaie Choice):
  x_norm = 2 × (rank(x) / (N-1)) - 1

Benefits:
  1. Outlier robust
     Extreme values only affect their own rank

  2. Uniform distribution
     By construction: equal density in [-1, 1]

  3. Bounded range
     x_norm ∈ [-1, 1] → no clipping needed

  4. Cross-sectional invariance
     Same distribution every day (critical for ranking model)

Example Comparison (100 symbols on 2025-01-15):

Feature: log_return_1d
┌────────┬────────────┬─────────────┬──────────────┐
│ Symbol │ Raw Return │  Z-Score    │  Rank Norm   │
├────────┼────────────┼─────────────┼──────────────┤
│ AAPL   │  -0.025    │  -2.13      │   -0.92      │
│ TSLA   │  +0.045    │  +3.87      │   +0.88      │  ← Outlier
│ MSFT   │  +0.008    │  +0.42      │   +0.12      │
│ QQQ    │  -0.003    │  -0.18      │   -0.08      │
│ SPY    │  +0.002    │  +0.05      │   +0.04      │
└────────┴────────────┴─────────────┴──────────────┘

Distribution:
  Z-Score: Most values in [-1, 1], but tails extend to ±4
  Rank Norm: Perfectly uniform in [-1, 1]
```

#### 4.1.3 Target Computation

**Risk-Adjusted Ranking**:

```python
def _compute_targets(self, df: pl.DataFrame) -> pl.DataFrame:
    # Rename raw target
    df = df.rename({"target_return_raw": "y_rank"})

    # Use forward vol if available, else historical vol
    fwd_vol = pl.coalesce([pl.col("fwd_volatility"), pl.col("volatility_20d")])

    # Sharpe-like ratio
    risk_adj = (pl.col("y_rank") / (fwd_vol + 1e-8)).alias("risk_adj_return")
    df = df.with_columns(risk_adj)

    # Rank by risk-adjusted return
    r = pl.struct(["risk_adj_return", "symbol"]).rank("ordinal").over("date") - 1
    n_t = pl.col("symbol").count().over("date")
    rank_pct = r / (n_t - 1)  # → [0, 1]

    # Top 30% = tradable
    y_trade = (rank_pct >= 0.70).cast(pl.Int32).alias("y_trade")

    return df.with_columns([y_trade])
```

**Worked Example**:

```
Date: 2025-01-15 (500 symbols)

Step 1: Forward Returns (next 5 trading days)
┌────────┬──────────┬──────────────┬────────────────┐
│ Symbol │ Price_t  │ Price_{t+5}  │ Forward Return │
├────────┼──────────┼──────────────┼────────────────┤
│ AAPL   │  150.00  │   152.50     │   +0.0166      │  (+1.66%)
│ MSFT   │  380.00  │   382.80     │   +0.0073      │  (+0.73%)
│ TSLA   │  248.00  │   255.20     │   +0.0287      │  (+2.87%)
│ QQQ    │  425.00  │   428.50     │   +0.0082      │  (+0.82%)
│ SPY    │  450.00  │   451.20     │   +0.0027      │  (+0.27%)
└────────┴──────────┴──────────────┴────────────────┘

Step 2: Forward Volatility (std of next 5 days' returns)
┌────────┬────────────────┬──────────────────┐
│ Symbol │ Forward Return │ Forward Vol      │
├────────┼────────────────┼──────────────────┤
│ AAPL   │   +0.0166      │   0.0120         │  (stable move)
│ MSFT   │   +0.0073      │   0.0095         │  (stable)
│ TSLA   │   +0.0287      │   0.0280         │  (volatile!)
│ QQQ    │   +0.0082      │   0.0088         │  (stable)
│ SPY    │   +0.0027      │   0.0075         │  (very stable)
└────────┴────────────────┴──────────────────┘

Step 3: Risk-Adjusted Return (Sharpe-like)
┌────────┬────────────────┬──────────────────┬──────────────────┐
│ Symbol │ Forward Return │ Forward Vol      │  Risk-Adj Return │
├────────┼────────────────┼──────────────────┼──────────────────┤
│ AAPL   │   +0.0166      │   0.0120         │   1.383          │
│ MSFT   │   +0.0073      │   0.0095         │   0.768          │
│ TSLA   │   +0.0287      │   0.0280         │   1.025          │  ← High return BUT high vol
│ QQQ    │   +0.0082      │   0.0088         │   0.932          │
│ SPY    │   +0.0027      │   0.0075         │   0.360          │
└────────┴────────────────┴──────────────────┴──────────────────┘

Step 4: Rank Percentile (within date, 500 symbols)
  Assume AAPL ranks 420/500 on risk-adj return
  rank_pct = 420 / 499 = 0.841

Step 5: Binary Label
  y_trade = (0.841 >= 0.70) ? 1 : 0
  y_trade = 1 (tradable)

Interpretation:
  AAPL: Good risk-adjusted performer → Long candidate
  TSLA: High raw return but volatile → Lower risk-adj rank
```

**Why 70th Percentile Threshold**:

```
Threshold Analysis:

50th Percentile (Median):
  - Top 50% selected (250 symbols)
  - Too many positions → diversification limit
  - Includes mediocre performers

70th Percentile:
  - Top 30% selected (150 symbols)
  - Reasonable portfolio size (actual top-K might be 5-50)
  - High-confidence trades only

90th Percentile:
  - Top 10% selected (50 symbols)
  - Misses good opportunities
  - Overfitting risk (cherry-picking)

Empirical Tuning:
  - 70th percentile chosen via walk-forward validation
  - Balances precision (avoid false positives) and recall (capture winners)
```

*To be continued: Section 4.2 (Chronos-2 Teacher Architecture)...*

