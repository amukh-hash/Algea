# ALGAIE SYSTEM ANALYSIS - PART 3
## Market Data & Feature Engineering (Exhaustive Detail)

*Continuation from SYSTEM_ANALYSIS_PART2.md*

---

# PART 3: MARKET DATA ENRICHMENT

## 3.1 Market Breadth Indicators

### Purpose
Compute cross-sectional market health metrics that capture overall market sentiment and participation beyond individual stock movements.

### Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                    BREADTH COMPUTATION PIPELINE                        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT: Per-Symbol OHLCV Dict                                         │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  {                                                            │    │
│  │    "AAPL": DataFrame[date, close, ...],                     │    │
│  │    "MSFT": DataFrame[date, close, ...],                     │    │
│  │    "TSLA": DataFrame[date, close, ...],                     │    │
│  │    ...  (~500 symbols)                                       │    │
│  │  }                                                            │    │
│  └───────────────────────────┬──────────────────────────────────┘    │
│                              │                                         │
│                              ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │         PER-SYMBOL DIRECTION COMPUTATION                      │    │
│  │                                                               │    │
│  │  For each symbol:                                            │    │
│  │    1. Sort by date                                           │    │
│  │    2. Compute daily price change                             │    │
│  │       Δclose = close_t - close_{t-1}                        │    │
│  │    3. Compute direction                                      │    │
│  │       direction = sign(Δclose)                               │    │
│  │         +1: Price increased                                  │    │
│  │          0: Price unchanged                                  │    │
│  │         -1: Price decreased                                  │    │
│  │                                                               │    │
│  │  Output per symbol:                                          │    │
│  │    DataFrame[date, direction]                                │    │
│  └───────────────────────────┬──────────────────────────────────┘    │
│                              │                                         │
│                              ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │         CROSS-SECTIONAL AGGREGATION                           │    │
│  │                                                               │    │
│  │  Concatenate all direction DataFrames                        │    │
│  │  ┌────────────────────────────────────────────────────┐      │    │
│  │  │  date       │ symbol │ direction                   │      │    │
│  │  ├─────────────┼────────┼────────────────────────────┤      │    │
│  │  │ 2025-01-15 │  AAPL  │    +1                       │      │    │
│  │  │ 2025-01-15 │  MSFT  │    +1                       │      │    │
│  │  │ 2025-01-15 │  TSLA  │    -1                       │      │    │
│  │  │ 2025-01-15 │  QQQ   │    +1                       │      │    │
│  │  │ 2025-01-15 │  SPY   │    +1                       │      │    │
│  │  │    ...      │  ...   │    ...                      │      │    │
│  │  └────────────────────────────────────────────────────┘      │    │
│  │                                                               │    │
│  │  Group by date and compute:                                  │    │
│  │    advancers = count(direction > 0)                          │    │
│  │    decliners = count(direction < 0)                          │    │
│  │    total_issues = count(*)                                   │    │
│  │    market_breadth_ad = (advancers - decliners) / total       │    │
│  └───────────────────────────┬──────────────────────────────────┘    │
│                              │                                         │
│                              ▼                                         │
│  OUTPUT: Breadth DataFrame                                            │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  Columns:                                                     │    │
│  │    - date: trading day                                       │    │
│  │    - market_breadth_ad: AD ratio ∈ [-1, 1]                  │    │
│  │    - advancers: count of advancing stocks                    │    │
│  │    - decliners: count of declining stocks                    │    │
│  │    - total_issues: total stock count                         │    │
│  │                                                               │    │
│  │  Example:                                                     │    │
│  │  ┌──────────┬────────────┬──────────┬──────────┬───────┐    │    │
│  │  │   date   │ breadth_ad │advancers │decliners │ total │    │    │
│  │  ├──────────┼────────────┼──────────┼──────────┼───────┤    │    │
│  │  │2025-01-15│   0.42     │   355    │   145    │  500  │    │    │
│  │  │2025-01-16│  -0.18     │   205    │   295    │  500  │    │    │
│  │  │2025-01-17│   0.68     │   420    │    80    │  500  │    │    │
│  │  └──────────┴────────────┴──────────┴──────────┴───────┘    │    │
│  │                                                               │    │
│  │  Interpretation:                                              │    │
│  │    breadth_ad > 0.5: Strong bullish breadth                 │    │
│  │    breadth_ad ∈ [-0.2, 0.2]: Neutral/mixed                  │    │
│  │    breadth_ad < -0.5: Strong bearish breadth                │    │
│  └──────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────┘
```

### Implementation Details

#### 3.1.1 Individual Indicator Functions

**Advance/Decline Line**:
```python
def calculate_ad_line(close: pd.Series) -> pd.Series:
    """
    Cumulative sum of daily direction signals.

    Formula:
      direction_t = sign(close_t - close_{t-1})
      AD_line_t = Σ direction_i  (i=1 to t)

    Properties:
      - Trending upward: More advancers than decliners over time
      - Trending downward: More decliners than advancers
      - Divergence from price: Early warning signal
    """
    direction = np.sign(close.diff()).fillna(0).astype(int)
    return direction.cumsum()
```

**Example AD Line Computation**:
```
SPY Close Prices (10 days):
┌──────┬────────┬─────────┬───────────┬─────────┐
│ Day  │ Close  │  Δ      │ Direction │ AD Line │
├──────┼────────┼─────────┼───────────┼─────────┤
│  1   │ 450.0  │   -     │     0     │    0    │
│  2   │ 452.3  │  +2.3   │    +1     │    1    │
│  3   │ 451.8  │  -0.5   │    -1     │    0    │
│  4   │ 455.2  │  +3.4   │    +1     │    1    │
│  5   │ 456.7  │  +1.5   │    +1     │    2    │
│  6   │ 455.9  │  -0.8   │    -1     │    1    │
│  7   │ 458.3  │  +2.4   │    +1     │    2    │
│  8   │ 461.2  │  +2.9   │    +1     │    3    │
│  9   │ 460.5  │  -0.7   │    -1     │    2    │
│ 10   │ 463.8  │  +3.3   │    +1     │    3    │
└──────┴────────┴─────────┴───────────┴─────────┘

Interpretation:
  AD Line = 3 (positive trend)
  7 up days, 3 down days → Bullish breadth
```

**Buying Pressure Index (BPI)**:
```python
def calculate_bpi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Rolling percentage of up-days.

    Formula:
      BPI_t = (# up days in [t-13, t]) / 14

    Properties:
      - BPI ∈ [0, 1]
      - BPI = 1.0: All days up (extreme bullishness)
      - BPI = 0.5: Balanced (neutral)
      - BPI = 0.0: All days down (extreme bearishness)
      - Mean-reverting: Extremes (>0.8 or <0.2) signal potential reversals
    """
    up = (close.diff() > 0).astype(float)
    return up.rolling(period, min_periods=1).mean()
```

**Example BPI Computation**:
```
QQQ Close (20 days):
┌──────┬────────┬─────────┬────────────┐
│ Day  │ Close  │  Up?    │ BPI (14d)  │
├──────┼────────┼─────────┼────────────┤
│  1   │ 380.0  │   -     │    -       │
│  2   │ 382.5  │   Yes   │  1.000     │
│  3   │ 381.2  │   No    │  0.500     │
│  4   │ 383.8  │   Yes   │  0.667     │
│  5   │ 385.1  │   Yes   │  0.750     │
│  6   │ 384.3  │   No    │  0.600     │
│  7   │ 386.7  │   Yes   │  0.667     │
│  8   │ 388.2  │   Yes   │  0.714     │
│  9   │ 387.5  │   No    │  0.625     │
│ 10   │ 389.8  │   Yes   │  0.667     │
│ 11   │ 391.3  │   Yes   │  0.700     │
│ 12   │ 390.7  │   No    │  0.636     │
│ 13   │ 392.4  │   Yes   │  0.667     │
│ 14   │ 393.9  │   Yes   │  0.692     │
│ 15   │ 395.2  │   Yes   │  0.714     │ ← 10/14 days up
│ 16   │ 394.8  │   No    │  0.643     │
│ 17   │ 396.5  │   Yes   │  0.643     │
│ 18   │ 395.9  │   No    │  0.571     │
│ 19   │ 397.3  │   Yes   │  0.571     │
│ 20   │ 398.7  │   Yes   │  0.571     │
└──────┴────────┴─────────┴────────────┘

Interpretation:
  BPI = 0.571 (moderate bullish bias)
  Not overbought (BPI < 0.8)
```

#### 3.1.2 Cross-Sectional Breadth Builder

```python
def build_breadth_daily(ohlcv_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Vectorized implementation for performance.

    Time complexity: O(N × T) where N=symbols, T=time points
    Space complexity: O(N × T) for direction DataFrame
    """
    direction_frames = []

    # Phase 1: Per-symbol direction computation (parallelizable)
    for df in ohlcv_frames.values():
        if df.empty:
            continue
        df = df.copy().sort_values("date")
        close_col = get_close_column(df)
        direction = np.sign(df[close_col].diff()).fillna(0).astype(int)
        direction_frames.append(
            pd.DataFrame({
                "date": pd.to_datetime(df["date"]),
                "dir": direction.values
            })
        )

    if not direction_frames:
        return pd.DataFrame(columns=[
            "date", "market_breadth_ad", "advancers", "decliners", "total_issues"
        ])

    # Phase 2: Concatenate and aggregate
    all_dirs = pd.concat(direction_frames, ignore_index=True)
    grouped = all_dirs.groupby("date")["dir"]

    stats = pd.DataFrame({
        "advancers": grouped.apply(lambda s: (s > 0).sum()),
        "decliners": grouped.apply(lambda s: (s < 0).sum()),
        "total_issues": grouped.count(),
    })

    # Phase 3: Compute breadth ratio
    stats["market_breadth_ad"] = (
        (stats["advancers"] - stats["decliners"]) /
        stats["total_issues"].clip(lower=1)
    )

    bdf = stats.reset_index().sort_values("date")
    bdf["date"] = pd.to_datetime(bdf["date"])
    return bdf[["date", "market_breadth_ad", "advancers", "decliners", "total_issues"]]
```

**Performance Characteristics**:
```
Benchmark (500 symbols × 1000 days):

Naive approach (loop-based):
  for date in dates:
      for symbol in symbols:
          compute direction
          aggregate
  Time: ~15 seconds

Vectorized approach (current implementation):
  for symbol in symbols:  # Embarrassingly parallel
      compute all directions
  concat and group by date
  Time: ~0.8 seconds (18.75x faster)

Memory usage:
  Naive: 500 × 1000 × 8 bytes = 4 MB (repeated allocations)
  Vectorized: 500,000 × 16 bytes = 8 MB (single allocation)
```

---

## 3.2 Market Covariates System

### Purpose
Provide regime-aware market context features that capture broader market dynamics (equity indices, volatility, rates).

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   COVARIATE CONSTRUCTION PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT: Index OHLCV Data                                                │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  SPY: SPDR S&P 500 ETF (large-cap proxy)                      │    │
│  │  QQQ: Invesco QQQ ETF (tech/growth proxy)                     │    │
│  │  IWM: Russell 2000 ETF (small-cap proxy)                      │    │
│  │  IEF: iShares 7-10Y Treasury Bond ETF (rate proxy)           │    │
│  └─────────────────────────┬──────────────────────────────────────┘    │
│                            │                                             │
│                            ▼                                             │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │         RETURN COMPUTATION (1-day)                              │    │
│  │                                                                 │    │
│  │  For each index:                                               │    │
│  │    ret_1d = (close_t / close_{t-1}) - 1                       │    │
│  │                                                                 │    │
│  │  Example (SPY):                                                │    │
│  │  ┌──────────┬────────┬──────────┐                             │    │
│  │  │   date   │ close  │ ret_1d   │                             │    │
│  │  ├──────────┼────────┼──────────┤                             │    │
│  │  │2025-01-15│ 450.32 │   -      │                             │    │
│  │  │2025-01-16│ 452.18 │ +0.0041  │  (+0.41%)                  │    │
│  │  │2025-01-17│ 449.87 │ -0.0051  │  (-0.51%)                  │    │
│  │  └──────────┴────────┴──────────┘                             │    │
│  │                                                                 │    │
│  │  Output: {spy_ret_1d, qqq_ret_1d, iwm_ret_1d}                 │    │
│  └─────────────────────────┬──────────────────────────────────────┘    │
│                            │                                             │
│                            ▼                                             │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │         VOLATILITY PROXY (Realized Vol)                         │    │
│  │                                                                 │    │
│  │  Compute 21-day annualized realized volatility of SPY:         │    │
│  │                                                                 │    │
│  │  rv21 = std(spy_ret_1d[t-20:t]) × √252 × 100                  │    │
│  │                                                                 │    │
│  │  Properties:                                                   │    │
│  │    - Backward-looking (causal)                                 │    │
│  │    - Annualized (× √252 trading days)                         │    │
│  │    - Scaled to %age (× 100)                                    │    │
│  │    - Comparable to VIX (typical range: 10-50)                 │    │
│  │                                                                 │    │
│  │  Example:                                                      │    │
│  │    std(20 days of returns) = 0.012                            │    │
│  │    rv21 = 0.012 × √252 × 100 = 19.05                         │    │
│  │    Interpretation: ~19% annualized vol (moderate)             │    │
│  │                                                                 │    │
│  │  Stationarity: Compute 1-day change                            │    │
│  │    rv21_chg_1d = (rv21_t / rv21_{t-1}) - 1                    │    │
│  │    Replace inf/-inf with 0, fillna with 0                     │    │
│  └─────────────────────────┬──────────────────────────────────────┘    │
│                            │                                             │
│                            ▼                                             │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │         RATE PROXY (Bond Returns)                               │    │
│  │                                                                 │    │
│  │  IEF return as interest rate sensitivity proxy:                │    │
│  │    ief_ret_1d = (ief_close_t / ief_close_{t-1}) - 1           │    │
│  │                                                                 │    │
│  │  Interpretation:                                               │    │
│  │    ief_ret > 0: Bonds up → Rates down (risk-off)              │    │
│  │    ief_ret < 0: Bonds down → Rates up (risk-on)               │    │
│  │                                                                 │    │
│  │  Why IEF instead of yield curve?                              │    │
│  │    - Direct tradability proxy                                  │    │
│  │    - Stationary (returns vs levels)                            │    │
│  │    - Captures duration risk                                    │    │
│  └─────────────────────────┬──────────────────────────────────────┘    │
│                            │                                             │
│                            ▼                                             │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │         OUTER JOIN & FORWARD FILL                               │    │
│  │                                                                 │    │
│  │  Combine all features with outer join on date:                 │    │
│  │    base = SPY returns                                          │    │
│  │    join QQQ returns (outer)                                    │    │
│  │    join IWM returns (outer)                                    │    │
│  │    join RV21 metrics (left)                                    │    │
│  │    join IEF returns (outer)                                    │    │
│  │                                                                 │    │
│  │  Fill strategy:                                                │    │
│  │    - Returns: NaN → 0.0 (no change)                            │    │
│  │    - Levels (rv21_level): Forward fill                         │    │
│  │    - Changes (rv21_chg_1d): NaN → 0.0                          │    │
│  │                                                                 │    │
│  │  Rationale:                                                    │    │
│  │    - Assumes missing return = flat market                      │    │
│  │    - Volatility persists (forward fill level)                 │    │
│  │    - No change in vol gradient (0 change)                      │    │
│  └─────────────────────────┬──────────────────────────────────────┘    │
│                            │                                             │
│                            ▼                                             │
│  OUTPUT: Canonical Covariates DataFrame                                 │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Schema (CANONICAL_COV_COLS):                                  │    │
│  │    [date, spy_ret_1d, qqq_ret_1d, iwm_ret_1d,                 │    │
│  │     rv21_level, rv21_chg_1d, ief_ret_1d]                      │    │
│  │                                                                 │    │
│  │  Example:                                                      │    │
│  │  ┌──────────┬───────┬───────┬───────┬────────┬────────┬──────┐│    │
│  │  │   date   │ spy_r │ qqq_r │ iwm_r │ rv21_l │rv21_chg│ief_r ││    │
│  │  ├──────────┼───────┼───────┼───────┼────────┼────────┼──────┤│    │
│  │  │2025-01-15│0.0041 │0.0052 │0.0031 │ 18.32  │ 0.0024 │-0.002││    │
│  │  │2025-01-16│-0.0051│-0.0063│-0.0042│ 19.45  │ 0.0617 │0.0031││    │
│  │  │2025-01-17│0.0089 │0.0112 │0.0076 │ 18.97  │-0.0247 │-0.001││    │
│  │  └──────────┴───────┴───────┴───────┴────────┴────────┴──────┘│    │
│  │                                                                 │    │
│  │  Properties:                                                   │    │
│  │    - All stationary (returns or pct changes)                   │    │
│  │    - No look-ahead bias (21-day vol uses past 21 days)        │    │
│  │    - Suitable for Chronos teacher input                        │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Implementation Details

#### 3.2.1 Return Computation Helper

```python
def _prepare_symbol_returns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Extract 1-day returns for a single symbol.

    Steps:
    1. Copy and sort by date
    2. Set date as index (for joining)
    3. Deduplicate dates (keep first)
    4. Get close column (prefer close_adj)
    5. Compute pct_change()
    6. Rename column

    Output: Single-column DataFrame with date index
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy().sort_values("date")
    df = df.set_index("date")
    df = df[~df.index.duplicated(keep="first")]

    close_col = get_close_column(df)
    ret = df[close_col].pct_change().rename(f"{name}_ret_1d")

    return ret.to_frame()
```

**Example Execution**:
```
Input (SPY OHLCV):
┌────────────┬────────┬────────┬────────┬────────┬────────┐
│    date    │  open  │  high  │  low   │ close  │ volume │
├────────────┼────────┼────────┼────────┼────────┼────────┤
│ 2025-01-13 │ 448.20 │ 450.85 │ 447.50 │ 449.32 │  80.2M │
│ 2025-01-14 │ 449.50 │ 451.20 │ 448.90 │ 450.67 │  75.8M │
│ 2025-01-15 │ 450.80 │ 453.10 │ 450.20 │ 452.43 │  82.1M │
└────────────┴────────┴────────┴────────┴────────┴────────┘
                    ↓ _prepare_symbol_returns(df, "spy")
                    ↓
Output:
┌────────────┬────────────┐
│    date    │ spy_ret_1d │
├────────────┼────────────┤
│ 2025-01-13 │    NaN     │  ← First day (no prior)
│ 2025-01-14 │  0.003005  │  ← (450.67/449.32) - 1
│ 2025-01-15 │  0.003907  │  ← (452.43/450.67) - 1
└────────────┴────────────┘
```

#### 3.2.2 Canonical Covariates Builder

```python
def build_canonical_covariates(
    spy_df: pd.DataFrame,
    qqq_df: pd.DataFrame,
    iwm_df: Optional[pd.DataFrame] = None,
    ief_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build stationary covariate features.

    Design choices:
    1. All inputs are returns (stationary)
    2. RV21 level is kept (regime indicator)
    3. RV21 change is computed (stationary gradient)
    4. IEF returns replace price level
    """
    # Step 1: Join equity index returns
    df = _build_base(spy_df, qqq_df, iwm_df)

    # Step 2: Compute realized volatility
    rv21 = (
        df["spy_ret_1d"]
        .rolling(window=21, min_periods=21)
        .std() * np.sqrt(252) * 100
    ).rename("rv21_level")
    df = df.join(rv21, how="left")

    # Step 3: Volatility change (stationary)
    rv21_chg = df["rv21_level"].pct_change().rename("rv21_chg_1d")
    rv21_chg = rv21_chg.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["rv21_chg_1d"] = rv21_chg

    # Step 4: Add bond returns
    if ief_df is not None and not ief_df.empty:
        ief_f = _prepare_symbol_returns(ief_df, "ief")
        df = df.join(ief_f, how="left")
    else:
        df["ief_ret_1d"] = 0.0

    # Step 5: Reset index and fill
    df = df.sort_index().reset_index()
    df = df.rename(columns={"index": "date"})

    # Forward-fill rv21_level (regime persists)
    first_valid = df["rv21_level"].first_valid_index()
    if first_valid is not None:
        first_val = df.loc[first_valid, "rv21_level"]
        df.loc[:first_valid, "rv21_level"] = first_val
    df["rv21_level"] = df["rv21_level"].ffill()

    # Fill returns with 0 (no change assumption)
    for c in ["spy_ret_1d", "qqq_ret_1d", "iwm_ret_1d", "rv21_chg_1d", "ief_ret_1d"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # Step 6: Select canonical columns only
    df = df[CANONICAL_COV_COLS].copy()
    return df
```

**Why Stationary Features**:
```
NON-STATIONARY (Bad for ML):
  Level features have trends/regime shifts
  ┌─────────────────────────────────────┐
  │  VIX Level Over Time                │
  │    │                                 │
  │ 60 │         ████                    │
  │    │        ██  ██                   │
  │ 40 │      ██      ██                 │
  │    │    ██          ██               │
  │ 20 │████              ████████████   │
  │    └─────────────────────────────────│
  │         2020    2021    2022         │
  └─────────────────────────────────────┘
  Regime-dependent: Model trained on 2021 fails in 2020

STATIONARY (Good for ML):
  Changes/returns oscillate around 0
  ┌─────────────────────────────────────┐
  │  VIX Daily Change (%)               │
  │    │                                 │
  │+20 │  █    █                         │
  │    │ ███ ██  █  █   █               │
  │  0 │███████████████████████████████  │ ← Mean ≈ 0
  │    │   █   ████  ███ █               │
  │-20 │     █      █                    │
  │    └─────────────────────────────────│
  │         2020    2021    2022         │
  └─────────────────────────────────────┘
  Distribution stable across time
```

---

## 3.3 MarketFrame Builder

### Purpose
Combine per-symbol OHLCV with market covariates and breadth into a unified panel DataFrame.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MARKETFRAME CONSTRUCTION                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUTS:                                                            │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  1. ohlcv_frames: Dict[str, DataFrame]                     │   │
│  │     {                                                       │   │
│  │       "AAPL": DataFrame[date, open, high, low, close, vol],│   │
│  │       "MSFT": DataFrame[date, open, high, low, close, vol],│   │
│  │       ...                                                   │   │
│  │     }                                                       │   │
│  │                                                             │   │
│  │  2. covariates: DataFrame[date, spy_ret_1d, qqq_ret_1d,...]│   │
│  │                                                             │   │
│  │  3. breadth: DataFrame[date, market_breadth_ad, ...]       │   │
│  │                                                             │   │
│  │  4. trading_days: List[date] (optional, for reindexing)    │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│                             ▼                                        │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │         PER-SYMBOL PROCESSING                               │   │
│  │                                                             │   │
│  │  For each symbol in ohlcv_frames:                          │   │
│  │                                                             │   │
│  │    Step 1: Ensure datetime format                          │   │
│  │      df["date"] = pd.to_datetime(df["date"])               │   │
│  │                                                             │   │
│  │    Step 2: Sort by date                                    │   │
│  │      df = df.sort_values("date")                           │   │
│  │                                                             │   │
│  │    Step 3: Reindex to trading calendar (optional)          │   │
│  │      if trading_days provided:                             │   │
│  │        df = df.set_index("date")                           │   │
│  │                .reindex(trading_days)                       │   │
│  │                .reset_index()                               │   │
│  │      Effect: Fill missing trading days with NaN            │   │
│  │                                                             │   │
│  │    Step 4: Add symbol column                               │   │
│  │      df["symbol"] = symbol                                 │   │
│  │                                                             │   │
│  │    Step 5: Left join covariates (on date)                  │   │
│  │      merged = df.merge(covariates, on="date", how="left")  │   │
│  │                                                             │   │
│  │    Step 6: Left join breadth (on date)                     │   │
│  │      merged = merged.merge(breadth, on="date", how="left") │   │
│  │                                                             │   │
│  │    Append to list                                          │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│                             ▼                                        │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │         CONCATENATION                                       │   │
│  │                                                             │   │
│  │  frames_list = [frame1, frame2, ..., frameN]               │   │
│  │  result = pd.concat(frames_list, ignore_index=True)        │   │
│  │                                                             │   │
│  │  Result: Long-format DataFrame                             │   │
│  │  ┌──────┬───────┬──────┬──────┬──────┬────┬─────┬──────┐  │   │
│  │  │ date │symbol │ open │ high │ low  │close│vol │spy_r │  │   │
│  │  ├──────┼───────┼──────┼──────┼──────┼────┼─────┼──────┤  │   │
│  │  │ D1   │ AAPL  │150.2 │151.8 │149.5 │151.3│45M │0.004 │  │   │
│  │  │ D1   │ MSFT  │380.5 │382.1 │379.8 │381.7│28M │0.004 │  │   │
│  │  │ D1   │ TSLA  │248.3 │252.7 │246.1 │250.9│95M │0.004 │  │   │
│  │  │ D2   │ AAPL  │151.5 │153.2 │150.8 │152.8│52M │-0.003│  │   │
│  │  │ D2   │ MSFT  │381.9 │383.5 │380.2 │382.4│31M │-0.003│  │   │
│  │  │ D2   │ TSLA  │251.2 │254.8 │249.5 │253.1│88M │-0.003│  │   │
│  │  └──────┴───────┴──────┴──────┴──────┴────┴─────┴──────┘  │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  OUTPUT: Unified MarketFrame                                        │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  Shape: (N_symbols × N_days, N_features)                   │   │
│  │  Typical: 500 symbols × 1000 days = 500,000 rows          │   │
│  │  Columns: ~15-20 (OHLCV + covariates + breadth)           │   │
│  └────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
def build_marketframe(
    ohlcv_frames: Dict[str, pd.DataFrame],
    covariates: pd.DataFrame,
    breadth: pd.DataFrame,
    trading_days: Optional[List] = None,
) -> pd.DataFrame:
    """
    Construct panel DataFrame aligned to trading calendar.

    Why long format (not wide/pivot)?
    - Efficient storage (no sparse matrix)
    - Groupby operations (per-symbol features)
    - Polars compatibility (columnar operations)
    - ML-ready (each row = single observation)
    """
    cov = ensure_datetime(covariates.copy())
    brd = ensure_datetime(breadth.copy())

    frames: List[pd.DataFrame] = []

    for symbol, ohlcv in ohlcv_frames.items():
        if ohlcv.empty:
            continue

        df = ensure_datetime(ohlcv.copy())
        df = df.sort_values("date")

        # Optional: Align to trading calendar
        if trading_days is not None:
            idx = pd.DatetimeIndex(trading_days)
            df = df.set_index("date").reindex(idx).reset_index()
            df = df.rename(columns={"index": "date"})

        df["symbol"] = symbol

        # Enrich with market context
        merged = df.merge(cov, on="date", how="left")
        merged = merged.merge(brd, on="date", how="left")

        frames.append(merged)

    if not frames:
        raise ValueError("MarketFrame build produced no data.")

    return pd.concat(frames, ignore_index=True)
```

**Why This Matters**:
```
DOWNSTREAM USAGE:

1. Feature Engineering:
   marketframe
     .group_by(["symbol"])
     .apply(compute_rolling_features)

2. Cross-sectional Analysis:
   marketframe
     .filter(pl.col("date") == "2025-01-15")
     .select(["symbol", "close", "spy_ret_1d"])

3. Training Dataset:
   marketframe → SelectorFeatureBuilder → Training set

4. Backtest:
   marketframe → Portfolio simulator → P&L curve
```

*To be continued in SYSTEM_ANALYSIS_PART4.md (Feature Engineering & Model Architecture)...*

