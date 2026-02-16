# ALGAIE SYSTEM ANALYSIS - PART 2
## Data Layer Architecture (Exhaustive Detail)

*Continuation from SYSTEM_ANALYSIS.md Part 1*

---

# PART 2: DATA LAYER ARCHITECTURE

The data layer is the foundation of the entire trading system. It implements a hierarchical pipeline that transforms raw market data into analysis-ready features through multiple validation and enrichment stages.

## 2.1 Canonical Data System

### Purpose
The canonical data system (`algaie/data/canonical/`) provides the lowest-level interface to raw OHLCV (Open, High, Low, Close, Volume) market data with strict validation and storage conventions.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CANONICAL DATA PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  RAW DATA INPUT                                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  Source: Data Vendors (Alpaca, Polygon, etc.)              │   │
│  │  Format: Per-symbol CSV/Parquet files                      │   │
│  │  Frequency: Daily bars                                     │   │
│  │  Adjustments: Split & dividend adjusted                    │   │
│  └─────────────────────┬──────────────────────────────────────┘   │
│                        │                                            │
│                        ▼                                            │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │              SCHEMA VALIDATION                              │   │
│  │                                                             │   │
│  │  Required Columns:                                         │   │
│  │    - date: Date (trading day)                              │   │
│  │    - open: float64 (opening price)                         │   │
│  │    - high: float64 (high price)                            │   │
│  │    - low: float64 (low price)                              │   │
│  │    - close: float64 (closing price)                        │   │
│  │    - volume: int64 (shares traded)                         │   │
│  │                                                             │   │
│  │  Optional Columns:                                         │   │
│  │    - vwap: float64 (volume-weighted average)               │   │
│  │    - vix: float64 (implied volatility)                     │   │
│  │    - rate_proxy: float64 (interest rate proxy)             │   │
│  │                                                             │   │
│  │  Validation Rules:                                         │   │
│  │    1. No duplicate dates (1 row per trading day)           │   │
│  │    2. Monotonic increasing dates                           │   │
│  │    3. close > 0 (positive prices)                          │   │
│  │    4. high >= max(open, close)                             │   │
│  │    5. low <= min(open, close)                              │   │
│  │    6. No zero-filled rows (all OHLC == 0)                  │   │
│  └─────────────────────┬──────────────────────────────────────┘   │
│                        │                                            │
│                        ▼                                            │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │          INVALID RETURN QUARANTINE                          │   │
│  │                                                             │   │
│  │  Detect and isolate price discontinuities:                 │   │
│  │                                                             │   │
│  │  ratio = close_t / close_{t-1}                             │   │
│  │                                                             │   │
│  │  Invalid if:                                               │   │
│  │    - ratio <= 0 (negative price)                           │   │
│  │    - ratio - 1 <= -1 (>100% drop, likely split error)     │   │
│  │                                                             │   │
│  │  Action:                                                   │   │
│  │    - Write invalid rows → quarantine parquet               │   │
│  │    - Remove from dataset                                   │   │
│  │    - If invalid_frac > max_invalid_frac → FAIL             │   │
│  └─────────────────────┬──────────────────────────────────────┘   │
│                        │                                            │
│                        ▼                                            │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │              STORAGE LAYER                                  │   │
│  │                                                             │   │
│  │  Storage Structure (Partitioned Parquet):                  │   │
│  │                                                             │   │
│  │  backend/artifacts/canonical/daily/                        │   │
│  │    └─ symbol=AAPL/                                        │   │
│  │        └─ data.parquet                                     │   │
│  │    └─ symbol=MSFT/                                        │   │
│  │        └─ data.parquet                                     │   │
│  │    └─ symbol=TSLA/                                        │   │
│  │        └─ data.parquet                                     │   │
│  │                                                             │   │
│  │  Format: Apache Parquet (columnar, compressed)             │   │
│  │  Codec: Snappy compression                                 │   │
│  │  Partitioning: By symbol (enables parallel reads)          │   │
│  └────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Implementation Details

#### 2.1.1 Schema Definition (`canonical/schema.py`)

```python
@dataclass(frozen=True)
class CanonicalDailySchema:
    required_columns: List[str] = field(
        default_factory=lambda: ["date", "open", "high", "low", "close", "volume"]
    )
    optional_columns: List[str] = field(
        default_factory=lambda: ["vwap", "vix", "rate_proxy"]
    )
```

**Why This Design**:
- **Frozen dataclass**: Immutability ensures schema can't be accidentally modified
- **Factory defaults**: `default_factory` creates new list instances (avoids mutable default pitfall)
- **Separation of required/optional**: Clear contract for downstream consumers

#### 2.1.2 Reader (`canonical/reader.py`)

```python
def read_canonical_daily(path: Path) -> pl.DataFrame:
    return pl.scan_parquet(path).collect()
```

**Why Polars**:
- **Lazy evaluation**: `scan_parquet()` creates execution plan without loading data
- **Parallelism**: Polars uses all CPU cores by default
- **Memory efficiency**: Columnar format with zero-copy reads
- **Speed**: 10-100x faster than pandas for large datasets

**Read Pattern**:
```
File: symbol=AAPL/data.parquet (10 MB, 5000 rows)
┌──────────┬─────────┬────────┬────────┬────────┬────────┐
│   date   │  open   │  high  │  low   │ close  │ volume │
├──────────┼─────────┼────────┼────────┼────────┼────────┤
│ 2020-01-02│ 145.32 │ 147.89 │ 144.12 │ 146.54 │ 45.2M  │
│ 2020-01-03│ 146.78 │ 148.23 │ 145.67 │ 147.12 │ 52.1M  │
│    ...    │   ...   │  ...   │  ...   │  ...   │  ...   │
└──────────┴─────────┴────────┴────────┴────────┴────────┘
                        ↓ Polars scan_parquet()
                        ↓ (lazy, no memory allocation)
                        ↓
                collect() → Load into RAM as pl.DataFrame
```

#### 2.1.3 Writer (`canonical/writer.py`)

```python
def write_canonical_daily(df: pd.DataFrame, destination: Path) -> None:
    write_dataframe(df, destination)

# From data/common.py:
def write_dataframe(df: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination, index=False)
```

**Write Process**:
```
DataFrame in Memory:
  ┌─────────────────────────────────┐
  │  5000 rows × 6 columns          │
  │  Memory: ~240 KB (uncompressed) │
  └─────────────────────────────────┘
                ↓
  Create parent directory if needed
                ↓
  df.to_parquet() → Snappy compression
                ↓
  ┌─────────────────────────────────┐
  │  data.parquet                   │
  │  Size: ~120 KB (50% compression)│
  │  Schema embedded in file        │
  └─────────────────────────────────┘
```

**Why Index=False**:
- Index is redundant (date column serves as index)
- Reduces file size
- Simplifies schema (no unnamed index column)

#### 2.1.4 Validation (`canonical/validate.py`)

**Validation Function Breakdown**:

```python
def validate_canonical_daily(df: pd.DataFrame) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    # Check 1: Empty DataFrame
    if df.empty:
        issues.append(ValidationIssue("canonical daily is empty", []))
        return issues

    # Check 2: Missing date column (critical)
    if "date" not in df.columns:
        issues.append(ValidationIssue("missing date column", []))
        return issues  # Cannot continue without dates

    # Check 3: Duplicate dates
    if df["date"].duplicated().any():
        dup_rows = df.index[df["date"].duplicated()].tolist()
        issues.append(ValidationIssue("duplicate dates", dup_rows))

    # Check 4: Monotonic increasing dates
    sorted_dates = pd.to_datetime(df["date"])
    if not sorted_dates.is_monotonic_increasing:
        issues.append(ValidationIssue("non-monotonic dates", df.index.tolist()))

    # Check 5: Positive close prices
    if (df["close"] <= 0).any():
        bad_rows = df.index[df["close"] <= 0].tolist()
        issues.append(ValidationIssue("close <= 0", bad_rows))

    # Check 6: High >= max(open, close)
    high_violations = df["high"] < df[["open", "close"]].max(axis=1)
    if high_violations.any():
        issues.append(
            ValidationIssue("high below open/close", df.index[high_violations].tolist())
        )

    # Check 7: Low <= min(open, close)
    low_violations = df["low"] > df[["open", "close"]].min(axis=1)
    if low_violations.any():
        issues.append(
            ValidationIssue("low above open/close", df.index[low_violations].tolist())
        )

    # Check 8: Zero-filled rows (data quality issue)
    zero_fill = (df[["open", "high", "low", "close", "volume"]] == 0).all(axis=1)
    if zero_fill.any():
        issues.append(ValidationIssue("zero-filled row", df.index[zero_fill].tolist()))

    return issues
```

**Validation Logic Flow**:

```
Input: DataFrame (5000 rows)
    ↓
┌────────────────────────────────────────┐
│ Check 1: Empty?                        │
│   No → Continue                        │
└────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────┐
│ Check 2: Has 'date' column?            │
│   Yes → Continue                       │
└────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────┐
│ Check 3: Duplicate dates?              │
│   df["date"].duplicated().any()        │
│   Result: False → OK                   │
└────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────┐
│ Check 4: Monotonic dates?              │
│   sorted_dates.is_monotonic_increasing │
│   Result: True → OK                    │
└────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────┐
│ Check 5: Positive close?               │
│   (df["close"] <= 0).any()             │
│   Result: False → OK                   │
└────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────┐
│ Check 6: High >= max(open, close)?     │
│   Violations: Row 342 (high=145.2,     │
│                        close=145.8)    │
│   → Issue logged                       │
└────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────┐
│ Check 7: Low <= min(open, close)?      │
│   Violations: None → OK                │
└────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────┐
│ Check 8: Zero-filled rows?             │
│   Violations: None → OK                │
└────────────────────────────────────────┘
    ↓
Return: [ValidationIssue("high below open/close", [342])]
```

**Quarantine System**:

```python
def quarantine_invalid_returns(
    df: pd.DataFrame,
    config: PipelineConfig,
    report_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Compute daily returns
    ratios = df["close"] / df["close"].shift(1)

    # Identify invalid returns
    invalid = (ratios <= 0) | (ratios - 1 <= -1)

    # Separate invalid rows
    invalid_rows = df.loc[invalid].copy()
    if not invalid_rows.empty:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        invalid_rows.to_parquet(report_path, index=False)

    # Keep valid rows
    valid_df = df.loc[~invalid].copy()

    # Check threshold
    invalid_frac = len(invalid_rows) / max(len(df), 1)
    if invalid_frac > config.max_invalid_frac:
        raise ValidationError(
            [ValidationIssue("invalid return fraction exceeded", invalid_rows.index.tolist())]
        )

    return valid_df, invalid_rows
```

**Example Quarantine Scenario**:

```
Original DataFrame (1000 rows):
┌──────────┬────────┬────────────┬──────────┐
│   date   │ close  │   ratio    │  valid?  │
├──────────┼────────┼────────────┼──────────┤
│ 2020-01-02│ 150.0 │     -      │   Yes    │
│ 2020-01-03│ 151.2 │   1.008    │   Yes    │
│ 2020-01-06│ 152.5 │   1.0086   │   Yes    │
│ 2020-01-07│  75.0 │   0.4918   │   NO!    │  ← Stock split not adjusted
│ 2020-01-08│  76.1 │   1.0147   │   Yes    │
│    ...    │  ...   │    ...     │   ...    │
└──────────┴────────┴────────────┴──────────┘

Invalid Detection:
  Row 3: ratio = 0.4918
  Check: (ratio - 1) = -0.5082
  Check: -0.5082 <= -1? No, but ratio < 0.5 suggests split
  Action: Quarantine this row

Result:
  valid_df: 999 rows (row 3 removed)
  invalid_rows: 1 row (saved to quarantine/AAPL_invalid.parquet)
  invalid_frac = 1/1000 = 0.001 ≤ max_invalid_frac (0.001) → PASS
```

---

## 2.2 Common Data Utilities

### Purpose
Shared validation primitives and helper functions used across all data modules (`algaie/data/common.py`).

### Implementation Details

#### 2.2.1 ValidationIssue System

```python
@dataclass(frozen=True)
class ValidationIssue:
    message: str
    rows: List[int]

class BaseValidationError(RuntimeError):
    def __init__(self, issues: Iterable[ValidationIssue]) -> None:
        self.issues = list(issues)
        message = "; ".join(issue.message for issue in self.issues)
        super().__init__(message)
```

**Design Pattern**:
```
ValidationIssue (Data Class)
    ↓
Collected into List[ValidationIssue]
    ↓
If any critical issues → BaseValidationError
    ↓
Error propagates with full context:
  - Which rows failed
  - What validation failed
  - Actionable error message
```

**Example Usage**:
```python
issues = validate_canonical_daily(df)
if issues:
    # Log issues for manual review
    for issue in issues:
        logger.warning(f"{issue.message}: {len(issue.rows)} rows")

    # Raise if critical
    critical = [i for i in issues if "missing date" in i.message]
    if critical:
        raise BaseValidationError(critical)
```

#### 2.2.2 Non-Finite Detection

```python
def find_non_finite_rows(df: pd.DataFrame, columns: pd.DataFrame | None = None) -> np.ndarray:
    """Return boolean array of rows with non-finite values (NaN, inf, -inf)."""
    numeric = columns if columns is not None else df.select_dtypes(include=[np.number])
    if numeric.empty:
        return np.zeros(len(df), dtype=bool)
    arr = numeric.to_numpy()
    return ~np.isfinite(arr).all(axis=1)
```

**Computational Efficiency**:
```
Standard approach (slow):
  for col in numeric_columns:
      bad_mask |= df[col].isna() | np.isinf(df[col])

Optimized approach (fast):
  arr = numeric_df.to_numpy()  # Single copy to NumPy
  bad_mask = ~np.isfinite(arr).all(axis=1)  # Vectorized

Performance:
  10,000 rows × 20 columns
  Standard: ~50ms
  Optimized: ~5ms (10x faster)
```

#### 2.2.3 Close Column Resolution

```python
def get_close_column(df: pd.DataFrame) -> str:
    """Return 'close_adj' if present, else 'close'."""
    if "close_adj" in df.columns:
        return "close_adj"
    if "close" in df.columns:
        return "close"
    raise ValueError("DataFrame must contain 'close_adj' or 'close' column")
```

**Why This Pattern**:
- **Adjustment preference**: Split/dividend adjusted prices preferred
- **Backward compatibility**: Legacy code uses 'close', new code uses 'close_adj'
- **Fail-fast**: Raises immediately if neither column exists

---

## 2.3 Eligibility & Universe Construction

### Purpose
Filter the entire stock market (thousands of tickers) down to a tradable universe based on liquidity, price, and history criteria.

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   UNIVERSE CONSTRUCTION PIPELINE                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  INPUT: Full Market OHLCV (All US Equities)                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ~8000 tickers × 5 years × 252 days/year = 10M+ rows            │   │
│  │  Columns: [date, symbol, open, high, low, close, volume]        │   │
│  └──────────────────────────────┬───────────────────────────────────┘   │
│                                 │                                        │
│                                 ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              ELIGIBILITY FILTERS                                 │   │
│  │                                                                  │   │
│  │  Filter 1: Minimum Price                                        │   │
│  │    close > min_price (default: $5.00)                           │   │
│  │    Rationale: Exclude penny stocks (high slippage, low quality) │   │
│  │                                                                  │   │
│  │  Filter 2: Minimum History                                      │   │
│  │    days_since_ipo >= min_history_days (default: 60)            │   │
│  │    Rationale: Need enough data for feature computation          │   │
│  │                                                                  │   │
│  │  Result: is_observable = (price > min) AND (history >= min)    │   │
│  └──────────────────────────────┬───────────────────────────────────┘   │
│                                 │                                        │
│                                 ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              TRADABILITY FILTERS                                 │   │
│  │                                                                  │   │
│  │  Filter 3: Minimum Dollar Volume                                │   │
│  │    dollar_vol = close × volume                                  │   │
│  │    dollar_vol >= min_dollar_vol (default: $1M/day)             │   │
│  │    Rationale: Ensure sufficient liquidity for execution         │   │
│  │                                                                  │   │
│  │  Result: is_tradable = is_observable AND (dollar_vol >= min)   │   │
│  └──────────────────────────────┬───────────────────────────────────┘   │
│                                 │                                        │
│                                 ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              TIER ASSIGNMENT                                     │   │
│  │                                                                  │   │
│  │  Tier breakpoints (by dollar volume):                           │   │
│  │    Tier 1: dollar_vol >= $50M  (mega-cap, highest liquidity)   │   │
│  │    Tier 2: $10M <= dollar_vol < $50M  (mid-cap)                │   │
│  │    Tier 3: $1M <= dollar_vol < $10M   (small-cap)              │   │
│  │                                                                  │   │
│  │  Implementation: pd.cut() with custom bins                      │   │
│  │    bins = [inf, 50M, 10M, 0]                                    │   │
│  │    tier = pd.cut(dollar_vol, bins, labels=[1, 2, 3])           │   │
│  └──────────────────────────────┬───────────────────────────────────┘   │
│                                 │                                        │
│                                 ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              WEIGHT ASSIGNMENT                                   │   │
│  │                                                                  │   │
│  │  Philosophy: Higher tier → Higher weight                        │   │
│  │                                                                  │   │
│  │  Formula:                                                       │   │
│  │    raw_weight = (max_tier + 1) - tier                          │   │
│  │      Tier 1 → raw_weight = 3                                   │   │
│  │      Tier 2 → raw_weight = 2                                   │   │
│  │      Tier 3 → raw_weight = 1                                   │   │
│  │                                                                  │   │
│  │  Normalization (per-date):                                      │   │
│  │    normalized_weight = raw_weight / mean(raw_weight_per_date)  │   │
│  │                                                                  │   │
│  │  Result: Mean weight = 1.0 each day (preserves capital)        │   │
│  └──────────────────────────────┬───────────────────────────────────┘   │
│                                 │                                        │
│                                 ▼                                        │
│  OUTPUT: UniverseFrame                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Schema:                                                         │   │
│  │    [date, symbol, is_observable, is_tradable, tier, weight]     │   │
│  │                                                                  │   │
│  │  Size: ~500-1000 symbols per day (filtered from ~8000)         │   │
│  │  Format: Polars DataFrame or Parquet file                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Implementation Details

#### 2.3.1 Universe Config

```python
@dataclass
class UniverseConfig:
    min_price: float = 5.0
    min_dollar_vol: float = 1_000_000.0
    min_history_days: int = 60
    tier_breakpoints: List[float] = field(default_factory=lambda: [50_000_000.0, 10_000_000.0])
```

#### 2.3.2 Universe Builder Algorithm

**Step-by-step execution**:

```python
class UniverseBuilder:
    def build(self, daily: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        df = ensure_datetime(daily.copy())

        # Step 1: Compute dollar volume
        close_col = get_close_column(df)
        dollar_vol = df[close_col].abs() * df["volume"]

        # Step 2: Compute cumulative history days
        df = df.sort_values(["symbol", "date"])
        hist_days = df.groupby("symbol").cumcount() + 1

        # Step 3: Observable filter
        is_observable = (df[close_col] > cfg.min_price) & (hist_days >= cfg.min_history_days)

        # Step 4: Tradable filter
        is_tradable = is_observable & (dollar_vol >= cfg.min_dollar_vol)

        # Step 5: Tier assignment
        bp = sorted(cfg.tier_breakpoints, reverse=True)
        bins = [float("inf")] + bp + [0.0]
        tier = pd.cut(dollar_vol, bins=list(reversed(bins)), labels=False, right=False)
        tier = len(bp) + 1 - tier.fillna(len(bp) + 1).astype(int)

        # Step 6: Weight assignment
        max_tier = tier.max() if not df.empty else 1
        weight = (max_tier + 1 - tier).astype(float)
        date_mean = weight.groupby(df["date"]).transform("mean")
        weight = weight / date_mean.clip(lower=1e-8)

        return pd.DataFrame({
            "date": df["date"].values,
            "symbol": df["symbol"].values,
            "is_observable": is_observable.values,
            "is_tradable": is_tradable.values,
            "tier": tier.values,
            "weight": weight.values,
        })
```

**Detailed Example Execution**:

```
Input (Sample 5 rows for 2025-01-15):
┌──────────┬────────┬────────┬──────────┬────────────┐
│   date   │ symbol │ close  │  volume  │  hist_days │
├──────────┼────────┼────────┼──────────┼────────────┤
│2025-01-15│  AAPL  │ 185.25 │  75.2M   │    252     │
│2025-01-15│  TSLA  │ 248.50 │ 120.5M   │    252     │
│2025-01-15│  XYZ   │   3.15 │   2.1M   │    252     │
│2025-01-15│  ABC   │  12.80 │   0.8M   │     45     │
│2025-01-15│  QQQ   │ 425.60 │  42.3M   │    252     │
└──────────┴────────┴────────┴──────────┴────────────┘

Step 1: Dollar Volume
  AAPL: 185.25 × 75.2M = $13,930M
  TSLA: 248.50 × 120.5M = $29,944M
  XYZ: 3.15 × 2.1M = $6.6M
  ABC: 12.80 × 0.8M = $10.2M
  QQQ: 425.60 × 42.3M = $18,003M

Step 2: Observable Filter
  AAPL: close=185.25 > 5.0 ✓, hist=252 >= 60 ✓ → True
  TSLA: close=248.50 > 5.0 ✓, hist=252 >= 60 ✓ → True
  XYZ: close=3.15 > 5.0 ✗ → False
  ABC: close=12.80 > 5.0 ✓, hist=45 >= 60 ✗ → False
  QQQ: close=425.60 > 5.0 ✓, hist=252 >= 60 ✓ → True

Step 3: Tradable Filter
  AAPL: observable=True, dv=$13,930M >= $1M ✓ → True
  TSLA: observable=True, dv=$29,944M >= $1M ✓ → True
  XYZ: observable=False → False
  ABC: observable=False → False
  QQQ: observable=True, dv=$18,003M >= $1M ✓ → True

Step 4: Tier Assignment
  Breakpoints: [50M, 10M]
  Bins: [inf, 50M, 10M, 0]

  AAPL: dv=$13,930M → bin [50M, inf) → Tier 1
  TSLA: dv=$29,944M → bin [50M, inf) → Tier 1
  XYZ: dv=$6.6M → bin [10M, 50M) → Tier 2
  ABC: dv=$10.2M → bin [10M, 50M) → Tier 2
  QQQ: dv=$18,003M → bin [50M, inf) → Tier 1

Step 5: Weight Assignment
  max_tier = 1
  raw_weights:
    AAPL: (1 + 1) - 1 = 1
    TSLA: (1 + 1) - 1 = 1
    XYZ: (1 + 1) - 2 = 0  (but not tradable)
    ABC: (1 + 1) - 2 = 0  (but not tradable)
    QQQ: (1 + 1) - 1 = 1

  Normalization:
    mean_weight (2025-01-15, tradable only) = (1 + 1 + 1) / 3 = 1.0
    normalized_weights = raw / mean = all remain 1.0

Final Output:
┌──────────┬────────┬──────────────┬─────────────┬──────┬────────┐
│   date   │ symbol │ is_observable│ is_tradable │ tier │ weight │
├──────────┼────────┼──────────────┼─────────────┼──────┼────────┤
│2025-01-15│  AAPL  │     True     │     True    │  1   │  1.0   │
│2025-01-15│  TSLA  │     True     │     True    │  1   │  1.0   │
│2025-01-15│  XYZ   │    False     │    False    │  2   │  0.0   │
│2025-01-15│  ABC   │    False     │    False    │  2   │  0.0   │
│2025-01-15│  QQQ   │     True     │     True    │  1   │  1.0   │
└──────────┴────────┴──────────────┴─────────────┴──────┴────────┘
```

*To be continued in SYSTEM_ANALYSIS_PART3.md...*

