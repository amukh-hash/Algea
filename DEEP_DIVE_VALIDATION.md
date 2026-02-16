# DEEP DIVE: Validation System Architecture
## Complete Technical Analysis

*Deep-dive supplement to SYSTEM_ANALYSIS series*

---

## PURPOSE OF THIS DOCUMENT

This document provides an exhaustive analysis of the validation system - the critical infrastructure that ensures data quality, catches errors early, and prevents corrupt data from propagating through the pipeline.

**Why Validation Matters**:
- Trading systems operate on financial data → errors = monetary loss
- Bad data compounds through pipeline (garbage in, garbage out)
- Silent failures are worse than loud crashes
- Validation cost << cost of trading on bad signals

---

## VALIDATION ARCHITECTURE

### Layered Validation Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│                    VALIDATION LAYER HIERARCHY                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Layer 1: SCHEMA VALIDATION (Structural)                            │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Purpose: Ensure correct columns and types exist           │    │
│  │  When: Immediately after data load                         │    │
│  │  Cost: O(1) - constant time (just check metadata)          │    │
│  │                                                             │    │
│  │  Checks:                                                    │    │
│  │    ✓ Required columns present                              │    │
│  │    ✓ Column data types correct                             │    │
│  │    ✓ Optional columns if present have correct types        │    │
│  │                                                             │    │
│  │  Example Failure:                                           │    │
│  │    Missing "date" column → FAIL FAST                        │    │
│  │    "volume" is float instead of int → FAIL                  │    │
│  └────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  Layer 2: SEMANTIC VALIDATION (Logical)                             │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Purpose: Ensure data makes logical sense                  │    │
│  │  When: After schema validation                             │    │
│  │  Cost: O(N) - linear scan through rows                     │    │
│  │                                                             │    │
│  │  Checks:                                                    │    │
│  │    ✓ No duplicate dates (uniqueness)                       │    │
│  │    ✓ Monotonic increasing dates (ordering)                 │    │
│  │    ✓ Positive prices (domain constraint)                   │    │
│  │    ✓ OHLC relationships (high ≥ max(open, close), etc.)    │    │
│  │    ✓ No zero-filled rows (data quality)                    │    │
│  │                                                             │    │
│  │  Example Failure:                                           │    │
│  │    close = -5.00 → FAIL (negative price impossible)        │    │
│  │    high = 145 but close = 150 → FAIL (logical violation)   │    │
│  └────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  Layer 3: STATISTICAL VALIDATION (Anomaly Detection)                │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Purpose: Detect statistical anomalies and data quality    │    │
│  │  When: After semantic validation                           │    │
│  │  Cost: O(N) - requires computing statistics                │    │
│  │                                                             │    │
│  │  Checks:                                                    │    │
│  │    ✓ Return validity (no >100% single-day drops)           │    │
│  │    ✓ Outlier detection (3-sigma or IQR method)             │    │
│  │    ✓ Missing value patterns                                │    │
│  │    ✓ Suspicious repetitions (same value 10+ days)          │    │
│  │                                                             │    │
│  │  Example Failure:                                           │    │
│  │    Return = -99.5% → Likely unadjusted stock split         │    │
│  │    close = 150.00 for 30 consecutive days → Suspended?     │    │
│  └────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  Layer 4: CROSS-SECTIONAL VALIDATION (Relative Checks)              │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Purpose: Ensure consistency across symbols                │    │
│  │  When: After batch processing all symbols                  │    │
│  │  Cost: O(N × M) - N symbols, M time points                 │    │
│  │                                                             │    │
│  │  Checks:                                                    │    │
│  │    ✓ Coverage consistency (all symbols have same dates)    │    │
│  │    ✓ Correlation sanity (SPY vs QQQ should be high)        │    │
│  │    ✓ Breadth thresholds (min symbols per date)             │    │
│  │    ✓ Feature distribution checks (outlier stocks)          │    │
│  │                                                             │    │
│  │  Example Failure:                                           │    │
│  │    Only 50 symbols on 2025-01-15 (expected ~500)           │    │
│  │    SPY and QQQ have correlation = -0.8 (impossible)        │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## VALIDATION ISSUE PATTERN

### Design Philosophy

**Problem**: Simple boolean validation (pass/fail) loses information.

**Solution**: Structured issue tracking with row-level detail.

```python
@dataclass(frozen=True)
class ValidationIssue:
    """
    Immutable record of a single validation failure.

    Fields:
        message: Human-readable description
        rows: List of affected row indices

    Why frozen?
        - Issues should never be modified after creation
        - Enables hashing and set operations
        - Thread-safe (can validate in parallel)
    """
    message: str
    rows: List[int]
```

### Usage Pattern

```python
# Collect issues instead of failing immediately
issues: List[ValidationIssue] = []

# Check 1: Duplicate dates
if df["date"].duplicated().any():
    dup_rows = df.index[df["date"].duplicated()].tolist()
    issues.append(ValidationIssue("duplicate dates", dup_rows))

# Check 2: Negative prices
if (df["close"] <= 0).any():
    bad_rows = df.index[df["close"] <= 0].tolist()
    issues.append(ValidationIssue("close <= 0", bad_rows))

# ... more checks ...

# Final decision
if issues:
    # Log all issues for debugging
    for issue in issues:
        logger.warning(f"{issue.message}: rows {issue.rows}")

    # Classify severity
    critical = [i for i in issues if "duplicate" in i.message or "close <=" in i.message]

    if critical:
        raise BaseValidationError(critical)
    else:
        # Warning-level issues - log but continue
        pass
```

**Benefits**:

1. **Complete Error Context**:
   ```
   Bad: "Validation failed"
   Good: "3 validation failures:
          - duplicate dates: rows [42, 103]
          - close <= 0: row [157]
          - high below close: rows [89, 92, 201]"
   ```

2. **Batch Processing**:
   ```
   Inefficient: Check 1 → fail → fix → check 2 → fail → fix ...
   Efficient: Run all checks → report all issues at once → fix all
   ```

3. **Selective Handling**:
   ```python
   # Separate critical from warning
   critical_issues = [i for i in issues if i.message in CRITICAL_CHECKS]
   warning_issues = [i for i in issues if i.message in WARNING_CHECKS]

   # Fail on critical, log warnings
   if critical_issues:
       raise BaseValidationError(critical_issues)
   for warning in warning_issues:
       logger.warning(warning.message)
   ```

---

## CANONICAL DATA VALIDATION (8-STAGE PIPELINE)

### Stage 1: Empty Check

```python
if df.empty:
    issues.append(ValidationIssue("canonical daily is empty", []))
    return issues  # Cannot continue
```

**Why First**:
- Fastest check (O(1))
- Prevents null pointer / empty array errors in subsequent checks
- Early return saves computation

**Real-world Trigger**:
- Data vendor API returns empty response
- Network timeout → incomplete download
- Incorrect date filter → no rows match

---

### Stage 2: Date Column Check

```python
if "date" not in df.columns:
    issues.append(ValidationIssue("missing date column", []))
    return issues  # Cannot continue without dates
```

**Why Critical**:
- All subsequent checks require date column
- Time-series operations depend on temporal ordering
- Fail-fast prevents cryptic downstream errors

**Real-world Trigger**:
- Column renamed in data source (e.g., "Date" vs "date")
- Schema migration not applied
- CSV parsing error (delimiter mismatch)

---

### Stage 3: Duplicate Dates

```python
if df["date"].duplicated().any():
    dup_rows = df.index[df["date"].duplicated()].tolist()
    issues.append(ValidationIssue("duplicate dates", dup_rows))
```

**Algorithm Detail**:
```
Input DataFrame (5 rows):
┌────┬──────────┬────────┐
│idx │   date   │ close  │
├────┼──────────┼────────┤
│ 0  │2025-01-13│ 150.0  │
│ 1  │2025-01-14│ 151.5  │
│ 2  │2025-01-15│ 149.8  │
│ 3  │2025-01-14│ 151.5  │  ← Duplicate!
│ 4  │2025-01-16│ 152.3  │
└────┴──────────┴────────┘

Step 1: df["date"].duplicated()
  Returns boolean mask [False, False, False, True, False]
  (First occurrence = False, subsequent = True)

Step 2: .any()
  Returns True (at least one duplicate exists)

Step 3: Extract indices
  df.index[mask] = [3]
  dup_rows = [3]

Result:
  ValidationIssue("duplicate dates", [3])
```

**Why Problematic**:
- Forward returns ambiguous (which close to use?)
- Feature computation breaks (rolling windows double-count)
- Training labels misaligned

**Remediation**:
```python
# Keep first occurrence
df = df[~df["date"].duplicated(keep="first")]

# Or keep last occurrence (more recent data)
df = df[~df["date"].duplicated(keep="last")]
```

---

### Stage 4: Monotonic Dates

```python
sorted_dates = pd.to_datetime(df["date"])
if not sorted_dates.is_monotonic_increasing:
    issues.append(ValidationIssue("non-monotonic dates", df.index.tolist()))
```

**Algorithm Detail**:
```
Input (unsorted):
┌──────────┬────────┐
│   date   │ close  │
├──────────┼────────┤
│2025-01-13│ 150.0  │
│2025-01-15│ 149.8  │  ← Out of order!
│2025-01-14│ 151.5  │
│2025-01-16│ 152.3  │
└──────────┴────────┘

is_monotonic_increasing = False

Why it matters:
  shift(1) assumes previous row = previous day
  If unsorted:
    2025-01-15 close = 149.8
    shift(1) gives 2025-01-14 close = 151.5
    return = (149.8 / 151.5) - 1 = -1.12%  (WRONG!)

  Correct chronological:
    2025-01-14 → 2025-01-15
    return = (149.8 / 151.5) - 1 = -1.12%  (CORRECT)
```

**Remediation**:
```python
df = df.sort_values("date").reset_index(drop=True)
```

---

### Stage 5: Positive Close Prices

```python
if (df["close"] <= 0).any():
    bad_rows = df.index[df["close"] <= 0].tolist()
    issues.append(ValidationIssue("close <= 0", bad_rows))
```

**Failure Modes**:

```
Case 1: Negative Price
┌──────────┬────────┐
│   date   │ close  │
├──────────┼────────┤
│2025-01-15│ -5.23  │  ← Impossible
└──────────┴────────┘

Causes:
  - Data entry error
  - Sign error in calculation
  - Corrupted data feed

Case 2: Zero Price
┌──────────┬────────┐
│   date   │ close  │
├──────────┼────────┤
│2025-01-15│  0.00  │  ← Stock delisted or suspended
└──────────┴────────┘

Causes:
  - Stock delisted (bankruptcy)
  - Trading halted (no close price)
  - Missing data filled with 0
```

**Impact on Calculations**:
```python
# Log return formula:
log_return = np.log(close_t / close_t_minus_1)

# If close_t = 0:
log_return = np.log(0 / 150) = np.log(0) = -inf  (ERROR!)

# If close_t < 0:
log_return = np.log(-5 / 150) = np.log(negative) = NaN  (ERROR!)
```

---

### Stage 6: High/Low Constraints

```python
# High should be >= max(open, close)
high_violations = df["high"] < df[["open", "close"]].max(axis=1)
if high_violations.any():
    issues.append(
        ValidationIssue("high below open/close", df.index[high_violations].tolist())
    )

# Low should be <= min(open, close)
low_violations = df["low"] > df[["open", "close"]].min(axis=1)
if low_violations.any():
    issues.append(
        ValidationIssue("low above open/close", df.index[low_violations].tolist())
    )
```

**OHLC Invariants**:

```
Correct OHLC Bar:
        High (150.5)
          ▲
          │
Open ────┤       ┌──── Close (149.2)
(149.8)  │       │
         │   ▼   │
         └───────┘
          Low (148.7)

Invariants:
  high ≥ open
  high ≥ close
  low ≤ open
  low ≤ close

Violation Example:
┌──────────┬────────┬────────┬────────┬────────┐
│   date   │  open  │  high  │  low   │ close  │
├──────────┼────────┼────────┼────────┼────────┤
│2025-01-15│ 150.0  │ 149.5  │ 148.0  │ 151.2  │
└──────────┴────────┴────────┴────────┴────────┘
                     ↑                  ↑
                     └──────┬───────────┘
                    high (149.5) < close (151.2)  VIOLATION!

Causes:
  - Data recording error
  - Time zone mismatch (intraday data from different exchanges)
  - Adjustment error (splits/dividends applied incorrectly)
```

**Detection Algorithm**:
```python
# max(open, close) per row
max_oc = df[["open", "close"]].max(axis=1)

# Example row:
#   open = 150.0, close = 151.2
#   max_oc = 151.2

# Check if high < max_oc
high_violations = df["high"] < max_oc
#   149.5 < 151.2 → True (violation!)
```

---

### Stage 7: Zero-Filled Rows

```python
zero_fill = (df[["open", "high", "low", "close", "volume"]] == 0).all(axis=1)
if zero_fill.any():
    issues.append(ValidationIssue("zero-filled row", df.index[zero_fill].tolist()))
```

**Detection Logic**:
```
Input row:
  open=0, high=0, low=0, close=0, volume=0

Check: (df == 0).all(axis=1)
  [True, True, True, True, True].all() = True

Interpretation:
  - Market closed (holiday)
  - Data vendor missing data (filled with 0)
  - Stock not trading (delisted)

Why problematic:
  - Zero prices break log return calculations
  - Zero volume suggests no liquidity
  - Should be NaN (explicit missing) not 0 (implicit missing)
```

**Better Handling**:
```python
# Replace zero-filled rows with NaN
mask = (df[["open", "high", "low", "close", "volume"]] == 0).all(axis=1)
df.loc[mask, ["open", "high", "low", "close", "volume"]] = np.nan

# Then forward-fill or drop
df = df.ffill()  # or df.dropna()
```

---

## QUARANTINE SYSTEM

### Purpose
Isolate suspicious data without losing information for forensic analysis.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QUARANTINE WORKFLOW                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input: DataFrame (1000 rows)                                       │
│     ↓                                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Compute Daily Returns                                       │   │
│  │    ratio = close_t / close_{t-1}                             │   │
│  │    return = ratio - 1                                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│     ↓                                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Identify Invalid Returns                                    │   │
│  │    invalid = (ratio <= 0) | (ratio - 1 <= -1)               │   │
│  │                                                              │   │
│  │  Condition 1: ratio <= 0                                     │   │
│  │    Negative or zero price (impossible)                       │   │
│  │                                                              │   │
│  │  Condition 2: ratio - 1 <= -1                                │   │
│  │    Return <= -100% (total loss or worse)                     │   │
│  │    Likely unadjusted stock split                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│     ↓                                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Split Data                                                  │   │
│  │    valid_df = df[~invalid]      (995 rows)                  │   │
│  │    invalid_df = df[invalid]     (5 rows)                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│     ↓                                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Save Quarantine                                             │   │
│  │    if invalid_df not empty:                                  │   │
│  │      invalid_df.to_parquet(report_path)                      │   │
│  │                                                              │   │
│  │  Example path:                                               │   │
│  │    backend/artifacts/quarantine/AAPL_2025-01-15_invalid.parquet│
│  └─────────────────────────────────────────────────────────────┘   │
│     ↓                                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Threshold Check                                             │   │
│  │    invalid_frac = len(invalid_df) / len(df)                 │   │
│  │    = 5 / 1000 = 0.005                                        │   │
│  │                                                              │   │
│  │    if invalid_frac > max_invalid_frac (0.001):              │   │
│  │      FAIL - too much bad data                               │   │
│  │    else:                                                     │   │
│  │      PASS - acceptable loss                                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│     ↓                                                                │
│  Output: (valid_df, invalid_df)                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Example Scenario

```
Stock: XYZ (2:1 stock split on 2025-01-10)

Raw Data (unadjusted):
┌──────────┬────────┬─────────┬──────────┬──────────┐
│   date   │ close  │ ratio   │  return  │  valid?  │
├──────────┼────────┼─────────┼──────────┼──────────┤
│2025-01-08│ 200.00 │    -    │    -     │   Yes    │
│2025-01-09│ 202.50 │  1.0125 │  +1.25%  │   Yes    │
│2025-01-10│ 101.00 │  0.4988 │ -50.12%  │   NO!    │  ← Split not adjusted
│2025-01-13│ 102.30 │  1.0129 │  +1.29%  │   Yes    │
└──────────┴────────┴─────────┴──────────┴──────────┘

Detection:
  Row 2: ratio - 1 = -0.5012
  Check: -0.5012 <= -1? No, but close to -1
  Alternative check: ratio < 0.6 (heuristic for splits)

Adjusted Data (correct):
┌──────────┬────────┬─────────┬──────────┬──────────┐
│   date   │ close  │ ratio   │  return  │  valid?  │
├──────────┼────────┼─────────┼──────────┼──────────┤
│2025-01-08│ 100.00 │    -    │    -     │   Yes    │
│2025-01-09│ 101.25 │  1.0125 │  +1.25%  │   Yes    │
│2025-01-10│ 101.00 │  0.9975 │  -0.25%  │   Yes    │
│2025-01-13│ 102.30 │  1.0129 │  +1.29%  │   Yes    │
└──────────┴────────┴─────────┴──────────┴──────────┘

Lesson:
  Always use adjusted prices from data vendor
  Quarantine system catches these automatically
```

---

*End of Deep Dive - See SYSTEM_ANALYSIS_INDEX.md for navigation*

