# ALGAIE EXHAUSTIVE SYSTEM ANALYSIS
## Bottom-Up Architectural & Functional Walkthrough

*Status: Comprehensive Technical Documentation*
*Generated: 2026-02-13*

---

## TABLE OF CONTENTS

1. [System Foundation](#part-1-system-foundation)
2. [Data Layer Architecture](#part-2-data-layer-architecture)
3. [Feature Engineering Pipeline](#part-3-feature-engineering-pipeline)
4. [Model Architecture](#part-4-model-architecture)
5. [Training Pipelines](#part-5-training-pipelines)
6. [Inference & Execution](#part-6-inference--execution)
7. [Risk & Portfolio Management](#part-7-risk--portfolio-management)
8. [Complete System Integration](#part-8-complete-system-integration)

---

# PART 1: SYSTEM FOUNDATION

## 1.1 Core Configuration System

### Purpose
The configuration system (`algaie/core/config.py`) provides a hierarchical, type-safe configuration management layer that controls all aspects of the trading system.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PipelineConfig                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Root Configuration                                    │  │
│  │  - artifact_root: Path (backend/artifacts)           │  │
│  │  - max_invalid_frac: float (0.001)                   │  │
│  │  - enable_quantiles: bool (False)                    │  │
│  │  - run_id: Optional[str] (RUN-YYYYMMDD-HHMMSS)      │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│         ┌─────────────────┼─────────────────┐              │
│         │                 │                 │               │
│         ▼                 ▼                 ▼               │
│  ┌──────────┐     ┌─────────────┐    ┌──────────┐         │
│  │ Backtest │     │  Portfolio  │    │  Broker  │         │
│  │  Config  │     │   Config    │    │  Config  │         │
│  └──────────┘     └─────────────┘    └──────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1.1.1 BacktestConfig (Frozen Dataclass)

**Purpose**: Controls all backtesting simulation parameters including fill simulation, cost models, and walk-forward testing configuration.

**Fields**:
```python
@dataclass(frozen=True)
class BacktestConfig:
    # Time boundaries
    start: Optional[str] = None          # Start date (YYYY-MM-DD)
    end: Optional[str] = None            # End date (YYYY-MM-DD)

    # Fill simulation
    fill_price_mode: str = "next_open"   # Choices: "next_open", "same_close", "vwap"
    rounding_policy: str = "fractional"  # Choices: "fractional", "round", "floor", "ceil"

    # Cost models
    slippage_model: str = "none"         # Choices: "none", "fixed_bps", "volume_impact"
    slippage_bps: float = 0.0           # Basis points (1 bp = 0.01%)
    slippage_volume_impact: float = 0.0  # Volume impact coefficient
    commission_per_trade: float = 0.0    # Fixed per trade ($)
    commission_per_share: float = 0.0    # Per share ($)
    commission_bps: float = 0.0          # Basis points of notional
    commission_min: float = 0.0          # Minimum commission ($)

    # Walk-forward configuration
    walk_forward: bool = False           # Enable walk-forward testing
    train_window_days: int = 504         # Training window (~2 years of trading days)
    test_window_days: int = 126          # Test window (~6 months)
    step_days: int = 126                 # Step size for rolling window
    holdout_pct: float = 0.1            # Holdout percentage for validation
    expanding_window: bool = True        # Expanding vs rolling window
```

**Why This Design**:
- **Frozen dataclass**: Immutability prevents accidental configuration changes during execution
- **fill_price_mode**: Realistic fill simulation is critical for backtest validity
  - `next_open`: Conservative, assumes execution at next day's open
  - `same_close`: Optimistic, assumes execution at signal day's close
  - `vwap`: Medium realism, uses volume-weighted average price
- **Slippage modeling**: Three-tier approach from simple to sophisticated
  - `none`: For quick validation
  - `fixed_bps`: Standard industry model (typical: 2-5 bps)
  - `volume_impact`: Sophisticated model accounting for order size vs ADV
- **Walk-forward parameters**: Prevents overfitting through time-series aware validation
  - 504 training days ≈ 2 years of market data
  - 126 test days ≈ 6 months out-of-sample
  - Expanding window: Uses all historical data up to test period

#### 1.1.2 PortfolioConfig (Frozen Dataclass)

**Purpose**: Governs portfolio construction, position sizing, and rebalancing rules.

**Fields**:
```python
@dataclass(frozen=True)
class PortfolioConfig:
    # Selection
    top_k: int = 50                      # Number of positions to select

    # Weighting
    weight_method: str = "softmax"       # Choices: "softmax", "equal", "rank", "score"
    softmax_temp: float = 1.0           # Softmax temperature (lower = more concentrated)
    max_weight_per_name: float = 0.1    # Maximum weight per position (10%)
    max_names: int = 50                  # Maximum number of positions

    # Position sizing
    min_dollar_position: float = 0.0     # Minimum position size ($)
    cash_buffer_pct: float = 0.05       # Cash buffer (5%)

    # Exit policy
    exit_policy: str = "hybrid"          # Choices: "time", "signal", "hybrid"
    hold_days: Optional[int] = 10        # Default holding period (days)
    hold_days_min: Optional[int] = None  # Minimum hold (early exit prevention)
    hold_days_max: Optional[int] = None  # Maximum hold (forced exit)
```

**Why This Design**:
- **top_k selection**: Cross-sectional ranking naturally produces top-k portfolios
- **weight_method options**:
  - `softmax`: Differentiable, score-weighted (convex combination)
  - `equal`: Simple 1/k weighting, robust to score errors
  - `rank`: Rank-weighted, reduces impact of score magnitude
  - `score`: Direct score proportionality
- **softmax_temp**: Controls concentration
  - temp → 0: Approaches argmax (winner-take-all)
  - temp = 1.0: Standard Gibbs distribution
  - temp → ∞: Approaches equal weighting
- **Exit policy philosophy**:
  - `time`: Mechanical time-based exits (swing trading default)
  - `signal`: Signal decay triggers exit
  - `hybrid`: Combines both (recommended)

#### 1.1.3 BrokerConfig (Frozen Dataclass)

**Purpose**: Controls broker integration, execution mode, and order constraints.

**Fields**:
```python
@dataclass(frozen=True)
class BrokerConfig:
    mode: str = "paper"                  # Choices: "paper", "live", "backtest"
    dry_run: bool = False                # Log orders without execution

    # Order constraints
    max_orders_per_day: int = 200        # Circuit breaker
    max_notional_per_order: float = 100000.0  # Maximum order size ($)

    # Execution policy
    fractional_policy: str = "round"     # How to handle fractional shares
    rounding_policy: str = "round"       # Rounding method

    # Broker endpoints
    alpaca_base_url: Optional[str] = None    # Alpaca Markets API
    ibkr_gateway_url: Optional[str] = None   # Interactive Brokers Gateway
```

**Why This Design**:
- **mode separation**: Clear separation of simulation vs live execution
- **max_orders_per_day**: Protects against runaway loops
- **fractional_policy**: Some brokers support fractional shares, others don't
- **Multiple broker support**: Alpaca (modern API) vs IBKR (institutional)

#### 1.1.4 Configuration Loading

**Loading Mechanism**:
```python
def load_config(path: Path | str, *, strict: bool = False) -> PipelineConfig:
    """
    Load configuration from YAML or JSON file.

    Args:
        path: Configuration file path
        strict: If True, reject unknown keys

    Returns:
        PipelineConfig instance

    Raises:
        FileNotFoundError: Config file doesn't exist
        ValueError: Unsupported file extension or invalid format
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        payload = _load_yaml(config_path)
    elif suffix == ".json":
        payload = _load_json(config_path)
    else:
        raise ValueError(f"Unsupported config extension: {suffix}")

    return PipelineConfig.from_mapping(payload, strict=strict)
```

**Why This Design**:
- **Multi-format support**: YAML for human editing, JSON for programmatic generation
- **Strict mode**: Production deployments should use strict=True to catch typos
- **from_mapping pattern**: Enables config composition and overrides

---

## 1.2 Path Management System

### Purpose
The path management system (`algaie/core/paths.py`) provides a centralized, type-safe interface for artifact storage locations.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ArtifactPaths                               │
│                    (Frozen Dataclass)                           │
│                                                                 │
│  root: Path (e.g., backend/artifacts/)                         │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  Storage Hierarchy                        │  │
│  │                                                            │  │
│  │  canonical_daily/     → raw OHLCV data (partitioned)     │  │
│  │  eligibility/         → universe manifests                │  │
│  │  features/            → engineered features               │  │
│  │  priors/              → Chronos teacher outputs           │  │
│  │  models/              │                                    │  │
│  │    ├─ foundation/     → Chronos checkpoints               │  │
│  │    └─ ranker/         → Selector models                   │  │
│  │  signals/             → Daily ranking signals             │  │
│  │  reports/             → Analysis outputs                  │  │
│  │  runs/                → Production run logs               │  │
│  │  backtests/           → Backtest results                  │  │
│  │  paper/               → Paper trading logs                │  │
│  │  live/                → Live trading logs                 │  │
│  │  options_chains/      → Options market data               │  │
│  │  vrp_reports/         → Volatility risk premium analysis  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Details

**Directory Mapping**:
```python
_ARTIFACT_DIRS: Dict[str, tuple] = {
    "canonical_daily": ("canonical", "daily"),
    "eligibility": ("eligibility",),
    "features": ("features",),
    "priors": ("priors",),
    "models_foundation": ("models", "foundation"),
    "models_ranker": ("models", "ranker"),
    "signals": ("signals",),
    "reports": ("reports",),
    "runs": ("runs",),
    "backtests": ("backtests",),
    "paper": ("paper",),
    "live": ("live",),
    "options_chains": ("options_chains",),
    "vrp_reports": ("vrp_reports",),
    "vrp_audits": ("vrp_audits",),
    "lag_llama_series": ("lag_llama", "series"),
    "lag_llama_forecasts": ("lag_llama", "forecasts"),
    "lag_llama_validation": ("lag_llama", "validation"),
}
```

**Why This Design**:
- **Tuple-based paths**: Enables nested directory structures (e.g., `models/foundation`)
- **Property accessors**: Type-safe, autocomplete-friendly access
- **Centralized mapping**: Single source of truth for path structure
- **Partitioning support**: Natural partitioning scheme (e.g., `symbol=AAPL/date=2025-01-15`)

**Directory Initialization**:
```python
def ensure_artifact_dirs(paths: ArtifactPaths) -> None:
    """Create all artifact directories if they don't exist."""
    for name in _ARTIFACT_DIRS:
        paths._resolve(name).mkdir(parents=True, exist_ok=True)
```

---

## 1.3 Contract & Schema System

### Purpose
The contract system (`algaie/core/contracts.py`) provides dual-layer data validation: schema contracts (column presence + types) and feature contracts (hash-based versioning).

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   Contract Validation Layers                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Layer 1: Schema Contracts (Structural)                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Purpose: Ensure required columns exist with correct types │  │
│  │                                                             │  │
│  │  UNIVERSEFRAME_V2_REQUIRED_COLS:                           │  │
│  │    - date, symbol                                          │  │
│  │    - is_observable, is_tradable                            │  │
│  │    - tier, weight                                          │  │
│  │                                                             │  │
│  │  SELECTOR_FEATURES_V2_REQUIRED_COLS:                       │  │
│  │    - date, symbol                                          │  │
│  │    - x_lr1, x_lr5, x_lr20, x_vol, x_relvol               │  │
│  │    - y_rank, tier, weight                                  │  │
│  │                                                             │  │
│  │  PRIORS_REQUIRED_COLS:                                     │  │
│  │    - date, symbol                                          │  │
│  │    - prior_drift_20d, prior_vol_20d                        │  │
│  │    - prior_downside_q10_20d, prior_trend_conf_20d         │  │
│  │    - chronos_model_id, context_len, horizon, prior_version │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Layer 2: Feature Contracts (Versioning)                         │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Purpose: Detect feature set changes via SHA-256 hashing  │  │
│  │                                                             │  │
│  │  compute_contract_hash(columns: List[str]) -> str          │  │
│  │    1. Sort column names alphabetically                     │  │
│  │    2. Join with commas                                     │  │
│  │    3. SHA-256 hash                                         │  │
│  │    4. Return first 16 hex chars                            │  │
│  │                                                             │  │
│  │  validate_contract(df, contract_hash) -> bool              │  │
│  │    Compare current hash against expected                   │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Implementation Details

#### 1.3.1 Schema Normalization

**Purpose**: Standardize key columns across the system.

```python
def normalize_keys(df: Any) -> Any:
    """
    Normalize DataFrame keys to canonical schema:
      * Rename ``ticker`` → ``symbol``
      * Cast ``date`` → Date
      * Cast ``symbol`` → Utf8 / str

    Supports both Polars and Pandas DataFrames.
    """
    if _HAS_POLARS and isinstance(df, pl.DataFrame):
        if "ticker" in df.columns and "symbol" not in df.columns:
            df = df.rename({"ticker": "symbol"})
        if "date" in df.columns and df.schema["date"] != pl.Date:
            df = df.with_columns(pl.col("date").cast(pl.Date))
        if "symbol" in df.columns and df.schema["symbol"] != pl.Utf8:
            df = df.with_columns(pl.col("symbol").cast(pl.Utf8))
        return df

    # Pandas path
    if isinstance(df, pd.DataFrame):
        if "ticker" in df.columns and "symbol" not in df.columns:
            df = df.rename(columns={"ticker": "symbol"})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    raise TypeError(f"normalize_keys: unsupported type {type(df)}")
```

**Why This Design**:
- **ticker → symbol**: Historical consistency (older code used "ticker")
- **Date casting**: Ensures temporal operations work correctly
- **Dual library support**: Handles both Polars (preferred) and Pandas (legacy)

#### 1.3.2 Contract Hashing

**Purpose**: Create stable, reproducible identifiers for feature sets.

```python
def compute_contract_hash(columns: List[str]) -> str:
    """Stable SHA-256 hash (first 16 hex chars) of sorted column names."""
    sorted_cols = sorted(columns)
    joined = ",".join(sorted_cols)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
```

**Why This Design**:
- **Sorting**: Ensures hash is independent of column order
- **SHA-256**: Cryptographic hash prevents collisions
- **16 chars**: Balance between uniqueness and readability
- **Use cases**:
  - Detect feature engineering changes
  - Version compatibility checks
  - Cache invalidation

---

## 1.4 Artifact Registry & Versioning

### Purpose
Track all artifacts (models, data, features) with version hashes for reproducibility.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Artifact Registry                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ArtifactRecord                                            │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  name: str          (e.g., "selector_v1")            │ │
│  │  path: Path         (relative to artifact_root)      │ │
│  │  version: str       (12-char SHA-256 hash)           │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  Registry Operations:                                       │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  register(name, path, version)                        │ │
│  │    → Store artifact metadata                          │ │
│  │                                                        │ │
│  │  dump(destination)                                    │ │
│  │    → Serialize to JSON                                │ │
│  │    → Format: {name: {path, version}}                 │ │
│  │                                                        │ │
│  │  list_records()                                       │ │
│  │    → Iterate all registered artifacts                 │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Versioning System

**Hash Generation**:
```python
def stable_hash(payload: Dict[str, Any]) -> str:
    """
    Generate stable 12-character hash from dictionary.

    Process:
    1. JSON serialize with sorted keys
    2. Handle non-serializable types (dates, etc.) via str()
    3. SHA-256 hash
    4. Return first 12 hex characters
    """
    dumped = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(dumped).hexdigest()[:12]
```

**Why This Design**:
- **Dataclass support**: `dataclass_hash()` wraps `asdict()` for automatic hashing
- **Deterministic**: Sorted keys ensure reproducibility
- **12 characters**: ~2^48 combinations (sufficient for artifact versioning)
- **Production usage**:
  ```python
  # Example: Version a selector model
  config_hash = stable_hash({
      "model_type": "RankTransformer",
      "d_model": 64,
      "nhead": 4,
      "feature_version": "abc123def456"
  })
  # → "7f3a8c9b1e2d"
  ```

---

## 1.5 Trading Calendar System

### Purpose
Provide trading-day aware date arithmetic (critical for avoiding look-ahead bias).

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              Trading Calendar (NYSE/NASDAQ)                  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  trading_days(start: date, end: date) -> List[date]         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  1. Use pandas.bdate_range(start, end)                 │ │
│  │     - Excludes weekends                                 │ │
│  │     - Excludes major holidays (US market holidays)      │ │
│  │  2. Convert to Python date objects                      │ │
│  │  3. Return list of valid trading days                   │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  next_trading_day(current: date, calendar) -> date | None   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  1. Find current date in calendar                       │ │
│  │  2. Return next date in sequence                        │ │
│  │  3. Return None if at end of calendar                   │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

**Critical Importance**:
```
INCORRECT (Calendar Days):
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│  Mon  │  Tue  │  Wed  │  Thu  │  Fri  │  Sat  │  Sun  │
├───────┼───────┼───────┼───────┼───────┼───────┼───────┤
│ Day 1 │ Day 2 │ Day 3 │ Day 4 │ Day 5 │ Day 6 │ Day 7 │
│  ✓    │  ✓    │  ✓    │  ✓    │  ✓    │  ✗    │  ✗    │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┘
5-day forward horizon includes weekend!

CORRECT (Trading Days):
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│  Mon  │  Tue  │  Wed  │  Thu  │  Fri  │  Sat  │  Sun  │  Mon  │  Tue  │
├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
│  T1   │  T2   │  T3   │  T4   │  T5   │  ---  │  ---  │  T6   │  T7   │
│  ✓    │  ✓    │  ✓    │  ✓    │  ✓    │  ✗    │  ✗    │  ✓    │  ✓    │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
5-day forward horizon = 5 trading days (skips weekend)
```

**Why This Design**:
- **No look-ahead bias**: Forward returns computed using actual trading days
- **Annualization factor**: ~252 trading days/year (not 365)
- **Holiday awareness**: Market closed on federal holidays
- **Production usage**:
  ```python
  # Compute 5-day forward return (CORRECT)
  current_date = date(2025, 1, 15)  # Wednesday
  calendar = trading_days(date(2025, 1, 1), date(2025, 12, 31))

  # Get date 5 trading days later
  target_date = current_date
  for _ in range(5):
      target_date = next_trading_day(target_date, calendar)

  # target_date = 2025-01-22 (Wednesday, skipping weekend)
  forward_return = (price_at(target_date) / price_at(current_date)) - 1.0
  ```

---

This completes Part 1 of the exhaustive analysis. Continue to Part 2?

