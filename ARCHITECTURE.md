# ALGAIE - Comprehensive Architectural Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Core Module](#core-module)
3. [Data Module](#data-module)
4. [Models Module](#models-module)
5. [Engine Module](#engine-module)
6. [Risk Module](#risk-module)
7. [Portfolio Module](#portfolio-module)
8. [Options Module](#options-module)
9. [Preprocessing Module](#preprocessing-module)
10. [Evaluation Module](#evaluation-module)
11. [Inference Module](#inference-module)
12. [Scripts](#scripts)
13. [Tests](#tests)

---

## System Overview

ALGAIE is an institutional-grade algorithmic trading platform implementing a **Chronos-2 Priors + Rank-Transformer Selector** architecture for swing trading equities and options (put credit spreads).

### Key Design Patterns
- **Chronos-2 Priors + Rank-Transformer Selector**: Chronos-2 generates probabilistic priors -> Rank-Transformer ranks equities combining priors and market features
- **Pod-Based Engines**: EquityPod for swing equities, OptionsPod for credit spreads
- **Multi-Stage Gating**: Risk posture, crash override, options gate for trade filtering
- **Hierarchical Risk Parity (HRP)**: Portfolio allocation via correlation clustering (Planned)

### Technology Stack
- Python 3.11+
- PyTorch (neural networks)
- Polars (high-performance DataFrames)
- HuggingFace Transformers + PEFT/LoRA (Chronos models)
- exchange_calendars (NYSE trading calendar)
- scikit-learn (calibration, clustering)

---

## Core Module
**Location**: `backend/app/core/`

### config.py
Central configuration via environment variables.

```python
# Key Configuration Constants
ENABLE_OPTIONS: bool      # Toggle options trading (default: False)
OPTIONS_MODE: str         # "paper" | "live" | "shadow" (default: OFF)
EXECUTION_MODE: str       # "LEGACY" | "SHADOW" | "RANKING" (default: LEGACY)
REBALANCE_CALENDAR: str   # Calendar identifier for rebalancing
```

**Functions**: None (module-level constants only)

---

### artifacts.py
Artifact path resolution for versioned model checkpoints.

**Functions**:
| Function | Parameters | Returns | Purpose |
|----------|------------|---------|---------|
| `resolve_priors_path()` | `date: str, ticker: str` | `str` | Path to teacher priors parquet |
| `resolve_selector_checkpoint()` | `version: str` | `str` | Path to selector model checkpoint |
| `resolve_scaler_path()` | `version: str` | `str` | Path to fitted scaler |
| `resolve_calibration_path()` | `version: str` | `str` | Path to calibration artifact |
| `resolve_leaderboard_path()` | `date: str` | `str` | Path to daily leaderboard parquet |

---

### events.py
Event-driven architecture foundation.

**Classes**:

| Class | Fields | Purpose |
|-------|--------|---------|
| `MarketEvent` | `timestamp, ticker, data` | Raw market data tick |
| `SignalEvent` | `timestamp, ticker, signal, confidence` | Model output signal |
| `OrderEvent` | `timestamp, ticker, action, quantity, order_type` | Order to execute |
| `FillEvent` | `timestamp, ticker, quantity, price, commission` | Execution confirmation |

---

## Data Module
**Location**: `backend/app/data/`

### manifest.py
Trading universe definition.

```python
DEFAULT_UNIVERSE = {
    "leaders": ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA"],
    "vol_beasts": ["AMD", "COIN", "MARA", "RIOT"],
    "liquidity_proxies": ["SPY", "QQQ", "IWM"]
}
```

**Functions**:
| Function | Returns | Purpose |
|----------|---------|---------|
| `get_default_universe()` | `List[str]` | All tickers in universe |
| `get_leaders()` | `List[str]` | High-cap tech leaders |
| `get_vol_beasts()` | `List[str]` | High-volatility names |
| `get_liquidity_proxies()` | `List[str]` | ETF proxies for hedging |

---

### marketframe.py
Polars-based OHLCV + breadth data loading.

**Class: `MarketFrame`**

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `__init__()` | `ticker: str, timeframe: str` | - | Initialize frame for ticker |
| `load()` | `path: str` | `pl.DataFrame` | Load parquet data |
| `get_window()` | `start: datetime, end: datetime` | `pl.DataFrame` | Slice time window |
| `with_breadth()` | `ad_line: Series, bpi: Series` | `MarketFrame` | Attach breadth context |

**Schema**:
```
timestamp: datetime
open: float64
high: float64
low: float64
close: float64
volume: int64
ad_line: float64 (optional)
bpi: float64 (optional)
```

---

### calendar.py
NYSE trading calendar integration.

**Functions**:
| Function | Parameters | Returns | Purpose |
|----------|------------|---------|---------|
| `get_nyse_calendar()` | - | `ExchangeCalendar` | NYSE calendar instance |
| `is_trading_day()` | `date: date` | `bool` | Check if market open |
| `get_next_trading_day()` | `date: date` | `date` | Next open market day |
| `get_trading_days_between()` | `start, end` | `List[date]` | Range of trading days |

---

### breadth.py
Market breadth indicators (AD Line, BPI).

**Functions**:
| Function | Parameters | Returns | Purpose |
|----------|------------|---------|---------|
| `compute_ad_line()` | `df: DataFrame` | `Series` | Advance-Decline Line |
| `compute_bpi()` | `df: DataFrame` | `Series` | Bullish Percent Index |
| `get_breadth_context()` | `date: date` | `Dict` | Combined breadth metrics |

---

### windows.py
Cross-sectional batch creation for training.

**Class: `SwingWindowDataset`** (PyTorch Dataset)

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `__init__()` | `df, cols, lookback, stride` | - | Initialize windowing |
| `__len__()` | - | `int` | Number of windows |
| `__getitem__()` | `idx: int` | `Tuple[Tensor, Dict]` | Get (X, Y) pair |

**Window Structure**:
- Input: `[B, Lookback, Features]` - historical features
- Target: `Dict["1D": Tensor, "3D": Tensor]` - forward returns

---

### splits.py
Train/validation/test splitting with time-series awareness.

**Functions**:
| Function | Parameters | Returns | Purpose |
|----------|------------|---------|---------|
| `time_series_split()` | `df, train_ratio, val_ratio` | `Tuple[df, df, df]` | Chronological split |
| `walk_forward_split()` | `df, n_splits, train_size` | `Generator` | Walk-forward CV |

---

## Models Module
**Location**: `backend/app/models/`

### signal_types.py
Core signal dataclasses for inter-module communication.

**Dataclasses**:

```python
@dataclass
class ModelSignal:
    ticker: str
    timestamp: datetime
    direction: float      # -1 to +1
    confidence: float     # 0 to 1
    horizon: str          # "1D" | "3D" | "5D"

@dataclass
class ModelMetadata:
    model_version: str
    preproc_id: str
    training_start: str
    training_end: str

@dataclass
class RankScoreSignal:
    ticker: str
    rank_score: float     # 0-1 calibrated
    p_up: float           # Probability of up move
    raw_score: float      # Pre-calibration score

@dataclass
class ChronosPriors:
    drift_20d: float
    vol_20d: float
    downside_q10_20d: float
    trend_conf_20d: float

@dataclass
class SelectorOutputs:
    scores: Dict[str, float]      # ticker -> score
    rankings: List[str]           # sorted tickers
    calibrated_probs: Dict[str, float]
```

**Constants**:
```python
LEADERBOARD_SCHEMA = {
    "date": pl.Date,
    "ticker": pl.Utf8,
    "rank_score": pl.Float64,
    "p_up": pl.Float64,
    "selected": pl.Boolean
}
```

---

### baseline.py
Baseline MLP neural network.

**Class: `BaselineMLP`**

```python
class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int, lookback: int, horizons: int = 2, output_dim: int = 3):
        """
        Input:  (B, Lookback, InputDim)
        Output: (B, Horizons=2, Quantiles=3)

        Architecture:
        - Flatten: B x (Lookback * InputDim)
        - Linear: -> 128
        - ReLU
        - Dropout(0.2)
        - Linear: -> 64
        - ReLU
        - Linear: -> Horizons * OutputDim
        - Reshape: (B, Horizons, OutputDim)
        """
```

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `forward()` | `x: Tensor` | `Tensor` | Forward pass |

---

### rank_transformer.py
Encoder-only Transformer for cross-sectional ranking.

**Class: `RankTransformer`**

```python
class RankTransformer(nn.Module):
    def __init__(self,
                 input_dim: int = 10,      # Feature count
                 d_model: int = 64,        # Hidden dimension
                 nhead: int = 4,           # Attention heads
                 num_layers: int = 2,      # Encoder layers
                 dropout: float = 0.1):
        """
        Architecture:
        1. Input projection: Linear(input_dim -> d_model)
        2. Positional encoding: Learnable [1, max_len, d_model]
        3. Transformer encoder: num_layers x EncoderLayer
        4. Score head: Linear(d_model -> 1)
        5. Direction head: Linear(d_model -> 1) + Sigmoid

        Input:  [B, T, F] - Batch, Time, Features
        Output: {"score": [B, 1], "p_up": [B, 1]}
        """
```

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `forward()` | `x: Tensor` | `Dict[str, Tensor]` | Compute scores |
| `get_attention_weights()` | `x: Tensor` | `Tensor` | Extract attention for interpretability |

---

### rank_losses.py
Ranking-specific loss functions.

**Functions**:
| Function | Parameters | Returns | Purpose |
|----------|------------|---------|---------|
| `pairwise_ranking_loss()` | `scores, labels` | `Tensor` | Pairwise margin loss |
| `listwise_ranking_loss()` | `scores, labels` | `Tensor` | ListNet-style softmax loss |
| `combined_ranking_loss()` | `scores, labels, alpha` | `Tensor` | Weighted combination |

---

### calibration.py
Score calibration via isotonic regression.

**Class: `ScoreCalibrator`**

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `fit()` | `raw_scores, outcomes` | `self` | Fit isotonic regression |
| `calibrate()` | `raw_scores` | `np.ndarray` | Transform to probabilities |
| `save()` | `path: str` | - | Persist calibrator |
| `load()` | `path: str` | `ScoreCalibrator` | Load fitted calibrator |

---

### selector_scaler.py
Feature scaling for selector input.

**Class: `SelectorFeatureScaler`**

```python
class SelectorFeatureScaler:
    """
    Robust scaling with outlier clipping.

    For each feature:
    1. Clip to [q01, q99] percentiles
    2. Robust scale: (x - median) / IQR
    3. Final clip to [-3, 3]
    """
```

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `fit()` | `df: DataFrame` | `self` | Compute scaling params |
| `transform()` | `df: DataFrame` | `DataFrame` | Apply scaling |
| `fit_transform()` | `df: DataFrame` | `DataFrame` | Fit then transform |
| `save()` | `path: str` | - | Persist scaler params |
| `load()` | `path: str` | `SelectorFeatureScaler` | Load fitted scaler |

---

### feature_contracts.py
Feature contract validation.

```python
SELECTOR_FEATURE_CONTRACT = [
    "log_return_1d",
    "log_return_5d",
    "log_return_20d",
    "volatility_20d",
    "volume_log_change_5d",
    "ad_line_trend_5d",
    "bpi_level",
    "teacher_drift_20d",
    "teacher_vol_20d",
    "teacher_downside_q10_20d"
]
```

**Functions**:
| Function | Parameters | Returns | Purpose |
|----------|------------|---------|---------|
| `validate_features()` | `df: DataFrame` | `bool` | Check all features present |
| `get_feature_contract()` | - | `List[str]` | Return contract list |

---

### model_io.py
Model serialization with metadata.

**Functions**:
| Function | Parameters | Returns | Purpose |
|----------|------------|---------|---------|
| `save_model()` | `state_dict, metadata, path` | - | Save model + metadata |
| `load_model()` | `path, model_class` | `Tuple[Module, Metadata]` | Load model + metadata |
| `get_model_hash()` | `state_dict` | `str` | Hash of model weights |

---

### student_inference.py
Real-time student model inference wrapper.

**Class: `StudentRunner`**

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `__init__()` | `model_path, preproc_path` | - | Load model and preprocessor |
| `predict()` | `features: DataFrame` | `ModelSignal` | Generate trading signal |
| `get_model_metadata()` | - | `ModelMetadata` | Return loaded metadata |

---

### selector_runner.py
Selector model inference with scaling and calibration.

**Class: `SelectorRunner`**

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `__init__()` | `checkpoint_path, scaler_path, calibrator_path` | - | Load all artifacts |
| `rank()` | `df: DataFrame` | `SelectorOutputs` | Rank tickers cross-sectionally |
| `select_top_k()` | `outputs, k` | `List[str]` | Return top K tickers |

---

### chronos2_teacher.py
Chronos 2 model loading with LoRA/QLoRA support (~498 lines).

**Key Functions**:
| Function | Parameters | Returns | Purpose |
|----------|------------|---------|---------|
| `load_chronos_model()` | `model_name, device, quantize` | `ChronosModel` | Load base Chronos model |
| `apply_lora()` | `model, lora_config` | `PeftModel` | Apply LoRA adapters |
| `generate_forecasts()` | `model, context, horizons` | `np.ndarray` | Generate probabilistic forecasts |

**Class: `Chronos2Teacher`**

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `__init__()` | `model_name, adapter_path, device` | - | Initialize teacher |
| `forecast()` | `series: np.ndarray, horizons: List[int]` | `Dict` | Multi-horizon forecast |
| `compute_priors()` | `df: DataFrame, ticker: str` | `ChronosPriors` | Calculate teacher priors |

---

### chronos2_codec.py
Tokenization codec for multivariate time series.

**Class: `ChronosCodec`**

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `encode()` | `series: np.ndarray` | `torch.Tensor` | Series → tokens |
| `decode()` | `tokens: torch.Tensor` | `np.ndarray` | Tokens → series |
| `get_context_length()` | - | `int` | Max context window |

---

### mock_student_runner.py
Mock student for testing without GPU.

**Class: `MockStudentRunner`**

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `predict()` | `features` | `ModelSignal` | Return random signal |

---

### teacher_equity_inference.py / teacher_o_runner.py
Equity and Options teacher runners (similar interface to StudentRunner).

---

### tiny_o_runner.py / lag_llama_runner.py
Alternative teacher model runners for experimentation.

---

## Engine Module
**Location**: `backend/app/engine/`

### equity_pod.py
Main equity trading pod - orchestrates the full swing trading pipeline.

**Class: `EquityPod`**

```python
class EquityPod:
    """
    Execution Modes:
    - LEGACY: Old single-stock logic
    - SHADOW: Log signals without executing
    - RANKING: Full selector-based ranking

    Components:
    - StudentRunner: Signal generation
    - RiskManager: Signal filtering
    - SwingScheduler: Trading window detection
    - PortfolioBuilder: Position construction
    """
```

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `__init__()` | `ticker, model_path, preproc_path` | - | Initialize all components |
| `on_tick()` | `tick: Dict, breadth: Dict` | `Optional[Decision]` | Process market tick |
| `execute_decision()` | `decision: Decision` | - | Execute trade decision |
| `get_portfolio_state()` | - | `PortfolioState` | Current positions |

---

### options_pod.py
Options trading pod for put credit spreads.

**Class: `OptionsPod`**

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `__init__()` | `config: OptionsConfig` | - | Initialize options engine |
| `evaluate()` | `underlying: str, chain: DataFrame` | `Optional[OptionsDecision]` | Evaluate spread opportunity |
| `execute()` | `decision: OptionsDecision` | `FillEvent` | Execute spread trade |

---

### swing_scheduler.py
Trading window detection and scheduling.

**Enum: `TradingWindow`**
```python
class TradingWindow(Enum):
    CLOSED = "CLOSED"           # Market closed
    PRE_OPEN = "PRE_OPEN"       # 9:00-9:30 ET
    EARLY_ENTRY = "EARLY_ENTRY" # 9:30-10:30 ET (primary entry)
    MID_DAY = "MID_DAY"         # 10:30-15:00 ET
    LATE_ADJUST = "LATE_ADJUST" # 15:00-15:45 ET (adjustments)
    POST_CLOSE = "POST_CLOSE"   # After 16:00 ET
```

**Class: `SwingScheduler`**

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `get_current_window()` | `timestamp: datetime` | `TradingWindow` | Determine current window |
| `should_trade()` | `window: TradingWindow, action: ActionType` | `bool` | Check if action allowed |
| `next_window()` | `timestamp: datetime` | `Tuple[TradingWindow, datetime]` | Next window and start time |

---

### portfolio_construction.py
Portfolio construction with Top-K selection.

**Class: `PortfolioBuilder`**

```python
class PortfolioBuilder:
    """
    Configuration:
    - top_k: int = 5           # Number of positions
    - min_hold_days: int = 3   # Minimum hold period
    - max_hold_days: int = 10  # Maximum hold period
    - turnover_limit: float = 0.3  # Max daily turnover
    """
```

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `select()` | `rankings: List[str], current: PortfolioState` | `List[str]` | Select tickers to hold |
| `compute_weights()` | `tickers: List[str], allocator: HRPAllocator` | `Dict[str, float]` | Allocate weights via HRP |
| `get_rebalance_trades()` | `target: Dict, current: PortfolioState` | `List[Order]` | Generate rebalance orders |

---

## Risk Module
**Location**: `backend/app/risk/`

### types.py
Risk-related type definitions.

**Enum: `ActionType`**
```python
class ActionType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NO_NEW_RISK = "NO_NEW_RISK"
    REDUCE_EXPOSURE = "REDUCE_EXPOSURE"
    CLOSE_ALL = "CLOSE_ALL"
```

**Dataclass: `RiskDecision`**
```python
@dataclass
class RiskDecision:
    action: ActionType
    ticker: str
    quantity: int
    reason: str
    confidence: float
```

---

### posture.py
Risk posture states.

**Enum: `RiskPosture`**
```python
class RiskPosture(str, Enum):
    NORMAL = "NORMAL"       # Full risk budget
    CAUTIOUS = "CAUTIOUS"   # Reduced position sizes
    DEFENSIVE = "DEFENSIVE" # No new risk, reduce exposure
```

---

### risk_manager.py
Signal-to-decision conversion based on risk posture.

**Class: `RiskManager`**

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `__init__()` | `posture: RiskPosture` | - | Initialize with posture |
| `set_posture()` | `posture: RiskPosture` | - | Update risk posture |
| `evaluate()` | `signal: ModelSignal, portfolio: PortfolioState` | `RiskDecision` | Convert signal to decision |
| `get_position_size()` | `signal, portfolio, posture` | `int` | Calculate position size |

**Posture Rules**:
| Posture | Max Position Size | New Entries | Behavior |
|---------|-------------------|-------------|----------|
| NORMAL | 100% | Allowed | Full signal following |
| CAUTIOUS | 50% | Allowed | Reduced sizing |
| DEFENSIVE | 0% | Blocked | Only exits allowed |

---

### crash_override.py
Crash detection and automatic posture adjustment.

**Class: `CrashOverride`**

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `__init__()` | `bpi_threshold, ad_threshold` | - | Set crash thresholds |
| `check()` | `breadth: Dict` | `Optional[RiskPosture]` | Check for crash signal |
| `get_override_reason()` | - | `str` | Explanation of override |

**Thresholds**:
```python
BPI_CRASH_THRESHOLD = 30.0    # BPI < 30 = DEFENSIVE
BPI_CAUTION_THRESHOLD = 40.0  # BPI < 40 = CAUTIOUS
AD_CRASH_THRESHOLD = -500     # AD Line decline threshold
```

---

## Portfolio Module
**Location**: `backend/app/portfolio/`

### state.py
Portfolio state management.

**Dataclass: `Position`**
```python
@dataclass
class Position:
    ticker: str
    quantity: int
    avg_cost: float
    entry_date: date
    unrealized_pnl: float
    days_held: int
```

**Dataclass: `PortfolioState`**
```python
@dataclass
class PortfolioState:
    cash: float
    positions: Dict[str, Position]
    total_equity: float
    timestamp: datetime

    def get_weight(self, ticker: str) -> float: ...
    def get_exposure(self) -> float: ...
```

---

### hrp.py
Hierarchical Risk Parity allocation.

**Class: `HRPAllocator`**

```python
class HRPAllocator:
    """
    Hierarchical Risk Parity (HRP) for portfolio allocation.

    Algorithm:
    1. Compute correlation matrix of returns
    2. Hierarchical clustering (single linkage)
    3. Quasi-diagonalization of covariance matrix
    4. Recursive bisection to allocate weights

    Benefits:
    - No matrix inversion (stable)
    - Handles small sample sizes
    - Intuitive cluster-based allocation
    """
```

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `fit()` | `returns: DataFrame` | `self` | Fit on historical returns |
| `allocate()` | `tickers: List[str]` | `Dict[str, float]` | Compute HRP weights |
| `get_cluster_structure()` | - | `Dict` | Return dendrogram info |

---

## Options Module
**Location**: `backend/app/options/`

### types.py
Options-specific types.

**Enum: `OptionsMode`**
```python
class OptionsMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"
    SHADOW = "shadow"
```

**Enum: `GateReasonCode`**
```python
class GateReasonCode(str, Enum):
    PASSED = "PASSED"
    IV_TOO_LOW = "IV_TOO_LOW"
    IV_TOO_HIGH = "IV_TOO_HIGH"
    LIQUIDITY_FAIL = "LIQUIDITY_FAIL"
    UNCERTAINTY_HIGH = "UNCERTAINTY_HIGH"
    TREND_WEAK = "TREND_WEAK"
    DTE_INVALID = "DTE_INVALID"
```

**Dataclass: `SpreadCandidate`**
```python
@dataclass
class SpreadCandidate:
    underlying: str
    short_strike: float
    long_strike: float
    expiration: date
    credit: float
    max_loss: float
    probability_of_profit: float
    delta: float
```

**Dataclass: `OptionsDecision`**
```python
@dataclass
class OptionsDecision:
    action: str  # "OPEN" | "CLOSE" | "HOLD"
    spread: Optional[SpreadCandidate]
    reason: str
    gate_codes: List[GateReasonCode]
```

---

### config.py
Options configuration.

```python
@dataclass
class OptionsConfig:
    mode: OptionsMode = OptionsMode.PAPER
    min_dte: int = 21
    max_dte: int = 45
    target_delta: float = 0.15
    min_credit: float = 0.30
    max_loss_pct: float = 0.05
    min_pop: float = 0.70  # Probability of profit
```

---

### gate/gate.py
Multi-stage options gating.

**Class: `OptionsGate`**

```python
class OptionsGate:
    """
    Multi-stage gate for options trades.

    Stages:
    1. IV Gate: Check implied volatility range
    2. Liquidity Gate: Check bid-ask spread, volume
    3. Uncertainty Gate: Check teacher confidence
    4. Trend Gate: Check underlying trend strength
    5. DTE Gate: Check days to expiration
    """
```

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `evaluate()` | `context: GateContext` | `GateDecision` | Run all gates |
| `get_gate_status()` | - | `Dict[str, bool]` | Status of each gate |

---

### gate/context.py
Gate evaluation context.

**Dataclass: `GateContext`**
```python
@dataclass
class GateContext:
    underlying: str
    current_price: float
    iv_rank: float
    iv_percentile: float
    bid_ask_spread: float
    volume: int
    open_interest: int
    teacher_confidence: float
    trend_strength: float
    dte: int
```

---

### gate/features.py
Feature extraction for gate evaluation.

| Function | Parameters | Returns | Purpose |
|----------|------------|---------|---------|
| `compute_iv_rank()` | `iv_history, current_iv` | `float` | IV rank [0-100] |
| `compute_iv_percentile()` | `iv_history, current_iv` | `float` | IV percentile [0-100] |
| `compute_liquidity_score()` | `bid_ask, volume, oi` | `float` | Liquidity score [0-1] |

---

### strategy/strike_selector.py
Put credit spread strike selection.

**Class: `StrikeSelector`**

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `select()` | `chain: DataFrame, config: OptionsConfig` | `List[SpreadCandidate]` | Find valid spreads |
| `score_spread()` | `spread: SpreadCandidate` | `float` | Score spread quality |
| `get_best()` | `candidates: List[SpreadCandidate]` | `SpreadCandidate` | Select best spread |

---

### strategy/vibe_check.py
IV and liquidity gates.

| Function | Parameters | Returns | Purpose |
|----------|------------|---------|---------|
| `check_iv_environment()` | `iv_rank, iv_pct` | `Tuple[bool, GateReasonCode]` | IV gate check |
| `check_liquidity()` | `bid_ask, volume` | `Tuple[bool, GateReasonCode]` | Liquidity check |
| `check_uncertainty()` | `teacher_conf` | `Tuple[bool, GateReasonCode]` | Uncertainty check |

---

### strategy/spread_rules.py
Spread construction rules.

| Function | Parameters | Returns | Purpose |
|----------|------------|---------|---------|
| `validate_spread_width()` | `short, long, underlying` | `bool` | Check width rules |
| `calculate_pop()` | `spread, iv` | `float` | Probability of profit |
| `calculate_expected_value()` | `spread` | `float` | Expected P&L |

---

### data/iv_store.py
IV data storage and retrieval.

**Class: `IVStore`**

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `get_iv_history()` | `ticker, lookback` | `Series` | Historical IV |
| `get_current_iv()` | `ticker` | `float` | Current IV |
| `update()` | `ticker, iv, timestamp` | - | Store new IV reading |

---

### data/chain_store.py
Options chain data management.

**Class: `ChainStore`**

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `get_chain()` | `ticker, expiration` | `DataFrame` | Get options chain |
| `get_expirations()` | `ticker` | `List[date]` | Available expirations |
| `update()` | `ticker, chain_df` | - | Store chain snapshot |

---

### execution/executor.py
Base executor interface.

**Abstract Class: `Executor`**

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `execute()` | `decision: OptionsDecision` | `FillEvent` | Execute trade |
| `get_positions()` | - | `List[OptionsPosition]` | Current positions |
| `close_position()` | `position_id` | `FillEvent` | Close specific position |

---

### execution/paper_executor.py
Paper trading executor.

**Class: `PaperExecutor(Executor)`**
- Simulates fills at mid-price
- Tracks simulated positions
- No real money at risk

---

### execution/noop_executor.py
No-op executor for shadow mode.

**Class: `NoopExecutor(Executor)`**
- Logs decisions without executing
- Used for signal monitoring

---

### sim/credit_spread_sim.py
Credit spread simulation for backtesting.

**Class: `CreditSpreadSim`**

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `run()` | `trades: List[SpreadCandidate], prices: DataFrame` | `SimResults` | Run backtest |
| `compute_metrics()` | `results: SimResults` | `Dict` | Performance metrics |

---

## Preprocessing Module
**Location**: `backend/app/preprocessing/`

### preproc.py
Feature engineering preprocessor.

**Class: `Preprocessor`**

```python
class Preprocessor:
    """
    Feature Engineering Pipeline:

    Input columns: timestamp, open, high, low, close, volume, ad_line, bpi

    Output features:
    - log_return_1d: log(close_t / close_{t-1})
    - log_return_5d: log(close_t / close_{t-5})
    - log_return_20d: log(close_t / close_{t-20})
    - volatility_20d: 20-day rolling std of log returns
    - volume_log_change_5d: log1p(volume).diff(5)
    - ad_line_trend_5d: ad_line.diff(5)
    - bpi_level: raw BPI passthrough
    """
```

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `fit()` | `df: DataFrame` | `self` | Compute normalization params |
| `transform()` | `df: DataFrame` | `DataFrame` | Apply feature engineering |
| `attach_teacher_priors()` | `feature_df, priors_df` | `DataFrame` | Join teacher priors |
| `save()` | `path: str` | - | Persist fitted params |
| `load()` | `path: str` | `Preprocessor` | Load fitted preprocessor |

---

### versioning.py
Artifact versioning utilities.

| Function | Parameters | Returns | Purpose |
|----------|------------|---------|---------|
| `compute_hash()` | `data: Dict` | `str` | SHA256 of dict |
| `get_schema_hash()` | `columns: List` | `str` | MD5 of column list |
| `check_compatibility()` | `required_id, current_id` | - | Raise on mismatch |
| `get_priors_hash()` | `priors_meta: Dict` | `str` | Hash teacher priors |
| `get_selector_hash()` | `selector_config: Dict` | `str` | Hash selector config |

---

## Evaluation Module
**Location**: `backend/app/eval/`

### metrics.py
Model evaluation metrics.

| Function | Parameters | Returns | Purpose |
|----------|------------|---------|---------|
| `directional_accuracy()` | `predictions, actuals` | `float` | % correct direction |
| `coverage_probability()` | `intervals, actuals` | `float` | % actuals in interval |
| `interval_width()` | `intervals` | `float` | Mean interval width |
| `rank_correlation()` | `predicted_ranks, actual_ranks` | `float` | Spearman correlation |
| `top_k_precision()` | `predictions, actuals, k` | `float` | Precision@K |

---

### promotion_gate.py
Model promotion criteria.

**Class: `PromotionGate`**

```python
class PromotionGate:
    """
    Promotion Criteria:
    - directional_accuracy >= 0.52
    - coverage_probability >= 0.80
    - rank_correlation >= 0.10
    - top_5_precision >= 0.25
    """
```

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `check()` | `metrics: Dict` | `Tuple[bool, List[str]]` | Check all criteria |
| `get_failures()` | `metrics: Dict` | `List[str]` | List failed criteria |

---

## Inference Module
**Location**: `backend/app/inference/`

### spec_decode_options.py
Speculative decoding configuration for options inference.

```python
@dataclass
class SpecDecodeConfig:
    draft_model: str        # Fast draft model
    target_model: str       # Accurate target model
    gamma: int = 4          # Speculation depth
    temperature: float = 0.7
```

---

## Scripts
**Location**: `backend/scripts/`

### Training Scripts
| Script | Purpose |
|--------|---------|
| `train/student_baseline_train.py` | Train baseline MLP student |
| `train_selector.py` | Train RankTransformer selector |
| `train_chronos2_teacher.py` | Fine-tune Chronos teacher with LoRA |
| `fit_scaler.py` | Fit SelectorFeatureScaler |
| `fit_calibrator.py` | Fit ScoreCalibrator |

### Execution Scripts
| Script | Purpose |
|--------|---------|
| `run/run_swing_equities.py` | Run equity swing trading simulation |
| `run/run_swing_hybrid_monitor.py` | Run with options overlay |
| `run_options_only.py` | Run options-only mode |

### Data Scripts
| Script | Purpose |
|--------|---------|
| `download_data.py` | Download market data |
| `canonicalize/canonicalize_marketframe.py` | Build MarketFrame parquets |
| `compute_breadth.py` | Compute breadth indicators |
| `generate_teacher_priors.py` | Generate teacher priors |

### Evaluation Scripts
| Script | Purpose |
|--------|---------|
| `evaluate_selector.py` | Evaluate selector performance |
| `backtest_swing.py` | Backtest swing strategy |
| `compute_metrics.py` | Compute evaluation metrics |

---

## Tests
**Location**: `backend/tests/`

### Core Tests
| Test File | Coverage |
|-----------|----------|
| `test_preproc.py` | Preprocessor fit/transform |
| `test_baseline.py` | BaselineMLP forward pass |
| `test_windows.py` | SwingWindowDataset |
| `test_risk_manager.py` | RiskManager decisions |
| `test_portfolio_construction.py` | PortfolioBuilder selection |
| `test_hrp.py` | HRP allocation |

### Options Tests
| Test File | Coverage |
|-----------|----------|
| `test_options_gate.py` | Multi-stage gating |
| `test_strike_selector.py` | Spread selection |
| `test_vibe_check.py` | IV/liquidity gates |
| `test_paper_executor.py` | Paper execution |

---

## Data Flow Diagram

```
                                    ┌─────────────────┐
                                    │   Market Data   │
                                    │   (Parquet)     │
                                    └────────┬────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            PREPROCESSING                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐ │
│  │ MarketFrame │───▶│ Preprocessor│───▶│ Features (10-dim vector)    │ │
│  └─────────────┘    └─────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      │                                             │
                      ▼                                             │
        ┌─────────────────────────┐                                 │
        │      CHRONOS-2 (Nightly) │                                 │
        │  ┌─────────────────────┐ │                                 │
        │  │   Chronos2Teacher   │ │                                 │
        │  │   (T5 + LoRA)       │ │                                 │
        │  └──────────┬──────────┘ │                                 │
        │             │            │                                 │
        │             ▼            │                                 │
        │  ┌─────────────────────┐ │                                 │
        │  │   ChronosPriors     │ │                                 │
        │  │   (parquet)         │ │                                 │
        │  └──────────┬──────────┘ │                                 │
        └─────────────┼────────────┘                                 │
                      │                                             │
                      └───┐     ┌───────────────────────────────────┘
                          │     │
                          ▼     ▼
                        ┌──────────────┐
                        │   SELECTOR   │
                        │(RankTransf.) │
                        └──────┬───────┘
                               │
                               ▼
                       ┌───────────────┐
                       │  Leaderboard  │
                       └──────┬────────┘
                              │
                              ▼
                       ┌────────────────┐
                       │PortfolioManager│
                       └──────┬─────────┘
                              │
                              ▼
                       ┌────────────────┐
                       │   EquityPod    │
                       └────────────────┘
```

---

## Options Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            OPTIONS POD                                   │
│                                                                          │
│  ┌──────────────┐                                                       │
│  │ Equity Signal │ (from EquityPod)                                     │
│  └──────┬───────┘                                                       │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                        OPTIONS GATE                               │   │
│  │  ┌─────────┐  ┌─────────┐  ┌───────────┐  ┌───────┐  ┌───────┐  │   │
│  │  │ IV Gate │──│Liquidity│──│Uncertainty│──│ Trend │──│  DTE  │  │   │
│  │  └─────────┘  └─────────┘  └───────────┘  └───────┘  └───────┘  │   │
│  └──────────────────────────────────┬───────────────────────────────┘   │
│                                     │                                    │
│                           ┌─────────┴─────────┐                         │
│                           │                   │                         │
│                     PASSED │             FAILED│                         │
│                           ▼                   ▼                         │
│                  ┌─────────────────┐  ┌─────────────┐                   │
│                  │ StrikeSelector  │  │ NO_ACTION   │                   │
│                  └────────┬────────┘  └─────────────┘                   │
│                           │                                              │
│                           ▼                                              │
│                  ┌─────────────────┐                                     │
│                  │ SpreadCandidate │                                     │
│                  │ (short, long,   │                                     │
│                  │  credit, POP)   │                                     │
│                  └────────┬────────┘                                     │
│                           │                                              │
│                           ▼                                              │
│                  ┌─────────────────┐                                     │
│                  │    Executor     │                                     │
│                  │ (Paper/Live/Noop)│                                    │
│                  └─────────────────┘                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Artifact Versioning

All model artifacts are versioned with SHA256 hashes for reproducibility:

```
backend/models/
├── preproc/
│   └── preproc_v1.json          # Preprocessor params + hash
├── student/
│   └── student_v1.pt            # Student weights + metadata
├── selector/
│   ├── selector_v1.pt           # Selector checkpoint
│   ├── scaler_v1.json           # Fitted scaler
│   └── calibrator_v1.pkl        # Fitted calibrator
├── teacher/
│   └── chronos2_lora_v1/        # LoRA adapter weights
└── priors/
    └── 2024-01-15.parquet       # Daily teacher priors (all tickers)
```

---

## Configuration Reference

### Environment Variables
| Variable | Default | Purpose |
|----------|---------|---------|
| `ENABLE_OPTIONS` | `false` | Enable options trading |
| `OPTIONS_MODE` | `paper` | Options execution mode |
| `EXECUTION_MODE` | `LEGACY` | Equity execution mode |
| `REBALANCE_CALENDAR` | `weekly` | Rebalance frequency |

### Key Thresholds
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `TOP_K` | 5 | Number of positions |
| `MIN_HOLD_DAYS` | 3 | Minimum hold period |
| `BPI_CRASH_THRESHOLD` | 30 | Defensive trigger |
| `TARGET_DELTA` | 0.15 | Options delta target |
| `MIN_POP` | 0.70 | Min probability of profit |

---

*Document generated from comprehensive codebase analysis.*
*Last updated: 2026-02-04*
