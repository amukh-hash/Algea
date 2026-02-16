# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Algaie** is an institutional-grade algorithmic trading platform implementing a **Chronos-2 Priors + Rank-Transformer Selector** architecture for swing trading equities and options (put credit spreads).

### Core Architecture

1. **Chronos-2 Teacher** (T5 + LoRA) generates probabilistic priors (drift, volatility, trend confidence, downside risk) for 10-30 trading day horizons
2. **Rank-Transformer Selector** combines Chronos priors with market features to rank equities cross-sectionally
3. **Portfolio Construction** uses Top-K selection with Hierarchical Risk Parity (HRP) allocation
4. **Risk Management** applies multi-stage gating (risk posture, crash override, options gate)

### Technology Stack

- **Python 3.11+** with type hints
- **PyTorch** for neural networks
- **Polars** for high-performance DataFrames (prefer over pandas)
- **HuggingFace Transformers + PEFT/LoRA** for Chronos models
- **exchange_calendars** for NYSE trading calendar (all horizons are trading-day aligned)
- **scikit-learn** for calibration and clustering

## Development Commands

### Testing & Quality
```bash
# Run all tests
make test
# or
pytest

# Run specific test file
pytest backend/tests/test_chronos_priors.py

# Format code (black)
make fmt

# Lint code (ruff)
make lint
```

### Virtual Environment
```bash
# The project uses a virtual environment at .venv/
source .venv/bin/activate  # Unix
.venv\Scripts\activate     # Windows
```

### Python Path
Set `PYTHONPATH` to the backend directory before running scripts:
```bash
export PYTHONPATH="$PWD/backend"  # Unix
set PYTHONPATH=%CD%\backend       # Windows
```

## Directory Structure

### Core Code Locations

- **`algaie/`** - Core library package
  - `core/` - Configuration, artifacts, logging, paths
  - `data/` - Data loading, validation (canonical, eligibility, features, market, options)
  - `models/` - Neural network models and inference
  - `portfolio/` - Portfolio construction and HRP allocation
  - `risk/` - Risk management and posture
  - `strategies/` - Trading strategies (options VRP, etc.)

- **`backend/`** - Application code and scripts
  - `app/` - Application modules (data, evaluation, execution, features, monitoring, portfolio, risk, strategies)
  - `scripts/` - Operational scripts organized by purpose:
    - `canonicalize/` - Data canonicalization
    - `eligibility/` - Universe building
    - `features/` - Feature engineering
    - `priors/` - Chronos teacher priors generation
    - `ranker/` - Selector model training
    - `paper/` - Paper trading
    - `live/` - Live trading
    - `run/` - Nightly/weekly production runs
    - `verify/` - Validation and QA
  - `tests/` - Test suite

- **`data_lake/`** - Data storage (not in version control)
- **`deprecated/`** - Legacy code (reference only)
- **`legacy/`** - Old versions (archived)

## Key Operational Sequence (from RUNBOOK.md)

### Data Pipeline (Historical Setup)
1. **Universe Definition** - Define liquid swing universe (350-500 names, monthly rebalance)
2. **Data Ingestion** - Fetch and canonicalize daily OHLCV (adjusted for splits/dividends)
3. **Covariates & Breadth** - Compute market regime indicators (AD line, BPI, VIX proxy)
4. **Feature Engineering** - Build FeatureFrame (returns, volatility, volume, relative strength)
5. **Chronos Teacher Priors** - Generate distributional priors (drift, vol, trend_conf, tail_risk)
6. **Selector Dataset** - Build training dataset with forward return labels (5-10 day horizons)
7. **Selector Training** - Train RankTransformer with ranking loss (walk-forward, purged splits)

### Nightly/Weekly Production
```bash
# Nightly (after market close)
python backend/scripts/run/run_nightly_cycle.py --asof 2025-01-02

# Outputs: leaderboard with full provenance (feature/prior/model versions)
```

### Paper Trading
```bash
# Run paper trading cycle with IBKR
python backend/scripts/paper/run_paper_cycle_ibkr.py
```

## Critical Design Principles

### Trading Calendar Awareness
- **All horizons are NYSE trading days**, not calendar days
- Use `exchange_calendars` for date arithmetic
- Never assume 252 days/year - compute actual trading days
- Scripts should iterate over trading days, not calendar days

### Causality & No Leakage
- All features at time `t` use only data `≤ t`
- Training uses walk-forward with purged/embargoed splits
- Forward returns computed using `shift_trading_days(t, +H_sel)`

### Artifact Versioning
All model artifacts are versioned with SHA256 hashes:
```
backend/models/
├── preproc/preproc_v1.json          # Preprocessor params + hash
├── selector/selector_v1.pt          # Selector checkpoint
├── selector/scaler_v1.json          # Fitted scaler
├── selector/calibrator_v1.pkl       # Fitted calibrator
├── teacher/chronos2_lora_v1/        # LoRA adapter weights
└── priors/2024-01-15.parquet        # Daily teacher priors
```

### Data Format
- Prefer **Polars** over pandas for performance
- Store data as **Parquet** with partitioning (e.g., `ticker=AAPL/data.parquet`)
- Always include provenance columns: `feature_version`, `prior_version`, `model_version`

## Model Architecture Details

### Chronos-2 Teacher
- **Purpose**: Generate probabilistic priors, not trade signals
- **Outputs**: drift_20d, vol_20d, trend_conf_20d, downside_q10_20d
- **Training**: Light LoRA tuning monthly/quarterly (not nightly)
- **Context**: 60-250 daily bars
- **Horizon**: 10-30 trading days

### RankTransformer Selector
- **Input**: [B, T, F] where T=20-60 daily bars, F=10 features
  - Market features: log returns (1d/5d/20d), volatility, volume changes
  - Teacher priors: drift, vol, downside risk, trend confidence
  - Breadth: AD line trend, BPI level
- **Output**: {"score": [B, 1], "p_up": [B, 1]}
- **Architecture**: Encoder-only Transformer (d_model=64, nhead=4, num_layers=2)
- **Loss**: Multi-task (ranking + direction) with pairwise/listwise ranking loss

### Portfolio Construction
- **Top-K Selection**: Default K=5 positions
- **Allocation**: Hierarchical Risk Parity (HRP) based on correlation clustering
- **Rebalancing**: Weekly or 2-3x per week
- **Constraints**:
  - `MIN_HOLD_DAYS = 3` - Minimum hold period
  - `MAX_HOLD_DAYS = 10` - Maximum hold period
  - `TURNOVER_LIMIT = 0.3` - Max daily turnover

## Risk Management

### Risk Posture States
- **NORMAL**: Full risk budget (100% position size)
- **CAUTIOUS**: Reduced sizing (50% position size)
- **DEFENSIVE**: No new entries, reduce exposure only

### Crash Override Thresholds
```python
BPI_CRASH_THRESHOLD = 30.0    # BPI < 30 → DEFENSIVE
BPI_CAUTION_THRESHOLD = 40.0  # BPI < 40 → CAUTIOUS
AD_CRASH_THRESHOLD = -500     # AD Line decline threshold
```

### Options Gate (Multi-Stage)
When `ENABLE_OPTIONS = true`:
1. **IV Gate**: Check implied volatility rank/percentile
2. **Liquidity Gate**: Check bid-ask spread and volume
3. **Uncertainty Gate**: Check teacher confidence
4. **Trend Gate**: Check underlying trend strength
5. **DTE Gate**: Check days to expiration (21-45 days target)

Options config:
- `TARGET_DELTA = 0.15` - Delta target for put credit spreads
- `MIN_POP = 0.70` - Minimum probability of profit
- `MIN_CREDIT = 0.30` - Minimum credit per spread

## Configuration

### Environment Variables (backend/app/core/config.py)
- `ENABLE_OPTIONS` (bool, default: False) - Toggle options trading
- `OPTIONS_MODE` ("paper" | "live" | "shadow", default: OFF)
- `EXECUTION_MODE` ("LEGACY" | "SHADOW" | "RANKING", default: LEGACY)
- `REBALANCE_CALENDAR` (str) - Rebalance frequency

### Key Feature Contracts
```python
SELECTOR_FEATURE_CONTRACT = [
    "log_return_1d", "log_return_5d", "log_return_20d",
    "volatility_20d", "volume_log_change_5d",
    "ad_line_trend_5d", "bpi_level",
    "teacher_drift_20d", "teacher_vol_20d", "teacher_downside_q10_20d"
]
```

## Testing Approach

- **Unit tests** for individual components (preprocessor, models, risk manager)
- **Integration tests** for pipelines (test_cooc_pipeline.py, test_portfolio_backtest.py)
- **Validation gates** must pass before promotion:
  - Schema validation (canonical data, features, priors)
  - Coverage gates (≥95% completeness)
  - Determinism checks (rebuild hash consistency)
  - Data sanity (no split discontinuities, outlier detection)

### Key Test Files
- `test_chronos_priors.py` - Chronos teacher validation
- `test_cooc_phase15_alignment.py` - Pipeline alignment checks
- `test_portfolio_backtest.py` - Portfolio construction and backtesting
- `test_sharpe_validator.py` - Sharpe ratio and performance validation
- `test_ibkr_broker.py` - IBKR integration testing

## Common Gotchas

1. **Trading Calendar**: Always use NYSE calendar for date arithmetic. Never assume 252 trading days or use calendar day offsets.

2. **Data Versioning**: When modifying feature engineering or priors, increment version hashes and update PROD_POINTER.json.

3. **Polars vs Pandas**: This codebase prefers Polars for performance. Be mindful of API differences (e.g., `pl.col()` vs `df['col']`).

4. **No Stub Data**: Universe manifests, FeatureFrame, and priors builders must use real trading calendar iteration, not hardcoded symbol lists or calendar days.

5. **Causality**: When adding features, ensure they only use data ≤ t. Forward-looking leakage will invalidate backtests.

6. **Ranking Loss**: The selector uses cross-sectional ranking loss, not just classification. Preserve date grouping for listwise ranking.

## Workflow Integration

This project uses a GitHub-centric workflow:
- **GitHub** - Source of truth for code
- **Google Jules** - Remote AI agent for complex async tasks
- **Google Antigravity** - Local IDE with AI assistance

See `WORKFLOW.md` for details on branch management and PR process.

## Documentation References

- **`ARCHITECTURE.md`** - Comprehensive module documentation (1265 lines)
- **`RUNBOOK.md`** - Operational procedures and pipeline sequence
- **`WORKFLOW.md`** - Git workflow and tooling integration
- **`CODEBASE_WALKTHROUGH.md`** - Legacy reference (may be outdated)

## Important: Do Not

- Change trading calendar logic without careful review
- Mix calendar days and trading days in horizon calculations
- Use pandas where Polars is already established
- Skip validation gates when modifying data pipelines
- Hardcode symbol lists (use universe manifests)
- Train on unadjusted prices (always use split/dividend adjusted)
