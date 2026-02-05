# Algaie Operational Runbook (Swing Trading: Chronos Teacher → Stock-Transformer Selector)

**Status:** Operational Runbook (Target)
**Scope:** Data ingestion, preprocessing, teacher priors, selector training, portfolio construction, and nightly/weekly execution.
**Primary timeframe:** **Daily**
**Optional overlay:** **4H** (for distillation features and regime/state only; not required for v1)

> **Non-negotiables:**
>
> 1. All horizons are **trading-day** aligned (NYSE calendar).
> 2. All features are **causal** (≤ t).
> 3. Training uses **walk-forward** with **purged/embargoed splits**.
> 4. Evaluation is **cost-aware** (turnover + slippage; borrow costs if short).

---

## 0) Objectives & Prediction Targets (Critical)

### Bars

* **Primary:** Daily OHLCV (adjusted)
* **Optional:** 4H OHLCV (for overlay/distillation features only)

### Horizons

* **Chronos teacher horizon (H_teacher):** **10–30 trading days**
* **Selector horizon (H_sel):** **5–10 trading days**

### Targets (Selector)

Produce targets that match swing trading:

* **Forward return distribution** over **5–10 trading days**
* **Direction probability** over **5–10 trading days**
* **Forward realized volatility** over the same horizon (risk sizing)

**Avoid:** next-bar classification (noisy, increases turnover, weak for swing).

---

## 1) Environment Setup

1. Install dependencies (repo root)

```bash
pip install -r requirements.txt
```

2. Set imports (repo root)

```bash
export PYTHONPATH="$PWD/backend"
```

3. Ensure directories exist (bootstrap)

* Data roots:

  * `backend/data_raw/`
  * `backend/data_canonical/`
  * `backend/features/`
  * `backend/priors/`
  * `backend/datasets/`
  * `backend/models/`
  * `backend/calibration/`
  * `backend/manifests/`
  * `backend/outputs/`
  * `backend/logs/`

> All scripts must call the bootstrap helper first and fail fast if paths aren’t writable.

---

## 2) Universe Definition (Liquid Swing Universe)

**Universe target:** 350–500 liquid names
**Rebalance cadence:** monthly (or quarterly), not daily

**Eligibility filters (exact)**

* Common equity only (exclude ADRs, preferreds, ETFs/ETNs/CEFs)
* Price ≥ $5 (last close at rebalance date)
* IPO age ≥ 252 trading days
* Liquidity: median 20-day **$ADV ≥ $25M**
* Optional:

  * median 20-day volume ≥ 1M shares

**Sector balance**

* Max sector weight (e.g., 25%) or minimum sector coverage

**Outputs**

* `backend/manifests/universe_asof=YYYY-MM-DD.parquet`

  * includes `reason_code`, `adv20_median`, `ipo_age_td`, `sector`, `universe_version`

**Command**

```bash
python backend/scripts/02_build_universe_manifests.py --start 2016-01-01 --end 2025-12-31 --freq monthly
```

---

## 3) Data Ingestion & Canonicalization (Daily; optional 4H)

### 3.1 Daily OHLCV (Adjusted)

**Goal:** Write canonical partitioned adjusted daily bars.

**Step A — Fetch raw daily bars (legacy ingest if needed)**

```bash
python backend/scripts/ingest/fetch_data.py --start 2016-01-01 --end 2025-12-31
```

**Step B — Backfill canonical daily partitions**

```bash
python backend/scripts/01_backfill_canonical_daily.py --start 2016-01-01 --end 2025-12-31
```

**Canonical output**

* `backend/data_canonical/ohlcv_adj/ticker=XYZ/data.parquet`

**Hard requirement**

* Adjusted prices (splits/dividends). No unadjusted training.

### 3.2 Optional: 4H OHLCV (Overlay / Distillation only)

If you add 4H:

* Store under: `backend/data_canonical/ohlcv_4h/ticker=XYZ/data.parquet`
* Use only for:

  * intraday state signals (p_up_h1 / p_up_h4)
  * regime embeddings
  * trend persistence proxies

Not required for v1.

---

## 4) Covariates + Breadth (Daily; causal)

**Goal:** Provide minimal, stable market regime covariates used by teacher and selector.

Daily covariates (examples):

* SPY/QQQ/IWM returns + 20d vol
* VIX (level/returns) or proxy
* Rates proxy (IEF/TLT or yields)
* Breadth: AD line, BPI (your set)

**Outputs**

* `backend/data_canonical/covariates_daily.parquet`
* `backend/data_canonical/breadth_daily.parquet`

---

## 5) Feature Engineering (Selector Inputs: robust + small set)

### 5.1 FeatureFrame (Daily; causal)

**Goal:** Build a stable FeatureFrame used by the selector and dataset builder.

Per ticker-day inputs (minimum recommended):

* Returns: 1/3/5/10d (or 1/5/20 if you prefer)
* Volatility: ATR or realized vol (20d)
* Volume & dollar volume features
* Relative strength vs market and/or sector
* Join market covariates + breadth

**Outputs**

* `backend/features/featureframe_v{FEATURE_VERSION}.parquet`

  * includes `feature_version` and `data_version`

**Command**

```bash
python backend/scripts/03_build_featureframe.py --start 2016-01-01 --end 2025-12-31
```

> **Causality rule:** anything used at date t uses only data ≤ t.

---

## 6) Chronos Teacher (Distributional Priors; lightly tuned)

### 6.1 Chronos role

Chronos is **not** your trade signal. It produces **distributional features** (“priors”):

* `drift` (expected return over H_teacher)
* `vol_forecast` (dispersion)
* `trend_conf` (P(drift>0) or slope/dispersion ratio)
* `tail_risk` (downside quantile, e.g., q10)

**Inputs (keep minimal)**

* Daily returns/log-prices
* Minimal volatility proxy (EWMA vol / ATR)
* Optional: rolling z-scores (causal)
* Optional: market covariates as past-only covariates

### 6.2 Light LoRA tuning (on initial historic data, not nightly)

* cadence: monthly/quarterly
* context: 60–250 daily bars (or 120–240 4H bars)
* horizon: 10–30 trading days (daily equivalent for 4H)
* objective: calibration/stability of priors, not “alpha”

### 6.3 Priors generation (historical + nightly)

**Outputs**

* `backend/priors/date=YYYY-MM-DD/priors_v{PRIOR_VERSION}.parquet`

  * includes `prior_version`, `chronos_model_id`, `context_len`, `horizon`

**Command**

```bash
python backend/scripts/04_generate_priors_historical.py --start 2016-01-01 --end 2025-12-31
```

**Must-haves**

* Iterate **trading days**, not calendar days
* Use manifest-derived symbols (not hardcoded)
* Enforce coverage gate (e.g., ≥95%)

---

## 7) Distillation Interface (Teacher → Student, optional)

You will standardize on **tradable objects** (stable interface).

### 7.1 Per-symbol intraday overlay state (optional 4H/intraday inputs)

For each ticker-day:

* `p_up_h1`, `p_up_h4`
* `p_trend_day` (trend continuation vs mean reversion proxy)
* `conf` (calibrated confidence)
* `expected_edge` (optional mapping prob → EV)
* `regime_vector` (embedding) (optional)
* `microstructure_pressure` (optional scalar)

### 7.2 Market state (portfolio gate)

* `risk_on_score`
* `risk_off_score`
* `liquidity_stress_score`
* `conf_market`

**Outputs**

* `backend/priors/overlay_state/date=YYYY-MM-DD/state.parquet` (optional)
* `backend/priors/market_state/date=YYYY-MM-DD/state.parquet` (optional)

> The key benefit: the daily student gains intraday awareness without needing minute/L2 live.

---

## 8) Selector Dataset Build (Targets + sequences + joins)

### 8.1 Labels (H_sel = 5–10 trading days)

Build:

* `fwd_ret_{H_sel}` (forward return)
* `fwd_up_{H_sel}` (direction)
* `fwd_vol_{H_sel}` (forward realized vol)

**Trading calendar:** use NYSE `shift_trading_days(t, +H_sel)`.

### 8.2 Inputs (Daily sequences)

Per ticker-day:

* sequence length: 20–60 daily bars (recommended)
* join:

  * FeatureFrame channels
  * Chronos priors channels (teacher outputs)
  * optional overlay state

### 8.3 Dataset outputs

* `backend/datasets/labels_fwd{H_sel}d.parquet`
* dataset tensors (X/y) + date-group index for listwise ranking

**Command**

```bash
python backend/scripts/05_build_selector_dataset.py --start 2016-01-01 --end 2025-12-31 --horizon 10 --seq_len 60
```

**Hard rules**

* No leakage (purged splits later)
* Minimum per-date group size threshold
* Save schema + provenance

---

## 9) Selector Training (Stock-Transformer cross-sectional ranker)

### 9.1 Model outputs (multi-task)

For each ticker-day:

* `p_up` over 5–10 days
* `expected_return` over 5–10 days (or return bucket)
* `expected_vol`

### 9.2 Loss (recommended)

Multi-task:

* Direction: cross-entropy or focal
* Return: Huber or quantile loss
* Add ranking loss:

  * pairwise hinge / pairwise logistic, or
  * listwise softmax / ListNet

> Cross-sectional edge typically comes from ranking loss + clean grouping by date.

### 9.3 Training protocol (non-negotiable)

* Walk-forward: rolling train → validate → test
* Purged/embargoed splits (avoid overlap leakage with horizon)
* Cost-aware evaluation:

  * turnover
  * slippage model
  * borrow costs if shorting

### 9.4 Metrics

* Rank IC (Spearman)
* Hit rate conditional on threshold (e.g., top decile)
* Strategy Sharpe / max drawdown after costs

**Command**

```bash
python backend/scripts/06_train_selector.py --train_end 2025-12-31 --horizon 10 --seq_len 60
```

**Outputs**

* Model checkpoint: `backend/models/rank_tf_{MODEL_VERSION}.pt`
* Scaler: `backend/models/scaler_{FEATURE_VERSION}.pkl`
* Calibration: `backend/calibration/cal_{CAL_VERSION}.pkl`
* `backend/models/PROD_POINTER.json` (only if promotion gate passes)

---

## 10) Portfolio Construction & Execution (Rule-based; ML stays out of micro-timing)

### Rebalance cadence

* weekly or 2–3× per week (recommended for swing)
* Universe: liquid names only

### Sizing

* volatility targeting or max position caps
* tilt by confidence:

  * size ∝ score / forecast_vol
    (uses selector score and teacher vol_forecast)

### Entry / Exit rules (deterministic)

* Enter: close/open, or pullback/breakout confirmation (optional)
* Exit:

  * time stop (7–10 trading days)
  * volatility stop (ATR)
  * score decay / regime flip

**Outputs**

* Orders file / execution plan artifact (your EquityPod layer)

---

## 11) Nightly / Weekly Production Run

### Nightly (daily close)

* Update latest daily bars + covariates + breadth
* Generate priors for as-of date (Chronos)
* Build latest features for as-of date
* Run selector inference
* Produce leaderboard

**Command**

```bash
python backend/scripts/07_nightly_run.py --asof 2025-01-02
```

**Output**

* `backend/outputs/leaderboard_date=YYYY-MM-DD.parquet`

  * includes full provenance: feature/prior/model/cal versions

### Weekly (recommended)

* Rebalance portfolio using leaderboard + rules
* Optionally retrain selector (expanding window) + recalibrate
* Promote only if gate passes

---

## 12) Validation & QA (Mandatory gates, not optional)

Run after each major build step (and automatically inside scripts):

* Schema validation (B2/B5/B6/B7/B8/B9)
* Coverage gates (priors/features completeness)
* Determinism checks (featureframe rebuild hash)
* Data sanity checks:

  * split discontinuities
  * outlier returns
  * missingness rates

**Command**

```bash
python backend/scripts/validate_phase1_artifacts.py
```

---

## 13) Known caveats (must be resolved before “real” training)

Before you trust outputs:

* No stubbed FeatureFrame / priors / dataset builder
* No hardcoded symbol lists
* Trading calendar iteration everywhere
* Trainer/model must align on ranking objective and shapes
* Provenance columns and `PROD_POINTER.json` required for live inference

---
