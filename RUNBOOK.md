# Algaie Operational Runbook (Data Pipeline + Training)

This runbook documents the current end-to-end flow for the **data pipeline** and **selector model training**. It reflects the script structure in `backend/scripts/` and highlights known caveats so the pipeline can be executed safely and in the right order.

> **Important:** Several steps still rely on stubbed implementations (featureframe, marketframe, dataset builder, nightly pipeline). Those steps will execute but may produce empty or mocked outputs until the stubs are fully implemented.

---

## 1) Environment Setup

1. **Install dependencies** (from repo root):
   ```bash
   pip install -r requirements.txt
   ```
   The requirements file includes the packages used across the pipeline (e.g., `pandas_market_calendars`, `tqdm`, `transformers`, `peft`, `bitsandbytes`).【F:requirements.txt†L1-L21】

2. **Set project root** for Python imports (from repo root):
   ```bash
   export PYTHONPATH="$PWD/backend"
   ```

3. **Ensure data directories exist**  
   Most scripts call the bootstrap helper to create directories for artifacts and logs.【F:backend/scripts/01_backfill_canonical_daily.py†L21-L26】

---

## Training Protocol (Data Requirements & Cadence)

### A) Selector training data (Rank-Transformer + calibration)
- **Daily per-ticker OHLCV (adjusted for splits/dividends)** with at least 8–12 years of history where possible.
- **Daily market/regime covariates**: SPY/QQQ/IWM returns/vol, VIX level/returns, rates proxy (2Y/10Y or IEF/TLT), and preferred breadth features (AD line, BPI).
- **Labels**: primary label is forward 10-trading-day close-to-close return; optional auxiliary label is 10-day direction (up/down).
- **Corporate actions**: adjusted prices/returns are required; unadjusted series will pollute training.

### B) Chronos-2 priors (nightly generation)
- **Inputs**: same daily per-ticker series and covariates as training.
- **Outputs per date (all tickers)**: drift proxy, forecast dispersion/vol proxy, downside quantile (q10), and trend confidence (stored in priors parquet for selector features).
- Chronos priors are **signals inputs**, not trade signals.

### C) Nightly execution inputs
- Latest daily bar per ticker, updated breadth/context, updated covariates.
- Then generate priors for `as_of_date`, run selector inference, and write leaderboard.

### D) Chronos-2 training cadence
- **Recommended**: Chronos-2 is mostly inference-only.
- Fit the **Chronos codec once**, run nightly inference for priors, and only fine-tune (LoRA) occasionally if it improves calibration.
- If fine-tuning: do **monthly/quarterly** LoRA refreshes on the full historical daily dataset; treat it as adaptation, not alpha training.

---

## 2) Canonical Data Ingestion & Backfill (Daily Bars)

**Goal:** Create B2 OHLCV partitions in `backend/data_canonical/ohlcv_adj/...`.

1. **Fetch raw data from Alpaca (legacy pipeline)**  
   This writes a legacy raw OHLCV file used by the backfill step.
   ```bash
   python backend/scripts/ingest/fetch_data.py --start 2016-01-01 --end 2025-12-31
   ```
   (Script writes to `backend/data/artifacts/universe/raw_ohlcv.parquet` by default.)【F:backend/scripts/ingest/fetch_data.py†L1-L189】

2. **Backfill canonical partitions**
   ```bash
   python backend/scripts/01_backfill_canonical_daily.py --start 2016-01-01 --end 2025-12-31
   ```
   This reads the legacy raw OHLCV file and writes canonical partitions by ticker.【F:backend/scripts/01_backfill_canonical_daily.py†L23-L61】

---

## 3) Build Universe Manifests (B5)

**Goal:** Generate monthly universe snapshots using security master + legacy metrics.

```bash
python backend/scripts/02_build_universe_manifests.py --start 2016-01-01 --end 2025-12-31
```

This script loads legacy universe snapshots when available and writes manifest files under `backend/manifests/`.【F:backend/scripts/02_build_universe_manifests.py†L38-L87】【F:backend/app/ops/pathmap.py†L66-L88】

---

## 4) Build FeatureFrame (B6)

**Goal:** Create a canonical featureframe for training/inference.

```bash
python backend/scripts/03_build_featureframe.py --start 2016-01-01 --end 2025-12-31
```

**Notes / Caveats**
- The featureframe builder computes B6 features directly from canonical OHLCV partitions and will **fail hard** if required covariates or breadth files are missing. Ensure `backend/data_canonical/covariates_daily.parquet` and `backend/data_canonical/breadth_daily.parquet` exist before running.【F:backend/app/features/featureframe.py†L32-L166】
- FeatureFrame writes deterministic `feature_version`/`data_version` columns and enforces B6 schema validation before write, so re-runs with the same inputs produce identical hashes.【F:backend/app/features/featureframe.py†L142-L194】

---

## 5) Generate Historical Priors (B7)

**Goal:** Produce Chronos priors for each date in the training period.

```bash
python backend/scripts/04_generate_priors_historical.py --start 2016-01-01 --end 2025-12-31
```

**Notes / Caveats**
- The script iterates **NYSE trading days** and uses universe manifests when available; otherwise it falls back to the union of manifest symbols or the security master universe.【F:backend/scripts/04_generate_priors_historical.py†L29-L99】
- Priors are generated using Chronos if available; otherwise the runner uses a deterministic statistical fallback derived from recent returns (not random).【F:backend/app/teacher/chronos_runner.py†L13-L74】

---

## 6) Build Selector Dataset (B8 + tensors)

**Goal:** Build labels and the ranker dataset used for training.

```bash
python backend/scripts/05_build_selector_dataset.py --start 2016-01-01 --end 2025-12-31
```

**Notes / Caveats**
- The dataset builder constructs real forward-10TD labels from canonical OHLCV and builds listwise sequences from FeatureFrame + priors, but will skip dates that fail the minimum group size filter. Ensure priors and featureframe artifacts exist for the date range before running.【F:backend/app/selector/dataset_builder.py†L12-L191】
- Group metadata (per-date symbol lists) is persisted alongside the dataset for traceability.【F:backend/scripts/05_build_selector_dataset.py†L24-L47】

---

## 7) Train the Selector (Ranker)

**Goal:** Train the ranker model after features/priors/labels exist.

```bash
python backend/scripts/06_train_selector.py --train_end 2025-12-31
```

**Notes / Caveats**
- The training wrapper now trains the ranker directly from the selector dataset artifact and writes the model, scaler, calibrator, and PROD pointer. Ensure the dataset artifact exists before running.【F:backend/app/selector/train.py†L5-L88】

If you need the legacy trainer directly:
```bash
python backend/app/training/trainer.py
```
The trainer currently expects listwise scores but the model returns a pooled scalar, so it needs alignment before use.【F:backend/app/training/trainer.py†L25-L58】【F:backend/app/models/rank_transformer.py†L100-L127】

---

## 8) Nightly Run (Inference Pipeline)

**Goal:** Generate daily leaderboard signals.

```bash
python backend/scripts/07_nightly_run.py --asof 2025-01-02
```

**Notes / Caveats**
- The nightly script loads featureframe and priors artifacts using the PROD pointer and uses the manifest (or security master) for the universe. Ensure the required artifacts exist for the `asof` date before running.【F:backend/scripts/07_nightly_run.py†L39-L70】

---

## 9) Validation & QA (Optional)

You can use the validation script to inspect artifact integrity:
```bash
python backend/scripts/validate_phase1_artifacts.py
```
This is useful once the dataset/featureframe/priors are real outputs.

---

## 10) Windows Helper (Optional)

If you are on Windows, the training helper provides a simple menu:
```
run_training.bat
```
It supports global base training and finetuning workflows.【F:run_training.bat†L1-L67】
