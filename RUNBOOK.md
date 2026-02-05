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
- The script currently migrates from a legacy feature file if present; otherwise it warns that the full build is not implemented.【F:backend/scripts/03_build_featureframe.py†L24-L57】
- The underlying featureframe builder is still a stub returning an empty DataFrame, so ensure legacy feature migration is available or implement the builder first.【F:backend/app/features/featureframe.py†L12-L40】

---

## 5) Generate Historical Priors (B7)

**Goal:** Produce Chronos priors for each date in the training period.

```bash
python backend/scripts/04_generate_priors_historical.py --start 2016-01-01 --end 2025-12-31
```

**Notes / Caveats**
- The script still uses a hardcoded 3-symbol universe (`AAPL`, `MSFT`, `GOOGL`). Replace with manifest-derived symbols for full coverage.【F:backend/scripts/04_generate_priors_historical.py†L45-L55】
- It iterates every calendar day, not trading days. Consider switching to trading calendars to avoid weekend/holiday writes.【F:backend/scripts/04_generate_priors_historical.py†L40-L63】
- Priors are currently generated via a stub runner returning mock values.【F:backend/app/teacher/chronos_runner.py†L6-L37】

---

## 6) Build Selector Dataset (B8 + tensors)

**Goal:** Build labels and the ranker dataset used for training.

```bash
python backend/scripts/05_build_selector_dataset.py --start 2016-01-01 --end 2025-12-31
```

**Notes / Caveats**
- The dataset builder currently returns mock tensors and does not yet perform real joins across features/priors/labels.【F:backend/app/selector/dataset_builder.py†L12-L52】
- Labels are currently returned as an empty schema template.【F:backend/app/selector/dataset_builder.py†L12-L28】

---

## 7) Train the Selector (Ranker)

**Goal:** Train the ranker model after features/priors/labels exist.

```bash
python backend/scripts/06_train_selector.py --train_end 2025-12-31
```

**Notes / Caveats**
- The training wrapper still delegates to a stub (`train_selector`) and will not train a real model until the trainer is fully wired in.【F:backend/scripts/06_train_selector.py†L16-L38】【F:backend/app/selector/train.py†L5-L8】

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
- The nightly script still uses empty dataframes for features and priors and a hardcoded symbol list, so it currently produces a mock leaderboard only.【F:backend/scripts/07_nightly_run.py†L42-L65】

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
