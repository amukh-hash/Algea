# Algea Pipeline Runbook (Hardened v2.0)

This runbook documents the strictly enforced data pipeline for Algea. All artifacts must pass the `phase0_preflight.py` check before training or inference can proceed.

## 1. Pipeline Modes & Environment Variables

The pipeline is gated by `PIPELINE_MODE` in `phase0_preflight.py`.

| Mode           | Checks                                                                 | Use Case                          |
|----------------|------------------------------------------------------------------------|-----------------------------------|
| `full`         | Gold, Silver, Universe, Features, Chronos, Priors, Targets            | **Production Training / Retrain** |
| `chronos_only` | Gold, Universe, Chronos Dataset                                        | Chronos Pre-training only         |
| `selector_only`| Gold, Silver, Universe, Features, Priors, Targets                      | Selector/Rank Model Training      |
| `live_ready`   | Gold, Silver, Universe, Features, Priors                              | **Live Inference** (No Targets)   |

### Critical Environment Variables

- `PIPELINE_MODE`: (Default: `full`) See above.
- `AS_OF_DATE`: (Default: Today) The reference date for validations. Checks existence of daily artifacts.
- `PRIORS_VERSION`: (Default: `latest`) The specific version hash for `priors` (e.g., `6fb4e3e851d5`).
- `UNIVERSE_VERSION`: (Default: `v2`)
- `FEATURE_VERSION`: (Default: `v2`)

## 2. Preflight Validation

Before any operation, run the preflight check. It enforces:

1. Canonical Schema (`symbol`, `date` as `pl.Date`)
2. Pathmap Resolution (`backend/data_canonical/...`)
3. Artifact Existence & Partitioning
4. Statistical Health (Breadth, Nulls)

### Command
```powershell
# Example: Full check with specific date/priors
$env:PIPELINE_MODE="full"; $env:AS_OF_DATE="2025-01-07"; $env:PRIORS_VERSION="6fb4e3e851d5"; python -m backend.scripts.teacher.phase0_preflight
```

### Success Criteria
- Output must end with: `Preflight Complete. Global OK: True`
- Exit Code: `0`
- Report saved to: `backend/data/runs/RUN-YYYY-MM-DD-XXX/reports/preflight_report.json`

## 3. Artifact Generation

If preflight fails due to missing artifacts, generate them in order:

### A. Universe Frame
```powershell
# Sourcing from Canonical Prices
python -m backend.scripts.data.build_universe_frame
```

### B. Selector Features (Strict Schema)
```powershell
# Requires Universe Frame
python -m backend.app.features.selector_features_v2 --start 2006-01-01 --end 2025-12-31
```

### C. Priors (Factor Models)
```powershell
# Requires Features
python -m backend.scripts.model.build_priors --date 2025-01-07
```

## 4. Pathmap & Schema Contracts

All code **MUST** use `backend.app.ops.pathmap` for paths and `backend.app.data.schema_contracts` for DataFrames.

- **DO NOT** hardcode paths like `backend/data/...`. Use `pathmap.get_universe_frame_root()`, etc.
- **DO NOT** use `ticker` column. Use `schema_contracts.normalize_keys(df)` to ensure `symbol` and `pl.Date`.

## 5. Troubleshooting Feature

- **Missing Priors**: Check `backend/priors/date=YYYY-MM-DD`. If file is `priors_vHASH.parquet`, set `$env:PRIORS_VERSION="HASH"`.
- **Schema Errors**: "Column 'symbol' missing". Run `normalize_keys()` on your source dataframe before saving.
- **Date Check**: Preflight defaults to *today*. If working with historical data, valid `AS_OF_DATE` is mandatory.
