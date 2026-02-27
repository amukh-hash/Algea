# Algea Codebase Migration Audit

**Date:** February 16, 2026
**Context:** Folder migration to new Windows user (`crick`) after PC upgrade (dual GPU: 4070 Super + 3090 Ti)

---

## Executive Summary

The codebase structure and Python module connectivity are **healthy** — no broken imports or missing core modules. The audit found issues related to hardcoded Windows paths from the old `Aishik` user profile, stale virtual environments, and missing GPU routing for the dual-GPU setup.

**All actionable items have been resolved.** Remaining manual step: run `cleanup_migration.bat` on your Windows machine to remove `.venv` and recreate it.

---

## Changes Applied

### 1. Stale Batch Files Moved to Legacy  ✅

`run_gpu.bat`, `run_training.bat`, and `finish_phase3.bat` referenced **9 scripts that no longer exist** in `backend/scripts/` — they were from the old ensemble-based architecture (v2). All three files moved to `legacy/stale_bat/`.

### 2. Hardcoded Path in Active Script Fixed  ✅

**File:** `backend/scripts/_audit_portfolio.py`

**Before:**
```python
RUN = r"C:\Users\Aishik\Documents\Workshop\Algea\backend\data\selector\runs\SEL-20260211-141209"
```

**After:**
```python
from pathlib import Path
RUN = str(Path(__file__).resolve().parents[1] / "data" / "selector" / "runs" / "SEL-20260211-141209")
```

### 3. All GPU Scripts Now Route to 3090 Ti (cuda:1)  ✅

Created a centralised device selection module: **`algea/core/device.py`**

Resolution order:
1. Explicit `override` argument
2. `ALGAIE_CUDA_DEVICE` env var (added to `.env` as `cuda:1`)
3. `cuda:1` default (3090 Ti)
4. `cpu` fallback

Includes automatic validation — if `cuda:1` doesn't exist at runtime, falls back to `cuda:0` with a warning.

**13 files updated** to use `get_device()`:

| File | Change |
|---|---|
| `algea/models/tsfm/lag_llama/config.py` | Default `"cuda"` → `"cuda:1"` |
| `algea/training/foundation_train.py` | `torch.device(...)` → `get_device()` |
| `algea/training/ranker_train.py` | `torch.device(...)` → `get_device()` |
| `algea/training/selector_train.py` | `torch.device(...)` → `get_device()` |
| `backend/scripts/build_priors_cache.py` | `torch.device(...)` → `get_device()` |
| `backend/scripts/eval_chronos_metrics.py` | `torch.device(...)` → `get_device()` |
| `backend/scripts/train_chronos2_teacher.py` | `setup_device()` → uses `get_device()` + GradScaler fix |
| `backend/scripts/train_selector.py` | `torch.device(...)` → `get_device()` |
| `backend/scripts/verify_chronos_priors.py` | `torch.device(...)` → `get_device()` |
| `backend/scripts/verify_inference_stability.py` | `torch.device(...)` → `get_device()` |
| `backend/scripts/walkforward_eval.py` | `torch.device(...)` → `get_device()` |
| `backend/tests/test_mlp_selector.py` | `.cuda()` → `.to(get_device())` |
| `backend/tests/test_pairwise_loss.py` | `device="cuda"` → `device=str(get_device())` |

**To override at runtime** (e.g. route to 4070 Super instead):
```bash
set ALGAIE_CUDA_DEVICE=cuda:0
```

### 4. GradScaler Device Fix  ✅

**File:** `backend/scripts/train_chronos2_teacher.py`

**Before:** `torch.amp.GradScaler("cuda", enabled=use_amp)`
**After:** `torch.amp.GradScaler(device.type, enabled=use_amp)`

### 5. Missing `__init__.py` Files Added  ✅

Created empty `__init__.py` in:
- `algea/core/artifacts/`
- `backend/app/data/`
- `backend/app/portfolio/`

### 6. `algea/models/__init__.py` Fixed  ✅

**Before:** `__all__ = ["foundation", "ranker", "common"]`
**After:** `__all__ = ["foundation", "ranker", "common", "tsfm"]`

### 7. `algea/core/__init__.py` Updated  ✅

Added `"device"` to `__all__` for the new device selection module.

### 8. `.env` Updated with GPU Config  ✅

Added:
```
ALGAIE_CUDA_DEVICE=cuda:1
```

---

## Manual Steps Required

### Run `cleanup_migration.bat` on Windows

This script (created in the project root) will:
1. Delete the broken `.venv/` (created under old `Aishik` profile with Python 3.14)
2. Remove stale test output logs (`pack_test_out.txt`, `phase2_tests.txt`, etc.)
3. Recreate `.venv` with your current Python and install dependencies

**Note:** `venv_gpu/` is already correct for the `crick` user — no action needed there.

### Verify API Keys

The `.env` file has Alpaca and FRED API keys. Verify they still work from your new account — these are account-level, not machine-level, so they should be fine.

---

## Remaining Low-Priority Items (Optional)

### Run Manifest JSON Files (85+) Contain Old `Aishik` Paths

Files in `backend/data/runs/RUN-2026-02-07-*` and `RUN-2026-02-08-*` have hardcoded paths. These are historical records and don't affect runtime (code uses `algea.core.paths` for dynamic resolution). If you want to clean them up:

```python
# One-time migration script
import json, pathlib
old = r"C:\Users\Aishik\Documents\Workshop\Algea"
new = r"C:\Users\crick\Documents\Workshop\Algea"
for f in pathlib.Path("backend/data/runs").rglob("*.json"):
    text = f.read_text()
    if old in text:
        f.write_text(text.replace(old, new))
```

### Checkpoint Manifest `WindowsPath()` Strings

Some `manifest.json` files contain `WindowsPath('...')` instead of plain strings. Same bulk replacement applies. Consider also fixing the serialization code in the teacher training script to use `str(path)` instead of `repr(path)`.

### `scaler.joblib` at Project Root

A 605-byte leftover file with no code referencing it. Safe to delete.

### `.claude/settings.local.json`

Contains old `Aishik` paths in Bash command patterns. Regenerate for your new Claude Code configuration.

---

## What's Working Well

- **`algea/core/paths.py`** — Dynamic `Path` resolution with relative segments. Survived migration cleanly.
- **`algea/core/config.py`** — All config loading uses `Path` objects, no hardcoded strings.
- **All Python imports** — 137 algea files and 216 backend files verified. Zero broken imports, zero circular dependencies, clean `algea ← backend` separation.
- **Data artifacts** — All 5,571 canonical ticker files, 998 prior cache files, model checkpoints, universe manifests present and intact.
- **`venv_gpu/`** — Correctly configured for the `crick` user.
- **PROD_POINTER.json** — Production model pointer intact and properly versioned.
- **pyproject.toml and Makefile** — Use relative paths, fully portable.
