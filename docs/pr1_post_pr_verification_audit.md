# PR-1 Post-PR Verification Audit (Default Runtime Path)

Scope audited: `backend/scripts/orchestrate.py -> Orchestrator.run_once -> default_jobs`.

## Executive verdict

**Mostly resolved with follow-ups**.

The PR-1 fixes for position aliasing, shadow-intent canonical fields, and observability are present and mostly correct on the default path. However, the mode-normalization change introduces a critical default-path mismatch: `run_once` now normalizes `live -> ibkr` before filtering, while `default_jobs()` still allows `"live"` only, so `filtered_jobs(..., mode="ibkr")` returns no jobs.

## Key findings

1. **Critical follow-up required:** mode normalization is applied at the boundary but not harmonized with `default_jobs` mode allowlists.
2. **Position field normalization is correctly implemented** (`quantity` preferred, `qty` fallback with hit counting).
3. **Shadow intent fix is correctly implemented** (canonical `TargetIntent` fields used in shadow ledger).
4. **Observability additions exist and appear telemetry-only**, but one metric (`mode_alias_applied`) is now embedded in `orders.json` summary and should be treated as non-breaking additive schema change.
5. **Tests added in PR-1 exist and pass**, but they miss the actual default-path regression (`ibkr` filtered-job emptiness).

## Evidence commands run

- `python - <<'PY' ... filtered_jobs(default_jobs(), mode='ibkr') ... PY` -> count `0`.
- `python - <<'PY' ... for m in ['live','paper','ibkr'] ... PY` -> `live=3`, `paper=3`, `ibkr=0` at session OPEN.
- `PYTHONPATH=. pytest -q ...` -> `5 passed` for added PR-1 tests.

