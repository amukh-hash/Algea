# Full-Repo Technical Audit (2026-02-24)

## 1) Executive summary

- **P0 – Portfolio accounting is mathematically wrong on partial position reductions.** Selling part of a long (or covering part of a short) incorrectly recomputes `avg_cost` and fails to realize PnL, contaminating equity, risk, and all backtest metrics. **Impact:** silent wrong backtests and potentially wrong live risk posture. **Confidence: High**.
- **P0 – Eligibility semantics are inconsistent with backtest/live consumers.** Eligibility is built as a one-row-per-ticker as-of snapshot, but target construction joins on exact `date,ticker`; most historical days become ineligible by construction. **Impact:** sparse/no trades or day-skewed behavior in backtests and paper/live cycles. **Confidence: High**.
- **P0 – Exit policy fails to liquidate when today has zero targets.** `_apply_exit_policy` returns empty targets unchanged, so holdings remain when signal-driven exits should flatten. **Impact:** stale risk retained, strategy behavior diverges from spec. **Confidence: High**.
- **P1 – Trading calendar for research/backtest ignores exchange holidays/early closes.** `pd.bdate_range` is used instead of exchange-aware sessions while orchestrator uses exchange calendars, creating environment-specific semantic drift. **Impact:** off-by-session fills/performance distortion and backtest/live mismatch. **Confidence: High**.
- **P1 – Order idempotency is missing in paper/live path.** Intents default to `client_order_id=None` and cycle scripts do not inject deterministic IDs; retries can duplicate orders. **Impact:** duplicate fills and unintended exposure. **Confidence: High**.
- **P1 – Orchestrator can silently emit stub signals in production fallback.** The core handler falls back to synthetic placeholder signals containing a TODO marker. **Impact:** accidental trading on fake signals during integration failures. **Confidence: Medium**.
- **P2 – Duplicate `BacktestConfig` definitions create split source-of-truth risk.** Config parsing and runtime engine models diverge structurally and must be manually synchronized. **Impact:** maintainability and configuration drift bugs. **Confidence: High**.
- **P2 – Fill/order timestamp semantics are inconsistent in simulation.** `next_open` fills execute on `next_date` while order log date remains intent `asof`; this can misstate T+1 execution in analytics/reconciliation. **Impact:** reporting confusion and reconciliation friction. **Confidence: High**.

## 2) Findings table

| ID | Severity | Category | Location | Symptom | Root cause | Concrete fix | Verification |
|---|---|---|---|---|---|---|---|
| F-001 | P0 | Correctness | `algaie/trading/portfolio.py` (`update_from_fill`, ~L75-109) | Partial sells/buys against an existing position mutate cost basis incorrectly and do not realize PnL; trade log misses partial exits. | Single weighted-average formula is applied to both add and reduce flows; realized PnL logic only triggers on full close/flip. | Split execution paths into: add-to-position, reduce-position, close, and flip. Keep avg cost unchanged on reductions; realize PnL on reduced quantity; append partial-close trade entries. | Add unit tests: long reduce, short reduce, flip, and partial-close PnL conservation. Run `pytest backend/tests/test_portfolio.py -q`. |
| F-002 | P0 | Integration | `algaie/data/eligibility/build.py` (~L13-32), `algaie/trading/risk.py` (`build_targets`, ~L39-43), `backend/scripts/research/run_backtest.py` (~L54) | Most backtest dates produce empty targets unless they match the eligibility snapshot dates exactly. | Eligibility builder emits only latest row per ticker up to as-of; target builder requires exact `date,ticker` match. | Define eligibility contract explicitly: either daily panel (`date,ticker,is_eligible`) or as-of snapshot keyed by ticker only. Then update `build_targets` merge keys accordingly. | Add integration test that backtest over multi-day sample generates non-empty targets on each session when eligible. Run `pytest backend/tests/test_eligibility_asof.py backend/tests/test_portfolio_targets.py -q`. |
| F-003 | P0 | Correctness | `algaie/research/backtest_engine.py` (`_apply_exit_policy`, ~L150-165) | Signal exit policy does not flatten positions when no targets are produced that day. | Guard `if not exit_tickers or targets.empty: return targets` returns early when `targets` is empty, skipping forced exits. | When `exit_tickers` is non-empty and `targets` empty, return explicit zero-weight rows for `exit_tickers` (or directly generate liquidation intents). | Add regression test: portfolio starts with open position, empty targets, `exit_policy='signal'` should produce sell intent and flat position after run. |
| F-004 | P1 | Correctness | `algaie/trading/calendar.py` (`trading_days`, ~L9-10) | Backtest calendar includes non-trading weekdays for many exchanges and ignores early-close semantics. | Uses `pd.bdate_range` (weekday calendar) instead of exchange session calendar used by orchestrator. | Replace with exchange/session calendar utility (single shared calendar module) and pass exchange as config. | Add golden test for known NYSE holiday and CME holiday week to ensure excluded sessions. |
| F-005 | P1 | Ops | `backend/scripts/paper/run_paper_cycle.py` (~L78-98), `backend/scripts/live/run_live_cycle.py` (~L91-99), `algaie/trading/orders.py` (`client_order_id`, ~L16) | Retries/restarts can submit duplicate orders because intents lack deterministic IDs; dedupe impossible broker-side. | `OrderIntent.client_order_id` defaults to `None`; cycle scripts never populate deterministic idempotency keys. | Generate deterministic `client_order_id` from `(run_id, asof, ticker, side, target_qty_hash)` and persist before submission; on retry, reuse IDs and reconcile existing orders first. | Add test that rerunning same cycle with same run context produces identical IDs and no extra submissions in a mocked broker. |
| F-006 | P1 | Integration | `backend/app/orchestrator/job_defs.py` (`_generic_signal_handler` + fallback path, ~L172-204) | Production failures in strategy import/execution can fall back to stub synthetic signals and still return `status=ok`. | Exception handler catches broad errors and returns `_generic_signal_handler` output marked `stub=True` but successful. | Gate fallback behind explicit `ALLOW_STUB_SIGNALS=1` dev flag; in prod mode raise hard failure. Emit explicit alert metric on fallback. | Add orchestrator test ensuring in `mode=live/paper` fallback raises and job status is failed. |
| F-007 | P2 | Maintainability | `algaie/core/config.py` (`BacktestConfig`, ~L24-42) vs `algaie/research/backtest_engine.py` (`BacktestConfig`, ~L22-30) | Two independent `BacktestConfig` dataclasses can drift in fields/defaults and semantics. | Duplicated concept with different responsibilities but same name and overlapping fields. | Create a single canonical runtime config model (e.g., `algaie/research/config_models.py`) and map file config into it explicitly. | Add schema contract test asserting serialized config used in artifacts round-trips into engine config. |
| F-008 | P2 | Integration | `algaie/trading/fills.py` (`simulate`, ~L31-61), `algaie/research/backtest_engine.py` (orders log date, ~L103-106) | In `next_open`, fill executes on `next_date` but order log stamps original `intent.asof`, creating ambiguous execution timelines. | Order object stores `asof=intent.asof` regardless of fill date; downstream logs rely on `order.asof`. | Extend `Order` with `submitted_at` and `filled_at`, write both fields, and standardize consumers on `filled_at` for execution analytics. | Add test validating `next_open` has `submitted_at=t` and `filled_at=t+1` and reconciliation uses `filled_at`. |
| F-009 | P3 | Security | `backend/app/api/main.py` (~L22-27) | CORS policy hard-codes localhost origins; risk is not direct exploit but prod deployment fragility and ad-hoc overrides. | Static origin list in code, no environment-based configuration. | Move CORS allowlist to env/config and enforce explicit production values at startup. | Add startup test for invalid/missing prod CORS config. |
| F-010 | P2 | Correctness / TODO risk | `algaie/models/foundation/lagllama.py` (~L22-30), `backend/app/orchestrator/job_defs.py` (~L181) | TODO/stub implementations can be wired accidentally and fail at runtime or produce fake outputs. | Stub classes/functions are registered and fallback code paths can emit TODO-marked artifacts. | Mark stubs as experimental and block from production registration; fail-fast if selected in prod config. | Add config validation test that rejects stub model/handlers for `mode in {paper, live}`. |

## 3) Patch plan

### Patch 1 (highest priority): Correct portfolio fill accounting
- **Goal:** Make position/PnL accounting financially correct and deterministic.
- **Files:** `algaie/trading/portfolio.py`, `backend/tests/test_portfolio.py`, `backend/tests/test_portfolio_backtest.py`.
- **Exact changes:**
  - Refactor `update_from_fill` into explicit branches (increase, reduce, close, flip).
  - Realize PnL on reductions; keep avg cost invariant on reductions.
  - Emit partial trade-log entries with reduced quantity and reason (`partial_exit`, `flip_exit`).
- **Tests to add:** long/short partial exits, flips, and equity conservation against known ledger examples.
- **Done when:** all new portfolio accounting tests pass and no legacy test regressions.

### Patch 2: Unify eligibility semantics across flows
- **Goal:** Ensure eligibility contract matches target builder usage for backtest/paper/live.
- **Files:** `algaie/data/eligibility/build.py`, `algaie/trading/risk.py`, `backend/scripts/research/run_backtest.py`, `backend/scripts/paper/run_paper_cycle.py`, `backend/scripts/live/run_live_cycle.py`, relevant tests.
- **Exact changes:**
  - Choose one contract: daily eligibility panel (preferred).
  - Build daily `is_eligible` per ticker/date from canonical bars.
  - Keep merge on `date,ticker` and document schema.
- **Tests to add:** multi-day integration test proving non-empty targets on eligible days and consistent counts across backtest/paper/live.
- **Done when:** backtest and cycle scripts produce aligned target universes for identical inputs.

### Patch 3: Fix exit liquidation behavior
- **Goal:** Enforce exits even when signal set is empty.
- **Files:** `algaie/research/backtest_engine.py`, tests in `backend/tests/test_portfolio_backtest.py` (or new dedicated file).
- **Exact changes:**
  - `_apply_exit_policy` returns explicit zero-weight targets (or separate liquidation ticker list) when `exit_tickers` non-empty and `targets` empty.
  - Keep behavior deterministic and documented.
- **Tests to add:** empty-target liquidation for `signal`, timed liquidation for `time`, mixed behavior for `hybrid`.
- **Done when:** open positions are flattened per policy in all edge scenarios.

### Patch 4: Single source of truth for calendar/session logic
- **Goal:** Remove backtest/live calendar drift.
- **Files:** `algaie/trading/calendar.py`, `algaie/research/backtest_engine.py`, config docs/tests.
- **Exact changes:**
  - Replace `pd.bdate_range` with exchange-aware calendar (reuse orchestrator calendar utilities or dedicated shared adapter).
  - Add exchange and timezone to backtest config.
- **Tests to add:** known-holiday and DST boundary tests.
- **Done when:** backtest sessions match orchestrator sessions for the same exchange and date range.

### Patch 5: Add order idempotency + safe retry semantics
- **Goal:** Prevent duplicate submissions during retry/restart.
- **Files:** `algaie/trading/orders.py`, `backend/scripts/paper/run_paper_cycle.py`, `backend/scripts/live/run_live_cycle.py`, broker adapters/tests.
- **Exact changes:**
  - Require deterministic `client_order_id` generation.
  - Persist outbound intents before submission; reconcile existing open orders before re-submit.
  - Add bounded retry/backoff with clear failure states.
- **Tests to add:** replay/retry test with mocked broker confirming no duplicate orders.
- **Done when:** repeated identical run contexts are idempotent.

### Patch 6: Harden orchestrator fallback behavior
- **Goal:** Eliminate silent “stub success” in non-dev modes.
- **Files:** `backend/app/orchestrator/job_defs.py`, orchestrator config/tests.
- **Exact changes:**
  - Make fallback conditional on explicit dev flag.
  - Emit `status=failed` on strategy load/exec exceptions in paper/live.
  - Add structured alert payload.
- **Tests to add:** production-mode fallback failure test and dev-mode explicit stub test.
- **Done when:** stub outputs cannot drive production routing silently.

### Patch 7: Consolidate config models and schema versioning
- **Goal:** Improve maintainability and contract clarity.
- **Files:** `algaie/core/config.py`, `algaie/research/backtest_engine.py`, artifact schema docs/tests.
- **Exact changes:**
  - Remove duplicated `BacktestConfig` naming collision.
  - Introduce explicit artifact schema/version fields in emitted JSON/parquet metadata.
- **Tests to add:** config/model compatibility test and artifact version presence test.
- **Done when:** one canonical model controls runtime semantics; artifacts are versioned and machine-verifiable.

### Patch 8: TODO/stub production guardrail sweep
- **Goal:** Convert TODO/FIXME risk into explicit feature gating.
- **Files:** `algaie/models/foundation/lagllama.py`, `backend/app/orchestrator/job_defs.py`, docs/config validators.
- **Exact changes:**
  - Block stub model registration for production.
  - Add startup validation that rejects TODO-marked stubs unless explicitly enabled.
- **Tests to add:** config validation tests for mode gating.
- **Done when:** prod runs cannot execute stub/TODO paths.

## Single source-of-truth inventory (current state)
- **Time/calendar/session logic:** split between `algaie/trading/calendar.py` (weekday-only) and orchestrator exchange-aware calendar (`backend/app/orchestrator/calendar.py`) → **competing implementations**.
- **Weight semantics:** target weights centered in `algaie/trading/risk.py` and translated to quantities in `algaie/trading/portfolio.py`; semantics are mostly consistent but eligibility keying breaks end-to-end behavior.
- **Cost model I/O:** centralized in `algaie/trading/costs.py`, applied in simulator/backtest engine.
- **Risk report schema:** orchestrator handles both legacy and canonical shapes in `backend/app/orchestrator/job_defs.py` (`_canonical_or_legacy_risk`) indicating schema duality.
- **Artifact/schema versioning:** artifact registry stores ad-hoc `version` strings in `algaie/core/artifacts/registry.py`; no enforced global schema version contract.

## Assumptions
- Assumed broker adapters may be used for both paper and live retries, so idempotency is mandatory.
- Assumed “signal exit” policy should flatten absent-target holdings (common systematic strategy interpretation).
- Assumed strategy correctness in backtest must match session semantics used by orchestrator for paper/live.
