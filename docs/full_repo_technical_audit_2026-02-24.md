# Final consolidated implementation plan (no further iterations)

## 1) Executive summary (max 12 bullets)

1. **P0 — Silent stub success can route orders.** Signal handlers and generic fallback can emit `status=ok` with synthetic/stub payloads; without a strict route gate, paper/live can trade on placeholder outputs. Impact: incorrect trading + false confidence. Confidence: High.

2. **P0 — Orchestrator↔paper-cycle mode contract mismatch.** Multiple mode surfaces (`--broker`, `--mode`, session handlers) lack a single typed boundary mapping; paper can be passed where only `{noop, ibkr}` is supported, triggering unintended fallbacks. Impact: wrong backend / degraded execution. Confidence: High.

3. **P0 — Broker/reconciliation schema drift.** Mixed key conventions (`qty` vs `quantity`, `ticker` vs `symbol`) and lack of canonical validation cause reconciliation failures or mis-accounting. Impact: broken post-trade controls and analytics. Confidence: High.

4. **P0 — Portfolio accounting incorrect for partial reductions/flips + short return sign.** `update_from_fill` conflates increase/reduce logic; partial reductions can mutate basis and fail to realize PnL; trade return denominator can invert for shorts. Impact: corrupted PnL/equity and misleading strategy evaluation. Confidence: High.

5. **P0 — Backtest eligibility/targets mismatch + exit liquidation bug.** Eligibility built as latest snapshot while targets merge by exact `(date,symbol)`; exit policy can skip liquidation when targets empty. Impact: suppressed historical trades and stale exposure. Confidence: High.

6. **P1 — Artifact serving traversal + symlink escape.** Artifact endpoints risk serving files outside artifacts root if `../` or symlink tricks are possible without realpath confinement. Impact: unauthorized file exposure. Confidence: High.

7. **P1 — Orchestrator timeouts not preemptive.** If handlers hang while the tick loop holds execution responsibility, the scheduler can stall without hard cancellation. Impact: deadlock/outage risk. Confidence: High.

8. **P1 — Telemetry fan-out can block progress under pressure.** If publish holds locks while blocking on subscriber queues, system can stall/lag under load. Impact: latency spikes/stalls. Confidence: High.

9. **P1 — Control plane enforcement gap.** Pause/overrides exist but are not guaranteed to be applied *pre-run* and *pre-route* under a consistent snapshot. Impact: operator controls ineffective during incidents. Confidence: High.

10. **P1 — Missing robust idempotency + lifecycle rules.** `client_order_id` optional; restart/retry can duplicate submissions without a defined lifecycle state machine and reconciliation-first semantics. Impact: duplicated orders. Confidence: High.

11. **P1/P2 — Calendar + timestamp semantics drift.** Weekday-only backtest calendar diverges from exchange sessions; intent timestamps vs fill timestamps ambiguous in logs/recon. Impact: research↔paper/live divergence and reporting confusion. Confidence: High.

12. **P2/P3 — TODO/NotImplemented runtime hazards.** Stub models/paths (e.g., LagLlama, TODO signal wiring) can leak into runtime absent startup policy gates. Impact: latent production foot-guns. Confidence: High.

---

## 2) Production readiness invariants (must become CI tests + runtime assertions)

### A. Fail-closed routing

1. In **paper/live**, routing is forbidden if any upstream `SignalArtifact` or `TargetArtifact` has `status != "ok"` **OR** `is_stub == true`.
2. In **paper/live**, any exception in signal generation produces `status="error"` and blocks downstream order build and route.

### B. Contract enforcement + versioning

3. All persisted artifacts/state **must** include `schema_version` and pass **write-time validation**.
4. All loads must enforce **read-time validation**: `load → validate(schema_version) → migrate(if allowed) → validate(post-migration) → use`.
5. Unknown `schema_version` **fails closed** in paper/live. Allowed only in noop/dev with explicit flag `ALLOW_UNKNOWN_SCHEMA=1`.
6. Every schema defines explicit allowed migrations `{from→to}`. No implicit migrations.

### C. Operational control correctness

7. **Pause dominance:** when paused, **no jobs run** and **no orders route**, regardless of triggers.
8. **Tick consistency:** exactly one immutable `ControlStateSnapshot` is captured per tick and referenced in all tick artifacts/logs.

**Control precedence (must be implemented and tested):**
9. Precedence order (highest to lowest):

* `paused == true` (hard stop)
* `ROUTING_ENABLED == false` (hard stop for routing only; signal/risk can run)
* `execution_mode_override` (e.g., force noop/paper/live semantics at routing boundary)
* `blocked_symbols` (filters targets/orders)
* `risk_caps/exposure_caps` (scales/clips orders)
* per-sleeve modes (only if not overridden)

### D. Security

10. All mutating control endpoints require authentication + scoped authorization; unauthenticated/unauthorized requests fail.
11. Artifact serving must enforce **realpath confinement** under artifacts root; reject traversal and symlink escape.

### E. Determinism + idempotency

12. No wall-clock coupling: no `date.today()`/`datetime.now()` in routing/adapters except via injected `Clock`.
13. Submission idempotency: the same `(asof_date, tick_id, strategy_scope, client_order_id)` cannot produce duplicate broker submissions.
14. Restart safety: restart reconciles open orders/fills **before** any submit; only submits intents that are not terminal and not already known to broker.

### F. Backtest/live semantic parity

15. Calendar parity: backtest/paper/live session decisions consistent for declared exchanges/timezones.
16. Eligibility date alignment: backtests produce date-aligned targets across the full window (not single-day artifacts).
17. Exit liquidation correctness: exit policies flatten positions correctly even when current targets are empty.

### G. Accounting correctness

18. Portfolio accounting conserves equity; partial reductions realize PnL correctly; short return sign correct.

---

## 3) Canonical contract inventory (schemas, owners, validation, versions, migration)

> Codex must confirm exact paths via `rg -n "<class|symbol>" backend algaie` and create modules where noted.

### 3.1 SignalArtifact — `signals.v1`

* **Owner:** `backend/app/orchestrator/schemas/signal_artifact.py` (new)
* **Fields:** `schema_version`, `artifact_type="signal"`, `status`, `is_stub`, `asof_date`, `session`, `tick_id`, `run_id`, `producer`, `payload`, `error{type,message,trace_id?}`, `control_snapshot_id`
* **Write-validate:** at end of each signal handler before persistence
* **Read-validate:** before target building and before routing gate
* **Migration:** allow `signals.v0 → signals.v1` only via explicit converter; unknown fails paper/live
* **Notes:** payload must use canonical `symbol` key (not `ticker`)

### 3.2 TargetArtifact — `targets.v1`

* **Owner:** `backend/app/orchestrator/schemas/target_artifact.py` (new)
* **Fields:** `schema_version`, `artifact_type="target"`, `status`, `is_stub`, `asof_date`, `targets[{symbol,target_weight}]`, `source_signal_run_id`, `control_snapshot_id`
* **Validation:** write-time after build; read-time at route gate
* **Migration:** explicit only

### 3.3 OrderIntent — `order_intents.v2`

* **Owner:** `algaie/trading/orders.py` (canonical)
* **Fields:** `schema_version`, `asof_date`, `tick_id`, `submitted_at` (planned submit time), `symbol`, `side`, `quantity`, `reason`, **required** deterministic `client_order_id`, optional `limit_price`, `time_in_force`
* **Validation:** pre-persist and pre-submit
* **Migration:** `v1→v2` adds required `tick_id`, `submitted_at`, deterministic `client_order_id`; missing in paper/live fails closed

### 3.4 OrderLifecycleRecord — `orders.v1`

* **Owner:** `backend/app/execution/schemas/order_lifecycle.py` (new; if existing module, extend it)
* **Fields:** `schema_version`, `client_order_id`, `broker_order_id`, `asof_date`, `tick_id`, `submitted_at`, `acknowledged_at`, `filled_at`, `canceled_at`, `status`, `reject_reason`, `last_broker_state`
* **States (must be enforced):**

  * Non-terminal: `planned`, `submitted`, `acked`, `partial_filled`
  * Terminal: `filled`, `canceled`, `rejected`, `expired`
* **Restart rule:** only submit intents lacking `broker_order_id` and `status in {planned}`; reconcile broker first to update statuses

### 3.5 FillRecord — `fills.v1`

* **Owner:** `backend/app/schemas/fill_position.py` (new)
* **Fields:** `schema_version`, `fill_id`, `symbol`, `quantity`, `price`, `side`, `commission`, `filled_at` (UTC ISO8601), `source`, `broker_order_id?`, `client_order_id?`
* **Validation:** adapter normalization output + reconciliation input
* **Normalization:** `ticker→symbol`, `qty→quantity`, timestamps→UTC ISO8601

### 3.6 PositionRecord — `positions.v1`

* **Owner:** same module as fills
* **Fields:** `schema_version`, `symbol`, `quantity`, `avg_cost`, `asof_date`, `source`
* **Validation:** broker snapshot adapters + risk/recon readers
* **Normalization:** symbol/ticker and quantity keys

### 3.7 PaperState — `paper_state.v1`

* **Owner:** `backend/app/orchestrator/paper_state_store.py` (new) or consolidate in paper broker module
* **Fields:** `schema_version`, `asof_date`, `cash`, `equity`, `positions[]`, `open_orders[]` (OrderLifecycleRecord subset), `fills[]`, `realized_pnl`, `unrealized_pnl`, `last_updated_at`
* **Read/write:** `load_paper_state()` / `save_paper_state()` only
* **Migration:** known upgrades supported; unknown versions quarantined under `artifacts/quarantine/paper_state/...`

### 3.8 ReconciliationReport — `reconcile.v1`

* **Owner:** `backend/app/execution/reconcile_futures.py` (or split schema module)
* **Fields:** `schema_version`, `asof_date`, `status`, `account_snapshot`, `positions_diff`, `fills_diff`, `violations[]`, `control_snapshot_id`
* **Validation:** output writer + API reader
* **Migration:** explicit only

### 3.9 RiskReport — `risk.v2`

* **Owner:** `backend/app/orchestrator/schemas/risk_report.py` (new canonical)
* **Fields:** `schema_version`, `status`, `checked_at`, `asof_date`, `session`, `inputs`, `metrics`, `limits`, `violations[]`, `missing_sleeves`, `control_snapshot_id`
* **Migration:** legacy→v2 converter retained; unknown fails paper/live

### 3.10 CalendarService — `calendar_service.v1`

* **Owner:** `algaie/trading/calendar_service.py` (new SSoT)
* **Methods:** `is_trading_day(date, exchange)`, `session_bounds(date, exchange)`, `trading_sessions(start,end,exchange,tz)`
* **Exchanges:** explicitly define at least `{XNYS, CME}` or repo-specific set
* **Parity tests:** required vs orchestrator session logic

### 3.11 ControlStateSnapshot — `control_state.v1`

* **Owner:** `backend/app/orchestrator/control_state.py`
* **Fields:** `schema_version`, `snapshot_id`, `paused`, `execution_mode_override`, `blocked_symbols`, `risk_caps`, `routing_enabled`, `updated_at`, `updated_by`
* **Rule:** one snapshot per tick; snapshot_id persisted into all tick artifacts

### 3.12 PortfolioSummaryResponse — `portfolio_summary.v1`

* **Owner:** `backend/app/api/schemas/portfolio_summary.py` (new)
* **Fields:** `schema_version`, `asof_date`, `cash`, `equity`, `positions`, `realized_pnl`, `unrealized_pnl`, `source_state_version`, `control_snapshot_id`
* **Source of truth:** PaperState when in paper mode; never derive from stale config when state exists

### 3.13 BacktestEligibilityPanel — `eligibility.v1`

* **Owner:** `algaie/data/eligibility/schema.py` + builder
* **Fields:** `schema_version`, `date`, `symbol`, `is_eligible`, `reason_codes[]`
* **Rule:** must be a daily panel across backtest window (not latest snapshot)

### 3.14 BacktestConfig — `backtest_config.v1`

* **Owner:** `algaie/research/config_models.py` (new canonical)
* **Mapping:** explicit conversion from `algaie/core/config.py` with strict unknown-key policy

### 3.15 CostModel — `cost_model.v1`

* **Owner:** choose `algaie/trading/costs.py` as canonical; keep adapters for legacy module(s)
* **Requirement:** deterministic, unit-documented, equivalence tests vs legacy behavior until legacy removed

---

## 4) Unified work plan (Patch 1..11, priority order)

### Patch 1 (P0): Fail-closed signal generation + route gating

* **Goal:** In paper/live, eliminate stub success and ensure routing is blocked on failed/stub signals/targets.
* **Rationale:** Prevent accidental trading on synthetic outputs.
* **Dependencies:** none.
* **Files:**

  * `backend/app/orchestrator/job_defs.py`
  * `backend/app/orchestrator/orchestrator.py`
  * `backend/tests/test_orchestrator_fail_closed.py` (new or existing test file)
* **Exact changes:**

  1. Introduce `RuntimeMode` enum `{noop,paper,live}` and `ExecutionBackend` enum `{stub,paper_broker,ibkr}` (single definition).
  2. Replace broad exception→stub fallback in signal handlers with:

     * paper/live: `SignalArtifact(status="error", is_stub=false, error=...)`
     * noop/dev: allow stub only when `ORCH_ALLOW_STUB_SIGNALS=1`
  3. Implement a single `route_preconditions_ok()` used by `handle_order_build_and_route`:

     * requires upstream artifacts validated, `status=="ok"`, `is_stub==false`
  4. Ensure downstream stages (target/risk/order build) also fail closed if upstream invalid.
* **Tests:**

  * `test_signal_exception_blocks_routing_paper_live`
  * `test_stub_artifact_blocked_in_paper_live`
  * `test_stub_allowed_only_noop_with_flag`
* **Observability:**

  * counters: `orch_fail_closed_block_total{reason=...}`
  * structured logs include `tick_id`, `run_id`, `control_snapshot_id`
* **Done when:** paper/live cannot route in any scenario where signal/target artifacts are non-ok or stubbed.

---

### Patch 2 (P0): Fix orchestrator↔paper-cycle mode contract (typed mapping) + hard-fail on mismatch

* **Goal:** Make mode/backend selection deterministic; eliminate “paper passed to noop|ibkr-only surface” failures.
* **Dependencies:** Patch 1.
* **Files:**

  * `backend/scripts/paper/run_paper_cycle_ibkr.py`
  * `backend/app/orchestrator/job_defs.py`
  * `backend/scripts/orchestrate.py`
  * `backend/tests/test_mode_mapping.py`
* **Exact changes:**

  1. Create `map_runtime_to_exec_backend(runtime_mode, broker_flag, account_type, sleeve)` returning an `ExecutionBackend` or error.
  2. Update `phase_open` signature to accept the exec backend contract (or wrap it).
  3. Remove any fallback-to-stub on invalid mapping in paper/live; emit explicit error artifact.
  4. Add config bit `CORE_INTEGRATION_REQUIRED` (or per-sleeve required list) that fails tick if missing.
* **Tests:**

  * `test_paper_mode_maps_to_ibkr_paper_backend`
  * `test_invalid_mode_mapping_fails_closed`
* **Done when:** paper mode always selects intended backend, and any mismatch is surfaced as a hard error.

---

### Patch 3 (P0): Canonical Fill/Position schema + adapter normalization + reconciliation robustness

* **Goal:** Broker-agnostic reconciliation based on canonical validated records; eliminate key drift.
* **Dependencies:** Patch 1.
* **Files:**

  * `backend/app/schemas/fill_position.py` (new)
  * `backend/app/orchestrator/broker_ibkr_adapter.py`
  * `backend/app/orchestrator/broker_paper.py`
  * `backend/app/execution/reconcile_futures.py`
  * `backend/tests/test_reconcile_contracts.py`
* **Exact changes:**

  1. Define `FillRecord` and `PositionRecord` with strict validation.
  2. Implement `normalize_fills(source, raw)` and `normalize_positions(source, raw)` in one shared location.
  3. Enforce:

     * adapter egress: must output canonical `fills.v1` / `positions.v1`
     * reconciliation ingress: only accepts canonical; unknown schema fails closed
  4. Add `schema_version` to persisted fills/positions artifacts and validate on load.
* **Tests:**

  * `test_normalize_ibkr_fills_to_canonical`
  * `test_normalize_paper_fills_to_canonical`
  * `test_reconcile_rejects_unknown_schema_version`
* **Done when:** reconciliation never references broker-native keys directly.

---

### Patch 4 (P0): Portfolio accounting correctness + short return sign + timestamps semantics

* **Goal:** Fix accounting invariants for partial reductions/flips; correct short return sign; clarify timestamps.
* **Dependencies:** Patch 3.
* **Files:**

  * `algaie/trading/portfolio.py`
  * `algaie/trading/orders.py`
  * `algaie/trading/fills.py`
  * `backend/tests/test_portfolio_accounting.py`
* **Exact changes:**

  1. Refactor fill application into branches:

     * `increase_position`
     * `reduce_position` (realize PnL on reduced qty; preserve remaining basis)
     * `close_position`
     * `flip_position` (close old side then open new with new basis)
  2. Fix trade return: denominator uses `abs(entry_notional)` not signed notional.
  3. Introduce `submitted_at` and `filled_at` consistently:

     * intents carry `submitted_at` (planned)
     * fills carry `filled_at` (executed)
     * analytics/recon uses `filled_at` for execution attribution
* **Tests:**

  * `test_partial_reduce_long_realized_pnl`
  * `test_partial_reduce_short_realized_pnl_and_return_sign`
  * `test_flip_position_conservation`
  * `test_equity_conservation_multi_fill_sequence`
* **Done when:** all conservation and sign tests pass.

---

### Patch 5 (P0/P1): Eligibility↔targets alignment + backtest causality guardrails + exit liquidation fix

* **Goal:** Make eligibility a daily panel; align target construction; guarantee liquidation on empty targets.
* **Dependencies:** Patch 4.
* **Files:**

  * `algaie/data/eligibility/schema.py` (new if absent)
  * `algaie/data/eligibility/build.py`
  * `algaie/trading/risk.py` (or target builder)
  * `algaie/research/backtest_engine.py`
  * `backend/scripts/research/run_backtest.py`
  * `backend/tests/test_backtest_eligibility_targets.py`
* **Exact changes:**

  1. Replace eligibility “latest snapshot” with daily panel `eligibility.v1` over the backtest window.
  2. Update target builder to merge eligibility by `(date,symbol)` consistently.
  3. Fix `_apply_exit_policy`:

     * if policy requires flatten and targets empty, generate explicit liquidation intents/targets for held symbols.
  4. Add causality guardrails:

     * For each date D, eligibility/inputs used must be `<= D` (no future rows).
* **Tests:**

  * `test_multiday_targets_exist_on_many_dates` (assert >N distinct dates)
  * `test_no_future_eligibility_rows_used`
  * `test_empty_targets_triggers_liquidation`
* **Done when:** multi-day backtests generate targets across the full window and liquidate correctly on empty-target days.

---

### Patch 6 (P1): Empty-order success path + strict quotes/pricing hard-fail split

* **Goal:** Make “no trade” a success; ensure quote absence fails in strict mode.
* **Dependencies:** Patches 1 and 5.
* **Files:** `backend/app/orchestrator/job_defs.py`, quote utilities (repo-confirm), tests.
* **Exact changes:**

  1. If `orders == []`, emit `status=ok`, `reason=no_rebalance_needed`, and skip routing.
  2. Implement `STRICT_QUOTES`:

     * paper/live default `1`
     * missing quote triggers `status=error` with explicit symbol list
* **Tests:**

  * `test_zero_orders_is_success`
  * `test_strict_quotes_missing_quote_fails_closed`
* **Done when:** no-trade days do not fail ticks; quote gaps fail loudly.

---

### Patch 7 (P1): Real timeouts/cancellation + telemetry non-blocking fan-out

* **Goal:** Prevent hung handlers and telemetry backpressure from stalling the orchestrator.
* **Dependencies:** Patch 1.
* **Files:** `backend/app/orchestrator/runner.py`, `backend/app/telemetry/storage.py`, tests.
* **Exact changes:**

  1. Execute jobs in a cancellable boundary (subprocess or thread w/ watchdog + hard timeout).
  2. Classify timeout distinctly (`status=error`, code=`timeout`).
  3. Telemetry publish:

     * do not block while holding global lock
     * use `put_nowait`; track drops/backpressure counters
* **Tests:**

  * `test_job_timeout_enforced`
  * `test_telemetry_backpressure_does_not_block_tick`
* **Done when:** hung jobs do not stall past timeout and telemetry cannot deadlock tick progress.

---

### Patch 8 (P1): Control plane effectiveness + authn/authz + artifact traversal + symlink confinement

* **Goal:** Ensure operator controls are effective and APIs are safe.
* **Dependencies:** Patch 1.
* **Files:** `backend/app/orchestrator/control_state.py`, `backend/app/orchestrator/orchestrator.py`, `backend/app/api/main.py`, `backend/app/api/control_routes.py`, artifact routes, tests.
* **Exact changes:**

  1. Capture `ControlStateSnapshot` once per tick; attach `snapshot_id` to all artifacts.
  2. Enforce precedence rules (pause dominance, routing_enabled, etc.) at:

     * tick start (job gating)
     * pre-route (routing gating)
  3. Add auth middleware + endpoint scopes; default deny for mutating endpoints.
  4. Artifact serving hardening:

     * compute candidate `realpath`
     * reject if not under artifacts root realpath
     * optionally disallow symlinks within artifacts root (or validate each path component)
* **Tests:**

  * `test_pause_blocks_all_jobs_and_routing`
  * `test_routing_enabled_false_still_allows_signal_generation`
  * `test_mutating_endpoints_require_auth`
  * `test_artifact_traversal_rejected`
  * `test_artifact_symlink_escape_rejected`
* **Done when:** operator controls apply within the same tick and artifact endpoints cannot escape root.

---

### Patch 9 (P1/P2): Broker mode semantics + deterministic idempotency + restart-safe lifecycle + remove wall-clock

* **Goal:** Eliminate duplicate submissions and nondeterminism.
* **Dependencies:** Patches 2–4.
* **Files:** `backend/scripts/orchestrate.py`, broker adapters, `algaie/trading/orders.py`, cycle scripts, tests.
* **Exact changes:**

  1. Centralize broker mode mapping and document it; remove divergent interpretations (`stub` vs persistent paper).
  2. Deterministic `client_order_id` algorithm (inputs include: `asof_date`, `tick_id`, `strategy_scope`, `symbol`, `side`, `quantity`, maybe `limit_price`).
  3. Persist intents and lifecycle records before submit; submit only if lifecycle state allows.
  4. Restart workflow:

     * load intents + lifecycle
     * reconcile broker open orders/fills
     * submit only remaining `planned` intents
  5. Remove all `date.today()/datetime.now()` from adapters; require injected `Clock`.
* **Tests:**

  * `test_replay_same_tick_no_duplicate_submit`
  * `test_restart_reconciles_before_submit`
  * `test_wallclock_calls_banned_in_adapters` (static grep gate + runtime assert hook)
* **Done when:** repeated runs do not create duplicate broker submissions and outputs are as-of deterministic.

---

### Patch 10 (P2): Calendar SSoT + cost model consolidation + config dedup + portfolio summary correctness

* **Goal:** Remove semantic drift and fix monitoring correctness.
* **Dependencies:** Patches 5 and 9.
* **Files:** calendar modules, cost modules, config models, portfolio summary API, tests.
* **Exact changes:**

  1. Introduce `CalendarService` and migrate backtest + orchestrator to use it.
  2. Add golden tests for holidays/early closes/DST for declared exchanges/timezones.
  3. Consolidate cost model to `cost_model.v1` and keep compatibility adapters until deletion.
  4. Deduplicate `BacktestConfig` into canonical `backtest_config.v1` with explicit mapping.
  5. Fix portfolio summary:

     * load from canonical PaperState path
     * realized/unrealized from state or computed from canonical fills/positions, not nonexistent fields
* **Tests:**

  * `test_calendar_golden_suite`
  * `test_cost_model_equivalence_matrix`
  * `test_backtest_config_mapping`
  * `test_portfolio_summary_uses_paper_state`
* **Done when:** research/paper/live session decisions match for same config and UI summary matches paper state.

---

### Patch 11 (P2/P3): TODO/stub hazards removal + CI lint gates + startup policy validation

* **Goal:** Prevent stub/TODO paths from entering paper/live.
* **Dependencies:** Patches 1–10.
* **Files:** stub modules (LagLlama etc.), CI config/scripts, startup validators.
* **Exact changes:**

  1. Add runtime-path lint gate: deny `TODO|FIXME|XXX|NotImplementedError` in runtime modules except allowlist.
  2. Startup validation:

     * fail in paper/live if any stub model/handler enabled
     * enforce flags defaults: `ORCH_ALLOW_STUB_SIGNALS=0`, `STRICT_QUOTES=1`, `AUTH_REQUIRED=1` (outside localhost)
  3. Ensure stub-only modules are dev-guarded.
* **Tests:**

  * `test_startup_rejects_stubs_in_paper_live`
  * `test_lint_gate_blocks_runtime_todos`
* **Done when:** paper/live startup fails if unsafe stubs remain.

---

## 5) CI / release gates (exact checks; required for paper/live merge)

### Required test commands (adapt filenames to repo after `rg`)

* `pytest -q backend/tests/test_orchestrator_fail_closed.py`
* `pytest -q backend/tests/test_mode_mapping.py`
* `pytest -q backend/tests/test_reconcile_contracts.py`
* `pytest -q backend/tests/test_portfolio_accounting.py`
* `pytest -q backend/tests/test_backtest_eligibility_targets.py`
* `pytest -q backend/tests/test_control_plane_security.py`
* `pytest -q backend/tests/test_artifact_security.py`
* `pytest -q backend/tests/test_timeouts_and_telemetry.py`
* `pytest -q backend/tests/test_idempotency_restart.py`
* `pytest -q backend/tests/test_calendar_parity.py`
* `pytest -q backend/tests/test_cost_model_equivalence.py`

### Static/security/determinism gates

* Deny runtime TODO/stubs:

  * `rg -n "TODO|FIXME|XXX|NotImplementedError" backend algaie | rg -v "allowlist|tests|docs"`
* Deny wall-clock in adapters/routing:

  * `rg -n "date\.today\(|datetime\.now\(" backend/app/orchestrator backend/scripts algaie/trading | rg -v "clock|tests"`
* Lint/type if present:

  * `ruff check backend algaie`
  * `python -m mypy backend/app/orchestrator backend/app/api algaie/trading` (only if enabled)

### Merge policy

* Paper/live branch requires all gates green.
* Any schema bump requires:

  * migration function
  * migration test
  * release note entry
* Any routing/control-state change requires updating integration tests.

---

## 6) Cutover and rollout notes (noop → paper → live; feature flags; break-glass)

### 6.1 Feature flags and defaults

* `ORCH_ALLOW_STUB_SIGNALS=0` default; must remain 0 in paper/live.
* `STRICT_QUOTES=1` default in paper/live.
* `AUTH_REQUIRED=1` default outside localhost.
* `CALENDAR_SERVICE_ENFORCED=0` during migration; set to 1 only after parity tests pass.
* `SCHEMA_STRICT_READ={warn|fail}` progression: warn in early paper dry-run → fail in paper production → fail in live.
* `ROUTING_ENABLED=0/1` independent kill-switch (default 0 for initial paper rollout).

### 6.2 Safe deployment sequence (paper-first)

1. Deploy Patch 1–3 with `ROUTING_ENABLED=0`, `SCHEMA_STRICT_READ=warn`. Confirm fail-closed artifacts and schema validation metrics.
2. Deploy Patch 4–6 in paper; keep strict quotes on; verify no-trade success behavior and missing quote failures.
3. Deploy Patch 7–9; run replay/restart drills and timeout/backpressure stress tests.
4. Deploy Patch 10; enable `CALENDAR_SERVICE_ENFORCED=1` only after parity goldens pass.
5. Deploy Patch 11; turn `SCHEMA_STRICT_READ=fail`; enforce lint gates.

**Go/no-go:** 2 consecutive paper sessions with:

* zero routed orders from stub/failed signals,
* zero unknown schema versions loaded,
* zero duplicate submissions,
* pause override verified in same tick.

### 6.3 Break-glass procedures

* Force safe noop: set runtime mode noop + `ROUTING_ENABLED=0`.
* Disable routing but keep visibility: `ROUTING_ENABLED=0`, keep telemetry/artifacts on.
* Emergency auth lockdown: set `AUTH_REQUIRED=1`, rotate keys, restrict scopes.

### 6.4 Artifact/state migration strategy

* For artifacts without `schema_version`:

  * migrate offline with deterministic `migrate_*()` tools where possible
  * otherwise quarantine under `artifacts/quarantine/<date>/...`
* Never load quarantined/unknown in paper/live after strict-read=fail.
* Maintain a migration manifest per contract: allowed paths, checksums/row counts post-migration.
