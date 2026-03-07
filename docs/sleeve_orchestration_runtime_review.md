# Default Runtime Migration Program Spec (Execution + Hardening)

Scope authority (proven): `backend/scripts/orchestrate.py -> Orchestrator.run_once -> default_jobs`.

Proof anchors:
- Default loop and entrypoint: `backend/scripts/orchestrate.py:58-80`
- Active job registration: `backend/app/orchestrator/orchestrator.py:64`
- Active graph: `backend/app/orchestrator/job_defs.py:1600-1614`
- Intent barrier active in same tick: `backend/app/orchestrator/orchestrator.py:200-208`

---

## 1. Executive summary

- This program migrates default runtime toward intent-canonical contracts with minimal operational risk.
- Work is split into:
  - **safe pre-parity phases** (observability + normalization shims + explicit contract boundaries)
  - **parity-gated phases** (translation + canonical risk/report + control-state provider + intent cutover)
  - **post-burn-in cleanup** (alias/path removal only after proven safety).
- **Implemented now (Phase 1 / Workstream 1):** mode alias normalization, planner position normalization (`quantity` canonical + `qty` fallback), shadow-intent field fix, and non-canonical/alias observability fields.

---

## 2. Phase-by-phase implementation plan

## Phase 1 (Safe pre-parity) — Boundary safety shims
Objective:
- Eliminate proven boundary mismatches without changing core trading semantics.

Changes:
- Normalize `live -> ibkr` before job filtering.
- Planner reads `quantity` canonical first, `qty` fallback alias second.
- Fix shadow interceptor to use canonical `TargetIntent.sleeve` and canonical intent fields.
- Add observability signals for alias and non-canonical path activation.

Risk level:
- **Low** (behavior-preserving + defensive compatibility).

Acceptance criteria:
- `live` mode alias no longer drops active jobs due to string mismatch.
- Planner works with both `quantity` and `qty` broker payloads.
- Shadow mode correctly intercepts by `sleeve`.
- Observability fields exist in outputs/logs.

Rollback:
- revert Phase 1 commit only.

Must NOT change:
- no execution-model cutover
- no contract module introduction
- no deletion of legacy/non-default paths.

---

## Phase 2 (Safe pre-parity) — Contract modules + validators
Objective:
- Introduce explicit contract modules/validators in compatibility mode.

Changes:
- Add modules for `SleeveOutput`, `RiskDecisionReport`, `PositionDeltaPlan`, `BrokerPosition`, `ExecutionResult`.
- Validate at runtime boundaries; accept compatibility aliases unless proven unsafe.

Risk level:
- **Low-Medium** (validation insertion).

Acceptance criteria:
- Contracts validate current default artifacts with no behavior break.
- Unsafe mismatches hard-fail only where already proven unsafe.

Rollback:
- disable validator gates via feature flags.

Must NOT change:
- no canonical source-of-truth switch yet.

---

## Phase 3 (Parity-gated) — Per-sleeve target->intent translation
Objective:
- Deterministically derive canonical intents for target-only sleeves.

Changes:
- Introduce translation module and policy table for core/vrp/selector/futures_overnight/statarb.
- Enforce traceability metadata on every translated intent.
- Hard-fail untranslatable statarb pair-only targets.

Risk level:
- **Medium**.

Acceptance criteria:
- All translatable sleeves emit valid canonical intents.
- Explicit untranslatable violations for ambiguous cases.

Rollback:
- disable translator flag and continue target-only path.

Must NOT change:
- no deletion of target artifacts.

---

## Phase 4 (Parity-gated) — Risk/report canonicalization
Objective:
- Emit canonical `risk_decision.v1` with provenance.

Changes:
- Keep legacy read shim while producing canonical report shape.
- Add `policy_version`, `input_contract_family`, `source_sleeves`, refs/hashes, `generated_by`.

Risk level:
- **Medium**.

Acceptance criteria:
- Planner gates on canonical risk decision.
- Legacy reader still works for compatibility paths.

Rollback:
- fallback to legacy report write/read mode.

Must NOT change:
- no execution cutover yet.

---

## Phase 5 (Parity-gated) — Control-state unification
Objective:
- One read/write abstraction for orchestrator/API/risk path.

Changes:
- Introduce `ControlStateProvider`.
- In-memory remains near-term active cache.
- Every write performs DB write-through.
- Orchestrator and risk path read through provider.

Risk level:
- **Medium-High**.

Acceptance criteria:
- same-tick mode consistency observable and tested.

Rollback:
- route orchestrator/API back to existing in-memory singleton.

Must NOT change:
- no scheduler/path retirement.

---

## Phase 6 (Parity-gated) — Intent-canonical cutover
Objective:
- Planner/risk consume canonical intents as primary input.

Changes:
- Targets remain compatibility outputs only.
- Ensure allocator scaling applied exactly once at canonical insertion point.

Risk level:
- **High** (semantic cutover).

Acceptance criteria:
- parity suite passes for planning/routing outcomes.
- traceability from routed order -> intent source is complete.

Rollback:
- switch feature flag to target-primary planning path.

Must NOT change:
- no deletion of compatibility artifacts.

---

## Phase 7 (Post-burn-in cleanup)
Objective:
- remove deprecated aliases and duplicate legacy math only after burn-in.

Changes:
- remove `qty` alias fallback after zero-hit burn-in window.
- remove legacy parser/deprecated paths proven unused.

Risk level:
- **High if premature; low after burn-in criteria met**.

Acceptance criteria:
- burn-in requirements met (section 11).

Rollback:
- re-enable aliases via emergency patch; restore previous tag.

Must NOT change:
- do not remove non-default runtime components without separate retirement proof.

---

## 3. Per-phase file targets

Phase 1 (implemented):
- `backend/app/core/runtime_mode.py`
- `backend/app/orchestrator/orchestrator.py`
- `backend/app/orchestrator/job_defs.py`
- `backend/app/orchestrator/intent_aggregator.py`
- `backend/tests/test_orchestrator.py`
- `backend/tests/test_stage2_cutover.py`

Phase 2:
- `backend/app/contracts/sleeve_output.py`
- `backend/app/contracts/risk_decision.py`
- `backend/app/contracts/position_delta_plan.py`
- `backend/app/contracts/broker_position.py`
- `backend/app/contracts/execution_result.py`
- boundary call sites in `job_defs.py`, `orchestrator.py`, broker adapters
- tests under `backend/tests/contracts/`

Phase 3:
- `backend/app/orchestrator/intent_translation.py`
- `backend/app/orchestrator/job_defs.py`
- tests for per-sleeve translation

Phase 4:
- `backend/app/orchestrator/job_defs.py`
- `backend/app/core/risk_gateway.py`
- risk report tests

Phase 5:
- `backend/app/orchestrator/control_state.py`
- `backend/app/orchestrator/durable_control_state.py`
- `backend/app/api/control_routes.py`
- `backend/app/orchestrator/orchestrator.py`
- `backend/app/core/risk_gateway.py`

Phase 6:
- `backend/app/orchestrator/job_defs.py`
- `backend/app/orchestrator/orchestrator.py`
- `backend/app/orchestrator/intent_aggregator.py`

Phase 7:
- legacy/alias removal at touched boundaries only

---

## 4. Per-phase tests

Phase 1 (implemented):
- mode alias normalization unit test
- planner position normalization tests (`quantity` and `qty`)
- intent aggregator target-intent file metric test
- shadow interception sleeve-field correctness test

Phase 2:
- contract validation tests per contract module
- compatibility-shape acceptance tests

Phase 3:
- translation determinism tests per sleeve
- statarb untranslatable hard-fail test
- allocator single-application test

Phase 4:
- `risk_decision.v1` schema/provenance tests
- legacy read shim compatibility tests

Phase 5:
- control write-through tests
- same-tick mode consistency tests across orchestrator+risk

Phase 6:
- intent-canonical E2E parity tests
- order traceability tests

Phase 7:
- alias-hit zero-window assertions
- no-regression smoke tests

---

## 5. Per-phase acceptance criteria

Phase 1:
- no regression in default orchestration flow
- alias/activation observability present
- compatibility behaviors preserved

Phase 2:
- explicit contracts validate active artifacts at boundaries

Phase 3:
- deterministic canonical intent derivation for all translatable sleeves

Phase 4:
- canonical risk decision emitted with provenance fields

Phase 5:
- unified control-state provider path in default runtime

Phase 6:
- canonical intent-driven planning/risk in default runtime

Phase 7:
- cleanup only after burn-in invariants pass

---

## 6. Per-phase rollback conditions

- Phase 1: any route/planner regression tied to new normalization fields.
- Phase 2: validation false-positives that block valid historical shapes.
- Phase 3: translation failure rate > threshold or parity mismatch.
- Phase 4: planner rejects due to report shape migration errors.
- Phase 5: mode inconsistency between provider consumers.
- Phase 6: parity drift in routed quantity/side or risk gating.
- Phase 7: any post-cleanup alias-dependent runtime failure.

---

## 7. Contract definitions to introduce

- `SleeveOutput` (`sleeve_output.v1`): migration-aware (`targets` or `intents` required when status=ok).
- `IntentSpec` / existing `TargetIntent`: canonical execution/risk contract.
- `RiskDecisionReport` (`risk_decision.v1`): includes provenance fields.
- `PositionDeltaPlan` (`position_delta_plan.v1`): canonical planner output.
- `BrokerPosition` (`broker_position.v1`): canonical `quantity`, compatibility alias `qty`.
- `ExecutionResult` (`execution_result.v1`): canonical placement result schema.

---

## 8. Per-sleeve translation policy table

| Sleeve | Current artifacts | Canonical intent derivation | asset_class source | phase default | multiplier source | allocator insertion point | Unsupported/ambiguous |
|---|---|---|---|---|---|---|---|
| core | signals + targets + core_intents | direct intents preferred; translator fallback from targets only if missing intents | handler/config map | futures_open | config/default | after translation before persistence | none |
| vrp | signals + targets | target row -> intent row | policy map: EQUITY | intraday | 1.0 | same canonical insertion | target extra fields become metadata only |
| selector | signals + targets | target row -> intent row | policy map: EQUITY | intraday | 1.0 | same canonical insertion | ensure no double-scaling |
| futures_overnight | signals + targets | target row -> intent row | policy map: FUTURE | futures_open | symbol map/config | same canonical insertion | unknown futures symbol multiplier => hard-fail |
| statarb | signals + targets (may include pair-level shape) | only symbol-level target rows translatable | policy map: EQUITY | intraday | 1.0 | same canonical insertion | pair-only targets => hard-fail `UNTRANSLATABLE_TARGET` |

---

## 9. Observability and metrics plan

Required metrics/signals:
- `mode_alias_normalization_hits_total`
- `position_qty_alias_hits_total`
- `noncanonical_target_intent_files_total`
- `intent_translation_success_total`
- `intent_translation_failure_total`
- `intent_translation_untranslatable_total{reason}`
- `risk_decision_status_total{status}`
- `execution_plan_reject_total{reason}`
- `same_tick_mode_mismatch_total`
- `allocator_double_scale_detected_total`

Required traceability fields on translated intents:
- `source_sleeve`
- `source_artifact_path`
- `source_row_index`
- `derivation_policy_version`
- `allocator_scale_applied`
- `translated_at`

Order trace requirement:
- each routed order must include link to canonical intent id/source.

---

## 10. Migration invariants

1. `status=ok` SleeveOutput must carry at least one routable family (`targets` or `intents`).
2. Allocator scaling is applied exactly once.
3. Planner reads canonical `quantity` first; alias `qty` second.
4. No routing if canonical risk decision not `ok`.
5. Translation must be deterministic for same input artifact.
6. Statarb pair-only targets cannot silently degrade; must explicit hard-fail.
7. Non-default paths (PhaseScheduler/route_phase_orders/dag_loader) remain adapter-only and intact during migration.

---

## 11. Burn-in requirements before deletion

Before any alias/legacy deletion:
- minimum burn-in window complete (trading-day count agreed by ops)
- zero hits for deprecated alias counters over window
- zero same-tick mode mismatch events
- no unresolved translation failures
- parity dashboards stable for routed order count, net/gross exposure, rejection rates
- rollback drill validated on current deployment target

---

## 12. Final recommended PR sequence

1. **PR-1 (safe pre-parity, implemented):** boundary shims + metrics + focused tests.
2. **PR-2 (safe pre-parity):** contract modules + compatibility validators.
3. **PR-3 (parity-gated):** deterministic per-sleeve translation + traceability.
4. **PR-4 (parity-gated):** canonical risk decision report + provenance.
5. **PR-5 (parity-gated):** control-state provider unification with write-through.
6. **PR-6 (parity-gated):** intent-canonical planner/risk cutover.
7. **PR-7 (post-burn-in):** alias/legacy cleanup.

Safe before parity tests: PR-1, PR-2 only.

---

## Phase 1 execution log (completed in this iteration)

Implemented file-level changes:
- mode alias normalization helper and orchestrator pre-filter normalization
- planner position normalization (`quantity` canonical, `qty` fallback + alias-hit counter)
- shadow interception fix to canonical `TargetIntent.sleeve`
- non-canonical targets-intent file metric surfaced from intent aggregator

Implemented tests:
- `backend/tests/test_orchestrator.py`:
  - mode alias normalization
  - planner reads `quantity`
  - planner records `qty` alias hits
- `backend/tests/test_stage2_cutover.py`:
  - target-intent file metric
  - shadow filter uses canonical sleeve field
