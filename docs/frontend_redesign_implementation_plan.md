# Native Frontend Redesign Validation + Implementation Plan

## Skill usage note
No session skill was invoked. The available skills (`skill-creator`, `skill-installer`) do not match this task (frontend architecture validation and implementation in existing native Qt/QML code).

## PHASE A — Codebase validation against prior audit

### Correct findings

- `main.cpp` is the real runtime composition root and wires ZMQ receiver, UI synchronizer, global store, REST polling, kill switch bridge, and QML context properties.
- `main.qml` was previously mixing presentation and business logic via direct XHR (broker/connect + retries).
- `GlobalStore` had an RCU-like pattern and explicit frame commit API, but `UiSynchronizer` was not calling `commitFrameUpdates()`.
- `StateReconciler` existed but callback/bootstrap wiring was incomplete for practical startup reconciliation.
- `RestClient` only covered a subset of endpoints; jobs and broker status were fetched in QML.
- QML packaging was fragile: `qml.qrc` previously only listed `main.qml`.
- Several sections presented static/non-live content as if authoritative.

### Partially correct findings

- `BuildEnvironment` registration risk was real in practice for this build style; now solved by explicit context property exposure in `main.cpp`.
- `ExecutionGrid`/Arrow worker wiring concerns remain partially true: this pass focused on trust/IA and did not fully implement all Arrow stream targets.

### Incorrect findings / updates

- None of the major risk findings were fundamentally incorrect; most were accurate and have now been addressed incrementally.

### Additional issues discovered

1. Operator status hierarchy had no single command surface; all domains were equal-weight tabs.
2. Kill switch row state in Risk tab was local-UI toggled instead of reflecting backend truth at render time.
3. Jobs table relied on QML-side dynamic parsing and swallowed failures silently.
4. Lab promotion path used placeholder hash semantics, creating a risk of false confidence.

### Highest-risk frontend problems (real)

1. Misleading static/live ambiguity in operator-critical panels.
2. Incomplete notifier/freshness wiring -> stale confidence risks.
3. Business networking inside QML with ad hoc error handling.
4. Missing explicit freshness/degraded/disconnected UI affordances.
5. Weak paper/live/paused safety prominence relative to other content.

## PHASE B — Target architecture (concrete for this repo)

### 1) Target folder/module structure

```text
native_frontend/src/
  engine/          # ingest + reconciliation + store
  network/         # REST adapters
  hardware/        # kill switch + FIDO
  rendering/       # custom QSG items
  qml/
    components/    # reusable UI primitives
    views/         # route-level surfaces
```

### 2) Target state/data architecture

- `GlobalStore` remains central state with explicit freshness and operations fields.
- `RestClient` owns periodic operational endpoint retrieval; QML should bind, not fetch.
- `UiSynchronizer` must always call `commitFrameUpdates()` for coherent reactive updates.
- `StateReconciler` callback should flush to `GlobalStore` and bootstrap off control-state response.

### 3) Target QML component architecture

Reusable primitives introduced:

- `MetricCard`
- `StatusBadge`
- `DataAgeChip`
- `PanelScaffold`
- `AlertRail`
- `SleeveStatusRow`
- `GuardrailMatrix`
- `ActionQueuePanel`
- `PositionsTable`
- `DetailDrawer`

### 4) Target page/screen architecture

Top-level IA implemented as:

- Command Center
- Sleeves
- Risk & Guardrails
- Portfolio & Execution
- Lab
- Operations
- Settings / Diagnostics

### 5) Target alert/freshness/status system

- Header now always displays execution mode, paused/running, broker up/down, and freshness chip.
- `GlobalStore.dataFreshness` computed from endpoint timestamps and backend reachability.
- Guardrail panel explicitly marks unavailable/unwired rows.

### 6) Target API/service layer

- `RestClient` expanded with `getJobGraph()` and `getBrokerStatus()`.
- `main.cpp` parses responses and updates `GlobalStore` (jobs and broker state), removing jobs polling from QML.

### 7) Target chart/table/panel abstractions

- `PositionsTable` wraps standardized table scaffold.
- Existing custom rendering items preserved, but unwired sections are labeled explicitly rather than implied live.

### 8) Target paper/live safety presentation model

- Always-visible mode badge in header.
- Paused/running badge in header.
- Freshness/disconnected chip in header.
- Simulation watermark preserved.

## PHASE C — Redesign specification

### New default screen

- **Command Center** (tab index 0) as default with:
  - mode/pause/freshness status
  - key metrics
  - alert rail
  - guardrail matrix
  - action queue
  - sleeve status rows

### Top-level navigation model

- Flat primary tabs with operator-centric names matching target IA.

### Screen content changes

- Command Center: new operational hub.
- Sleeves: retains signal visualization while explicitly marking unavailable status feeds.
- Risk & Guardrails: kill switch reflects backend truth; non-live breakers marked monitoring/unavailable.
- Portfolio & Execution: standardized metrics + reusable positions table.
- Lab: promotion blocked when model hash source unavailable (no fake promotions).
- Operations: jobs stats from centralized C++ polling; row table intentionally disabled until typed adapter exists.
- Settings/Diagnostics: better honesty around runtime properties.

### Move/merge/remove

- Merged old Overview into Command Center concept.
- Removed direct job graph QML fetching path.
- Removed auto-broker connect logic from QML.
- Deprecated misleading static breaker semantics by explicit labels.

### Required interactions

- Kill switch halt/resume controls now query backend truth (`KillSwitch.isSleeveHalted`) each render.
- FIDO promotion still available but action now blocked when candidate hash is unavailable.

### Drilldowns/drawers

- `DetailDrawer` primitive added for phased drilldown implementation.

## PHASE D — Implementation roadmap (exact touchpoints)

### Phase 0: remove misleading/static/live ambiguity

- Modified:
  - `native_frontend/src/qml/views/RiskTab.qml`
  - `native_frontend/src/qml/views/LabTab.qml`
  - `native_frontend/src/qml/views/SignalsTab.qml`
- Risk: Low
- Validation: visual smoke + logic review for explicit unavailable labels
- Rollback: restore previous tab files only

### Phase 1: design tokens + reusable primitives

- Created:
  - `native_frontend/src/qml/components/*.qml` primitives listed above
- Modified:
  - `native_frontend/src/qml/qml.qrc` to package all views/components
- Risk: Low-Medium
- Validation: QML load/import check
- Rollback: remove components and revert references

### Phase 2: unify data access/state handling; remove QML XHR business logic

- Modified:
  - `native_frontend/src/network/RestClient.h/.cpp`
  - `native_frontend/src/engine/GlobalStore.h`
  - `native_frontend/src/engine/UiSynchronizer.cpp`
  - `native_frontend/src/main.cpp`
  - `native_frontend/src/qml/views/JobsTab.qml`
  - `native_frontend/src/qml/main.qml`
- Risk: Medium-High
- Validation: compile + endpoint parsing + freshness state transitions
- Rollback: revert C++ networking/store changes in one commit

### Phase 3: rebuild Command Center and primary operational surfaces

- Created:
  - `native_frontend/src/qml/views/CommandCenterTab.qml`
- Modified:
  - `native_frontend/src/qml/main.qml`
  - `native_frontend/src/qml/views/PortfolioTab.qml`
- Risk: Medium
- Validation: operator walkthrough of critical state visibility
- Rollback: point tab 0 back to previous overview

### Phase 4: secondary screens and advanced workflows (deferred)

- Needed next:
  - typed per-job row adapter in C++ and re-enable operations detail table
  - model candidate hash contract for Lab promotion
  - richer drawers and cross-screen drilldowns
- Risk: Medium
- Validation: contract tests + UI scenario tests
- Rollback: feature-flag advanced modules

## PHASE E — What was implemented now vs deferred

### Implemented now

- Trust-first header and command center status system.
- Reusable QML primitives and panel scaffolds.
- Centralized jobs/broker reads into C++ `RestClient` + `GlobalStore`.
- Freshness metadata and visible freshness chip.
- `UiSynchronizer -> GlobalStore.commitFrameUpdates()` wiring fix.
- `StateReconciler` flush callback wiring + bootstrap trigger from control-state.
- qml resource packaging hardened.

### Deferred

- Full typed job rows model in C++ (currently high-level stats only).
- Full lab promotion candidate selection contract.
- Full detail-drawer orchestration and cross-screen navigation context.

