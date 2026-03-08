# Native Frontend Redesign — Execution Contract (Patch-Level)

This document converts the redesign plan into a strict implementation contract tied to the **actual codebase**.

## 1) Claim Validation With Code Evidence

## 1.1 `native_frontend/src/main.cpp`

| Claim | Old behavior | New behavior | Evidence | Status |
|---|---|---|---|---|
| Reconciler flush path was not wired to store | `StateReconciler` was instantiated but no flush callback assignment | `setFlushCallback` now routes buffered messages into `GlobalStore::routePayload` | `reconciler->setFlushCallback(...)` in `main.cpp` | Confirmed in code |
| Control-state bootstrap reconciliation was not triggered from REST | control-state callback only set a few fields | control-state callback now calls `reconciler->onBootstrapComplete(doc)` | `controlStateReceived` lambda in `main.cpp` | Confirmed in code |
| Jobs and broker were fetched in QML | no C++ handling for jobs/broker payloads | C++ now parses `jobGraphReceived` and `brokerStatusReceived` and updates store | `QObject::connect(...jobGraphReceived...)`, `...brokerStatusReceived...` | Confirmed in code |
| Freshness recompute timer absent | no periodic freshness recompute from C++ timer | `freshnessTimer` recomputes every 5s | `freshnessTimer` block in `main.cpp` | Confirmed in code |
| BuildEnvironment exposure uncertain | QML referenced BuildEnvironment without explicit context property | context property now explicitly set | `ctx->setContextProperty("BuildEnvironment", ...)` | Confirmed in code |

## 1.2 `native_frontend/src/qml/main.qml`

| Claim | Old behavior | New behavior | Evidence | Status |
|---|---|---|---|---|
| Global shell lacked trust-first status hierarchy | header had status, but mixed with per-tab logic and prior broker auto-connect XHR logic | header now always shows mode, paused/running, broker up/down, freshness chip | status badges + `DataAgeChip` in header | Confirmed in code |
| IA did not match target operator nav | old tab model was `Overview/Signals/Risk/...` | tab model now `Command Center/Sleeves/Risk & Guardrails/...` | `Repeater model` in `main.qml` | Confirmed in code |
| Default screen not command-centric | tab 0 was overview grid | tab 0 now loads `CommandCenterTab.qml` | first `Loader` entry in `StackLayout` | Confirmed in code |
| Direct QML broker business logic remained | prior file had startup broker XHR/retry timers | current file has no XHR/TImer network logic | absence of `XMLHttpRequest` and broker timer blocks | Confirmed in code |

## 1.3 Major views/tabs

| View | Old behavior | New behavior | Status |
|---|---|---|---|
| `CommandCenterTab.qml` | did not exist | new command-center composite with alert rail, action queue, sleeve status and critical KPIs | Confirmed in code |
| `JobsTab.qml` | owned direct `/api/control/job-graph` XHR polling and local model parsing | now binds to centralized `GlobalStore` job stats and explicitly marks row-detail unavailable until typed adapter exists | Confirmed in code |
| `RiskTab.qml` | kill switch visual state was local optimistic toggle (`Switch checked`) | kill switch row status now computed via `KillSwitch.isSleeveHalted(...)`, with halt/resume action button | Confirmed in code |
| `PortfolioTab.qml` | inline table + duplicated card markup | now uses reusable primitives (`MetricCard`, `PositionsTable`) and freshness chip | Confirmed in code |
| `LabTab.qml` | used placeholder model hash in promotion POST flow | promotion now explicitly blocked for unavailable candidate hash and no fake promote POST is sent on success callback | Confirmed in code |
| `SignalsTab.qml` | included status labels that could imply live state | now includes explicit text indicating limited/unavailable status feed for some rows | Confirmed in code |
| `SettingsTab.qml` | had static infra claims with weak runtime truth cue | now includes backend reachable cue and less absolute RHI claim text | Confirmed in code |

## 1.4 `engine/GlobalStore.*`

| Claim | Old behavior | New behavior | Evidence | Status |
|---|---|---|---|---|
| No global freshness domain | no explicit freshness properties or endpoint update timestamps | added `dataFreshness`, `backendReachable`, timestamps, job counters | new `Q_PROPERTY` set and fields in `ApplicationState` | Confirmed in code |
| No store-side jobs counters | no `jobTotal/jobRunning/jobFailed` properties | added properties + `setJobStats` mutator | `GlobalStore.h` | Confirmed in code |
| No explicit backend disconnect state | network errors were not mapped to store-level disconnected status | `markBackendDisconnected()` sets `backend_reachable=false` and `data_freshness=disconnected` | `GlobalStore.h` | Confirmed in code |
| Freshness did not recompute with age thresholds | none | `updateFreshness()` computes `fresh/stale/degraded/disconnected` from timestamp age | `GlobalStore.h` | Confirmed in code |

## 1.5 `engine/UiSynchronizer.*`

| Claim | Old behavior | New behavior | Evidence | Status |
|---|---|---|---|---|
| Store frame commit not called from queue drain | route payloads processed, but no `commitFrameUpdates()` in drain cycle | now invokes `m_store->commitFrameUpdates()` every drain tick | `UiSynchronizer.cpp` | Confirmed in code |

## 1.6 `engine/StateReconciler.*`

| Claim | Old behavior | New behavior | Evidence | Status |
|---|---|---|---|---|
| Reconciler had callback API but was underused | callback setter existed but not wired in app bootstrap | callback now assigned in `main.cpp`; bootstrap completion called from control-state REST | `StateReconciler.h` + `main.cpp` | Confirmed in code |

## 1.7 `network/RestClient.*`

| Claim | Old behavior | New behavior | Evidence | Status |
|---|---|---|---|---|
| Jobs/broker API not centralized in C++ | only control/portfolio/risk methods | added `getJobGraph()` + `getBrokerStatus()` + corresponding signals | `RestClient.h/.cpp` | Confirmed in code |

## 1.8 `qml.qrc`

| Claim | Old behavior | New behavior | Evidence | Status |
|---|---|---|---|---|
| Packaging fragile because only root qml was listed | only `main.qml` resource listed | now includes all views + components | `qml.qrc` entries | Confirmed in code |

## 1.9 Kill switch integration

| Claim | Old behavior | New behavior | Status |
|---|---|---|---|
| UI state could diverge from backend halt truth | local switch checked state drove label semantics | label/button now derive state from `KillSwitch.isSleeveHalted(sleeveId)` each render | Confirmed in code |

## 1.10 FIDO / Lab promotion path

| Claim | Old behavior | New behavior | Status |
|---|---|---|---|
| Placeholder hash path could imply live-safe promotion | placeholder hash + promote POST call in UI callback | explicit unavailable hash + blocked action message; no success promote POST dispatch | Confirmed in code |

## 1.11 Jobs data path

| Claim | Old behavior | New behavior | Status |
|---|---|---|---|
| Jobs business networking in QML | QML XHR polling owned parsing and UI counters | C++ `RestClient` polling + `GlobalStore` counters + QML bind-only | Confirmed in code |

## 1.12 Freshness/status presentation path

| Claim | Old behavior | New behavior | Status |
|---|---|---|---|
| No explicit cross-app freshness indicator | none | `GlobalStore.dataFreshness` exposed and shown in header + key views via `DataAgeChip` | Confirmed in code |

---

## 2) Patch Plan (Strict Patch Sets)

> Note: Patch sets include already-implemented work and remaining execution tasks. Treat as canonical sequencing contract.

## Patch 1 — Trust/Freshness Indicators
- **Files modified:**
  - `native_frontend/src/engine/GlobalStore.h`
  - `native_frontend/src/main.cpp`
  - `native_frontend/src/qml/main.qml`
  - `native_frontend/src/qml/views/PortfolioTab.qml`
  - `native_frontend/src/qml/views/JobsTab.qml`
- **Files created:** `native_frontend/src/qml/components/DataAgeChip.qml`, `StatusBadge.qml`
- **Files deprecated/deleted:** none
- **Purpose:** Surface backend reachability + freshness age semantics everywhere critical.
- **Behavior change:** UI now explicitly displays `fresh/stale/degraded/disconnected`.
- **Risk:** Medium (state semantics touch central store)
- **Rollback:** revert `GlobalStore` freshness fields/methods and QML chip usage.
- **Dependencies:** centralized timestamp updates from control/portfolio/jobs callbacks.
- **Acceptance criteria:**
  1. Freshness shows `fresh` shortly after successful polls.
  2. Freshness transitions to `stale/degraded` when polls stop.
  3. Freshness shows `disconnected` on network error callback.

## Patch 2 — Remove Misleading Static/Live Ambiguity
- **Files modified:**
  - `native_frontend/src/qml/views/RiskTab.qml`
  - `native_frontend/src/qml/views/SignalsTab.qml`
  - `native_frontend/src/qml/views/LabTab.qml`
  - `native_frontend/src/qml/views/JobsTab.qml`
- **Files created:** none
- **Files deprecated/deleted:** direct jobs row detail presentation (temporarily unavailable)
- **Purpose:** Ensure unsupported/live-unwired panels are labeled as such.
- **Behavior change:** explicit unavailable/monitoring labels replace fake-live implication.
- **Risk:** Low
- **Rollback:** restore previous text/status labels.
- **Dependencies:** none
- **Acceptance criteria:**
  1. No panel implies live backend truth when not wired.
  2. Lab promotion does not claim real candidate hash availability.

## Patch 3 — Centralize Jobs/Broker into C++ Runtime
- **Files modified:**
  - `native_frontend/src/network/RestClient.h`
  - `native_frontend/src/network/RestClient.cpp`
  - `native_frontend/src/main.cpp`
  - `native_frontend/src/engine/GlobalStore.h`
  - `native_frontend/src/qml/views/JobsTab.qml`
- **Files created:** none
- **Files deprecated/deleted:** QML jobs XHR logic
- **Purpose:** Remove business networking from QML for operations-critical state.
- **Behavior change:** jobs/broker now parsed in C++; QML binds store properties.
- **Risk:** Medium
- **Rollback:** revert new rest endpoints/signals and reconnect old QML polling (not recommended)
- **Dependencies:** `/api/control/job-graph` and `/api/control/broker-status` endpoints
- **Acceptance criteria:**
  1. `JobsTab.qml` has no `XMLHttpRequest`.
  2. `main.cpp` updates job stats on `jobGraphReceived`.
  3. broker status badge updates from `brokerStatusReceived`.

## Patch 4 — Fix Notifier/Reconciler/Update Propagation
- **Files modified:**
  - `native_frontend/src/engine/UiSynchronizer.cpp`
  - `native_frontend/src/main.cpp`
- **Files created:** none
- **Files deprecated/deleted:** none
- **Purpose:** enforce coherent store signaling and bootstrap stream reconciliation.
- **Behavior change:** `commitFrameUpdates()` called each drain; reconciler callback + bootstrap now wired.
- **Risk:** Medium-High
- **Rollback:** revert synchronizer and reconciler wiring.
- **Dependencies:** store methods and reconciler APIs already present.
- **Acceptance criteria:**
  1. `UiSynchronizer::drainQueue` always calls `commitFrameUpdates`.
  2. `setFlushCallback` assigned in bootstrap path.
  3. control-state callback triggers `onBootstrapComplete`.

## Patch 5 — Reusable QML Primitives and Tokens
- **Files modified:**
  - `native_frontend/src/qml/views/PortfolioTab.qml`
  - `native_frontend/src/qml/views/JobsTab.qml`
  - `native_frontend/src/qml/main.qml`
- **Files created:**
  - `native_frontend/src/qml/components/Theme.qml`
  - `MetricCard.qml`, `PanelScaffold.qml`, `AlertRail.qml`, `GuardrailMatrix.qml`, `ActionQueuePanel.qml`, `SleeveStatusRow.qml`, `PositionsTable.qml`, `DetailDrawer.qml`
- **Files deprecated/deleted:** none
- **Purpose:** reduce style duplication and normalize operator semantics.
- **Behavior change:** tabs consume shared primitives.
- **Risk:** Low-Medium
- **Rollback:** switch tabs back to inline markup.
- **Dependencies:** qml resource packaging update
- **Acceptance criteria:**
  1. Shared primitives load without import errors.
  2. Portfolio and Jobs screens render through shared components.

## Patch 6 — Command Center Shell
- **Files modified:** `native_frontend/src/qml/main.qml`
- **Files created:** `native_frontend/src/qml/views/CommandCenterTab.qml`
- **Files deprecated/deleted:** overview tab route in main shell
- **Purpose:** put operator-critical state/action in default first screen.
- **Behavior change:** first tab is command center with alert/action/risk focus.
- **Risk:** Medium
- **Rollback:** repoint first loader to old overview.
- **Dependencies:** primitives + store properties
- **Acceptance criteria:**
  1. App starts on Command Center.
  2. mode/pause/freshness/alert/action are visible without tab-switching.

## Patch 7 — Risk & Guardrails Cleanup
- **Files modified:** `native_frontend/src/qml/views/RiskTab.qml`
- **Files created:** none
- **Files deprecated/deleted:** local optimistic kill-switch switch semantics
- **Purpose:** align risk UI with authoritative halt state.
- **Behavior change:** status text derives from `KillSwitch.isSleeveHalted`.
- **Risk:** Medium
- **Rollback:** restore old `Switch` path
- **Dependencies:** KillSwitch bridge invokables
- **Acceptance criteria:**
  1. First render reflects actual halt status.
  2. Halt/resume action updates reflected state labels.

## Patch 8 — Portfolio & Execution Cleanup
- **Files modified:** `native_frontend/src/qml/views/PortfolioTab.qml`
- **Files created:** `native_frontend/src/qml/components/PositionsTable.qml`
- **Files deprecated/deleted:** inline table markup in tab
- **Purpose:** consistent table/panel scaffolding and freshness cue.
- **Behavior change:** standardized metric cards + reusable positions table.
- **Risk:** Low
- **Rollback:** revert to prior portfolio QML
- **Dependencies:** component library
- **Acceptance criteria:**
  1. Positions grid still binds `PositionsGrid` correctly.
  2. no row/value regression in displayed holdings.

## Patch 9 — Lab Truthfulness/Safety Cleanup
- **Files modified:** `native_frontend/src/qml/views/LabTab.qml`
- **Files created:** none
- **Files deprecated/deleted:** fake-success promotion path with placeholder model hash
- **Purpose:** no unsafe implied promotion capability without candidate hash contract.
- **Behavior change:** promotion attempt is explicitly blocked when hash unavailable.
- **Risk:** Low
- **Rollback:** restore previous promote POST path (unsafe)
- **Dependencies:** backend candidate hash contract (future)
- **Acceptance criteria:**
  1. UI states clearly why promotion is blocked.
  2. No promote POST is sent on successful FIDO callback in unavailable-hash mode.

## Patch 10 — Packaging/Resource Hardening
- **Files modified:** `native_frontend/src/qml/qml.qrc`
- **Files created:** none
- **Files deprecated/deleted:** implicit file-system-only loading assumption
- **Purpose:** package all views/components explicitly.
- **Behavior change:** resource list now includes all current views/components.
- **Risk:** Low
- **Rollback:** revert qrc entries
- **Dependencies:** component/view file stability
- **Acceptance criteria:**
  1. QML engine can resolve all current view/component files from resources.
  2. no missing import/resource errors at startup.

---

## 3) Acceptance Tests (Per-Patch)

## Patch 1 tests
- **Manual:** disconnect backend -> header freshness becomes disconnected; reconnect -> returns fresh after poll.
- **Regression risk:** stale threshold too aggressive or too lax.
- **Test opportunity:** add store unit test for freshness threshold transitions.
- **Expected UI:** DataAgeChip state transitions correctly without restart.

## Patch 2 tests
- **Manual:** inspect risk/lab/jobs/sleeves labels for explicit availability truthfulness.
- **Regression risk:** over-warning reduces usability.
- **Test opportunity:** UI snapshot tests (text assertions).
- **Expected UI:** unavailable feeds clearly marked, no fake-live impression.

## Patch 3 tests
- **Manual:** confirm `JobsTab` updates from poll cycle and `main.qml` has no job/broker XHR code.
- **Regression risk:** parse mismatch on job graph payload fields.
- **Test opportunity:** C++ parsing helper unit tests for job graph sample payload.
- **Expected UI:** job counters update with backend poll; no QML networking side effects.

## Patch 4 tests
- **Manual:** verify streamed values update in bound labels without forced rest calls.
- **Regression risk:** notifier storms/perf regressions.
- **Test opportunity:** integration test with synthetic queue pushes + notifier counts.
- **Expected UI:** reactive bindings continue updating under stream load.

## Patch 5 tests
- **Manual:** open command/portfolio/operations and verify consistent card/badge styles.
- **Regression risk:** component import path failures.
- **Test opportunity:** QML load test for component catalog.
- **Expected UI:** no component missing errors; visuals consistent.

## Patch 6 tests
- **Manual:** startup defaults to Command Center even when backend unavailable.
- **Regression risk:** null bindings when data absent.
- **Test opportunity:** QML engine smoke test with empty store state.
- **Expected UI:** command center loads with degraded/disconnected indicators, not crash.

## Patch 7 tests
- **Manual:** toggle halt/resume and verify label follows `isSleeveHalted` truth.
- **Regression risk:** state not refreshed after action.
- **Test opportunity:** bridge mock test for halt mask transitions.
- **Expected UI:** authoritative halt status visible on first render and post-action.

## Patch 8 tests
- **Manual:** confirm position rows and columns unchanged from previous functionality.
- **Regression risk:** table delegate/column width regression.
- **Test opportunity:** model row-count/value assertions on sample record batch.
- **Expected UI:** positions table correctness preserved.

## Patch 9 tests
- **Manual:** run promotion flow with FIDO callback success and verify no promote POST is sent in unavailable-hash mode.
- **Regression risk:** accidental disablement once backend hash contract arrives.
- **Test opportunity:** QML signal flow test for `onSignatureComplete` branch.
- **Expected UI:** user sees explicit blocked message.

## Patch 10 tests
- **Manual:** packaged run loads all tabs/components.
- **Regression risk:** stale qrc entries after renames.
- **Test opportunity:** build-time script to verify qrc contains all view/component files.
- **Expected UI:** no `qrc` missing-file errors.

---

## 4) Current State vs Target State Matrix

| Area | Current behavior | Target behavior | Implemented now | Not yet implemented | Backend dependency |
|---|---|---|---|---|---|
| Header/global status | mode/pause/broker/freshness visible in header | always-on trust strip with stronger diagnostics | Yes | richer latency/age detail | endpoint timestamps + health contract |
| Command Center | new default command center exists | full operator incident/action cockpit | Partial | advanced drilldowns/timeline | richer alert/guardrail backend feeds |
| Sleeves/Signals | signals tab retained with explicit availability notes | full sleeve health + inference freshness per sleeve | Partial | live sleeve status contract | sleeve health endpoint/stream |
| Risk & Guardrails | kill switch authoritative state in UI + explicit monitoring labels | full live guardrail matrix with thresholds/action provenance | Partial | full backend guardrail feed | risk-checks/guardrail state contract |
| Portfolio & Execution | reusable positions surface + freshness chip | full execution pipeline (orders/fills/latency) | Partial | detailed execution panels | execution endpoints |
| Lab | explicit blocked promotion when hash unavailable | safe promotion flow with candidate hash lineage and signed commit | Partial | candidate hash selection + audit trail | backend model-candidate contract |
| Operations/Jobs | centralized job stats from C++ and no QML XHR | typed row-level jobs model and detailed ops workflow | Partial | typed jobs model + controls | job detail contracts |
| Settings/Diagnostics | improved truthfulness for runtime diagnostics | comprehensive runtime diagnostics panel | Partial | deep runtime probes/export | backend/runtime diagnostics APIs |
| Data runtime/freshness | store-level freshness state and timers | full source-aware freshness with per-domain age chips | Partial | per-panel source age + stream health | stream heartbeat contracts |
| Packaging/resources | all views/components listed in qrc | enforce packaged-load invariants in CI | Partial | automated qrc completeness check | none |

---

## 5) Deferred Work Risk Register

| Deferred item | Why deferred | Operator risk if undone | Technical risk | Backend dependency | Priority |
|---|---|---|---|---|---|
| Typed jobs row model in C++ | first pass removed risky QML networking quickly | medium (reduced operational detail) | medium | stable `/job-graph` row schema | P1 |
| Full guardrail matrix live-binding | requires richer contract than current UI-friendly snapshot | high if traders assume controls are active | medium-high | guardrail state endpoint/stream | P0 |
| Lab candidate hash contract | unavailable in frontend runtime currently | medium (cannot promote from UI) but safer than fake promotion | low | model candidate metadata endpoint | P1 |
| Command center drilldown/drawer workflows | deferred to keep patch incremental | medium (slower root-cause navigation) | medium | alert/sleeve detail contracts | P2 |
| Per-panel freshness ages (not just global freshness) | needs domain timestamps and rendering policy | medium | medium | per-domain heartbeat timestamps | P1 |
| Automated qrc completeness check | no script added yet | low runtime risk if manual discipline slips | low | none | P2 |

---

## 6) Hard “Do Not Regress” List

1. App startup stability in `main.cpp` bootstrap chain.
2. QML loadability for root + all tab views/components.
3. Positions table correctness (`PositionsGrid` binding and row rendering).
4. Broker status visibility in header.
5. Paper/live visibility always present.
6. Paused/running visibility always present.
7. Kill switch state correctness must reflect backend truth on first render.
8. No fake live data in operator-critical panels.
9. No silent error swallowing for critical status panels.
10. No new direct QML business networking for critical data paths.

---

## 7) Recommended Next Steps

### A. Recommended next implementation patch

**Patch 11 — Typed Operations Model + Job Detail Panel Re-enable**
- Add a C++ typed adapter/model for `job-graph` rows (name, deps, sessions, last status, last run, duration, error).
- Publish to QML via dedicated model object in `main.cpp` context.
- Replace Operations “UNAVAILABLE” placeholder with live-backed table and explicit stale/error states.

### B. Recommended next review checkpoint for ChatGPT

**Checkpoint:** after Patch 11 compiles and loads, request focused review on:
1. schema validation and fallback behavior for malformed job payloads,
2. UI stale/error states in Operations under backend disconnect,
3. regression check against the “Do Not Regress” list,
4. whether guardrail matrix can be migrated from placeholder rows to true live bindings next.



## Implementation Reality Update (current wave)

- Added backend guardrail contract endpoint `GET /api/control/guardrails/status` (`guardrails_status.v1`) with deterministic five-row output (`ece_tracker`, `mmd_liveguard`, `max_drawdown`, `gap_risk_filter`, `slippage_monitor`).
- Frontend now polls and parses guardrails in C++ (`RestClient::getGuardrailStatus`, `guardrailStatusReceived` handler in `main.cpp`) and publishes rows through `GlobalStore.guardrails`.
- Risk screen no longer uses hardcoded monitoring literals for those guardrails; rows are bound to `GlobalStore.guardrails` and explicitly surface unknown/unwired states.
- Added typed `JobTableModel` and re-enabled row-level Operations rendering from C++ model (`JobsModel`) with no QML networking/parsing.
- Visual shell and major screens were redesigned with shared primitives (`SectionHeader`, `SeverityBanner`, `OperatorTable`) while preserving existing runtime-truth invariants.
