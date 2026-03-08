# Frontend Architecture + UX/UI Reverse-Engineering Report (Algae)

## Scope and method

This report is based on static reverse-engineering of the current frontend implementation in:

- `native_frontend/` (Qt6/QML + C++ native UI)
- `src-tauri/` (desktop supervisor shell, appears legacy/parallel)
- directly referenced backend API routes used by the frontend (`backend/app/api/control_routes.py`, `backend/app/api/orch_routes.py`)

I did **not** perform a visual runtime walkthrough with real market data in this pass, so where behavior depends on live backend payloads, those points are flagged as **runtime-uncertain**.

---

## 1) Executive Summary

### What the frontend currently is

Algae’s active frontend is a **native Qt Quick/QML desktop application** (not a browser SPA) with a C++ orchestration layer.

- Boot: `main.cpp` constructs Qt app, ZMQ receiver thread, UI synchronizer, global store, REST client, hardware gates, and QML engine.
- UI shell: one `ApplicationWindow` with a top status bar, top tab bar, and lazy-loaded tab views.
- Data planes:
  - **Streaming plane**: ZMQ event/grid sockets (ports 5556/5557) -> queue -> UI synchronizer -> global store.
  - **Polling plane**: REST calls every 30s to `/api/control/state` and `/api/control/portfolio-summary`.
  - **Ad hoc QML XHR plane**: some tabs directly call REST from QML JavaScript.

### Main architectural patterns in use

- Threaded ingestion (SPSC queue + timer-drained consumer loop).
- Global singleton state object (`GlobalStore`) exposed to QML via context property.
- QML tabs as independent monolithic view files, loaded through `Loader` inside a `StackLayout`.
- Custom QQuickItem renderers for fan chart / sankey / parallel coordinates.
- Partial out-of-band control integrations (kill switch SHM, FIDO flow).

### Main UX/UI weaknesses

- Heavy use of **static placeholders/mock labels** (especially signals/risk/lab/circuit-breaker content) mixed with live data, making operator trust difficult.
- Critical states (execution safety, live/paper boundary, stale data, alert causality) are present but not hierarchy-optimized for rapid decisioning.
- Visual language is uniform dark-card style without clear severity hierarchy, prioritization, or progressive disclosure.
- Tabbed IA is broad but shallow; little drilldown, trend context, or audit context.

### Main technical weaknesses

- **State update path appears incomplete**: `GlobalStore::commitFrameUpdates()` is designed to emit signals but is not wired from `UiSynchronizer`; this risks stale QML bindings for ZMQ-fed fields.
- **Reconciler bootstrapping appears partially wired**: `StateReconciler` exists, but no observed callback wiring from REST bootstrap completion to flush buffered messages.
- Architectural drift: components like `ArrowBuilderWorker`, `ExecutionGrid`, `VulkanChartItem`, and `WorkspaceManager` interfaces are present but underused/unwired in UI flow.
- Mixed data-fetch models (C++ REST + QML XHR + ZMQ) increase inconsistency and race risk.
- Several contracts are weakly typed JSON strings parsed ad hoc.

### Biggest opportunities

1. Unify all frontend data access through one typed data-service layer (remove QML XHR business logic).
2. Implement deterministic freshness model (timestamps, staleness badges, stream heartbeat, data source provenance).
3. Rebuild dashboard IA for trading operations: top-level command center + sleeve drilldowns + alert timeline + risk/guardrail cockpit.
4. Replace static placeholders with live-backed modules or clearly marked synthetic widgets.
5. Refactor tab monoliths into reusable panel system and design tokens.

---

## 2) Frontend File and Folder Map

## Primary frontend root: `native_frontend/`

| Path | Role | Notes |
|---|---|---|
| `native_frontend/src/main.cpp` | App entry, dependency wiring, context properties | Real boot chain lives here. |
| `native_frontend/src/qml/main.qml` | Root window, top bar, tabs, loader routing | Route host for all views. |
| `native_frontend/src/qml/views/*.qml` | Per-tab view modules | Large monolithic QML files, limited reuse. |
| `native_frontend/src/engine/*.h/.cpp` | Runtime engine: ZMQ, sync, state, alerts, reconciliation | Core data movement. |
| `native_frontend/src/network/RestClient.*` | C++ REST polling client | Polls control + portfolio summary every 30s. |
| `native_frontend/src/models/*.h/.cpp` | Arrow table/model processing | Partly wired, partly vestigial. |
| `native_frontend/src/rendering/*.h/.cpp` | Custom chart items | Fan/Sankey/Parallel items used in QML. |
| `native_frontend/src/hardware/*.h/.cpp` | Kill switch + FIDO integration | Used by Risk + Lab tabs. |
| `native_frontend/src/windowing/WorkspaceManager.*` | Workspace save/restore abstraction | Minimal UI integration. |
| `native_frontend/src/config/BuildEnvironment.h` | Build metadata singleton | Referenced in QML but not clearly registered. |
| `native_frontend/CMakeLists.txt` | Build graph | Includes many components not fully used. |

## Secondary desktop shell: `src-tauri/`

| Path | Role | Notes |
|---|---|---|
| `src-tauri/src/main.rs` | Tauri app + tray + backend process commands | Looks like parallel/legacy shell. |
| `src-tauri/src/backend_supervisor.rs` | Spawns `backend_dist/orchestrator.exe`, health checks | No direct linkage to QML frontend. |
| `src-tauri/tauri.conf.json` | Tauri config | References non-existent `frontend/` web app build paths. |

### Entry points

- **Current native frontend entry**: `native_frontend/src/main.cpp`
- **Potential legacy/alternate entry**: `src-tauri/src/main.rs`

### Layout containers / route handlers

- `native_frontend/src/qml/main.qml`
  - App window
  - Header status bar
  - Tab bar
  - `StackLayout + Loader` pseudo-routing

### Shared utilities / state containers / API clients

- State container: `engine/GlobalStore`
- Stream ingestion: `engine/ZmqReceiver`, `engine/UiSynchronizer`, `engine/StateReconciler`
- REST client: `network/RestClient`
- Alert state: `engine/AlertDag`

### Chart/render modules

- `rendering/FanChartItem`
- `rendering/SankeyDiagramItem`
- `rendering/ParallelCoordinatesItem`
- `rendering/VulkanChartItem` (compiled but not integrated in QML views)

### Reusable UI components / hooks / styles / config

- There are **no granular reusable QML components** (cards, badges, layout primitives) in a components folder.
- Styling is hard-coded inline in each tab (color literals, sizes, spacing constants).
- No tokenized theme or design system layer.

### Dead/duplicate/vestigial/confusing structure

- `ExecutionGrid` context property is injected but appears unused in QML.
- `ArrowBuilderWorker` is implemented but not instantiated/wired in `main.cpp`.
- `GlobalStore::setArrowWorker` exists but no observed invocation.
- `StateReconciler` is instantiated but bootstrap callback wiring appears incomplete.
- `src-tauri/tauri.conf.json` expects a web `frontend/` that is not present in repository tree.
- `BuildEnvironment` is referenced in QML without obvious registration as context/singleton in `main.cpp`.
- `qml.qrc` includes only `main.qml`; child tab files rely on runtime import-path filesystem presence, which can be deployment-fragile.

---

## 3) Frontend Runtime Architecture

## Boot sequence (actual)

1. OS/platform graphics/timer config in `main.cpp`.
2. `QGuiApplication` creation.
3. Optional KDDockWidgets init.
4. Construct runtime services:
   - `GlobalStore`
   - `AlertDag`
   - `StateReconciler`
   - `ZmqReceiver`
   - `UiSynchronizer`
   - `RestClient`
5. Wire selected signal/slot connections:
   - `UiSynchronizer.dataLossDetected -> GlobalStore.setDataLossActive(true)`
   - `GlobalStore.alertReceived -> AlertDag.processAlert(...)`
   - `RestClient.controlStateReceived -> GlobalStore setters`
   - `RestClient.portfolioSummaryReceived -> GlobalStore portfolio setters + PositionsGrid swap`
6. Start REST polling timer (30s + initial 2s delayed fetch).
7. Instantiate `ArrowTableModel` objects for execution/positions.
8. Instantiate KillSwitch and FIDO gateway.
9. Build QML engine, set context properties, load `qrc:/qml/main.qml`.
10. Start ZMQ receiver and synchronizer.
11. Restore workspace.

## Routing/layout composition

`main.qml` provides single-window pseudo-routing:

- Header toolbar (global status)
- TabBar with 8 tabs
- `StackLayout` with one `Loader` per tab

No URL router, no deep-linking, no screen state persistence beyond active tab default behavior.

## Component hierarchy (high-level)

```text
ApplicationWindow
├── ToolBar (status strip)
├── TabBar (8 domains)
└── StackLayout
    ├── OverviewTab
    ├── SignalsTab
    ├── RiskTab
    ├── AllocationTab
    ├── PortfolioTab
    ├── LabTab
    ├── JobsTab
    └── SettingsTab
```

## Global/local/derived state handling

- **Global state**: `GlobalStore` (C++ singleton object exposed via context property).
- **Specialized global state**: `AlertDag` (alert count and list).
- **Local tab state**: QML properties + `ListModel` (e.g., Jobs tab counters/model).
- **Derived state in QML**: many inline formulas (`toFixed`, status color conditions, mode labels).

## Side effects

- C++ timers for REST polling.
- QML timers/XHR for broker autoconnect and jobs polling.
- Hardware side-effects in QML click handlers (kill switch toggles, FIDO promotion POST).

## Data fetch/cache/refresh/render behavior

- ZMQ: push stream from backend -> queue -> periodic drain (`16ms`) -> route payload.
- REST: C++ poll every 30s for control + portfolio summary.
- Additional REST in QML:
  - `/api/control/broker-status`, `/api/control/broker/connect` (startup/retry)
  - `/api/control/job-graph` (jobs tab every 30s)
  - `/api/control/promote` (lab action)
- Rendering:
  - Most cards bind directly to `GlobalStore` properties.
  - Portfolio table binds to `PositionsGrid` model.
  - Custom chart items are rendered by QSG nodes.

## Polling/subscriptions/reactive updates

- Reactive: QML binding updates on notifier signals from store/engine.
- Interval polling: multiple timers (C++ and QML).
- Streaming: ZMQ with bounded queue/backpressure drop-head.

## Error/loading/empty-state handling

- In many QML XHR paths: `try/catch` with silent failure; little operator feedback.
- Jobs tab ignores parse/network errors (model just stops refreshing).
- No standardized loading skeletons/spinners except FIDO popup.
- Empty states are mostly implicit (zero values) rather than explicit “no data / stale data”.

## Backend-state change response quality

- Control and portfolio summary are reflected via poll-based updates.
- ZMQ updates should flow through `routePayload`, but signal-emission/latching design appears partially disconnected.
- No explicit freshness age displayed for key values.

---

## 4) Data Flow Analysis (by major screen/widget)

## Global header/status strip

**Consumes:** brokerConnected, currentSession, executionMode, systemPaused, dataLossFlag, totalPortfolioValue, totalPnl.

**Source:** `GlobalStore` (mixed ZMQ + REST writes).

**Transforms:** formatting only.

**Risks:**
- If `GlobalStore` notifier cadence is broken for stream updates, header can lag.
- Values come from mixed cadence (stream + 30s poll), risking temporal skew.

## Overview tab

**Consumes:** portfolio metrics + health + alert count + correlation.

**Sources:**
- `GlobalStore` for portfolio/health/correlation
- `AlertDag.activeAlertCount`

**Transforms:** static thresholds and color logic in QML.

**Risks:**
- “SYSTEM HEALTH” blends hard safety and informational values without freshness indicators.
- Correlation threshold hardcoded (`> 2.0`) with no context of calculation window/time.

## Signals tab

**Consumes:** sleeve weights/confidences from `GlobalStore`; `FanChartItem` visual.

**Sources:**
- `GlobalStore` fields
- FanChart data should come from `setActiveBatch` via `ArrowBuilderWorker` path.

**Risks:**
- Right-column sleeve status text is static strings (`Active/Monitoring/Training`), not backend-derived.
- If `ArrowBuilderWorker` is not wired, chart can be stale/empty while labels still look “live”.

## Risk tab

**Consumes:** statArbCorrelation, volRegimeOverride, alert count; kill switch bridge.

**Sources:** `GlobalStore`, `AlertDag`, `KillSwitchBridge`.

**Transforms/actions:** hardcoded circuit-breaker rows; toggles call `KillSwitch.haltSleeve/resumeSleeve`.

**Risks:**
- Switch checked state is local and defaults `true`; not initialized from actual backend halt mask.
- Hardcoded breaker statuses (“Armed/Active/Monitoring”) are presentation only.
- Operator may interpret static rows as actual active controls.

## Allocation tab

**Consumes:** leverage + sleeve weights from `GlobalStore`; sankey item.

**Sources:** `GlobalStore`; Sankey chart expected via renderer data path.

**Transforms:** target/current/deviation table currently mirrors same values and fixed deviation `0.0%`.

**Risks:**
- Table claims target vs current but displays identical values and no real drift computation.
- “Blend mode” / “Rebalance” text is static.

## Portfolio tab

**Consumes:** portfolio metrics + positions table.

**Sources:**
- metrics from `GlobalStore` (REST portfolio summary)
- table from `PositionsGrid` built from `holdings` JSON in REST response.

**Transforms:** table is direct display of selected fields.

**Trust level:** highest of current tabs (actual holdings payload mapping exists).

**Risks:**
- 30s refresh may be too coarse for execution monitoring.
- No data timestamp per row/payload.

## Lab tab

**Consumes:** shadow/promotion counts from `GlobalStore`, FIDO availability/waiting.

**Sources:** global store + hardware gateway + POST promote call.

**Transforms/actions:** static pipeline steps + promotion trigger with placeholder model hash.

**Risks:**
- `model_hash: "sha256:model_hash_placeholder"` indicates non-production stub path.
- Success/failure path is logged to console, not surfaced in operator-visible audit feedback.
- Parallel-coordinates chart may be static if no data feed wiring.

## Jobs tab

**Consumes:** `/api/control/job-graph` payload only (local `ListModel`).

**Sources:** direct QML XHR polling every 30s.

**Transforms:** maps backend job fields to row model; counts running/failed.

**Risks:**
- Separate data pipeline bypasses C++ REST client and global store.
- Silent failures produce stale-looking UI without explicit stale/error badge.

## Settings tab

**Consumes:** static strings + brokerConnected + FIDO state + BuildEnvironment singleton props.

**Sources:** `GlobalStore`, `FidoGateway`, expected `BuildEnvironment` singleton.

**Risks:**
- If `BuildEnvironment` is not registered, this tab may error or show blanks.
- Many claims are static (RHI shown as Vulkan despite runtime default potentially D3D11).

---

## 5) UI/UX Audit (product-interface perspective)

## Information hierarchy

- Primary KPI strip is useful, but too narrow for mission-critical triage.
- Critical control-state dimensions (paper/live, paused, data-loss, broker status, guardrails) are dispersed across header + multiple tabs.
- No dedicated “Operator Command Center” with prioritized risk stack.

## Readability + clutter

- Visual style is consistent dark cards, but typographic hierarchy is shallow.
- Many micro-labels + tiny font sizes (`9-11px`) increase cognitive load.
- Dense card grids without section-level narrative or causal grouping.

## Navigation/discoverability

- Top tabs are understandable but lack contextual breadcrumbs/drilldowns.
- Related flows are split awkwardly (risk controls in Risk tab, mode in header, promotion in Lab).
- No quick navigation for “what changed recently” / “what requires action now”.

## Accessibility

- Color coding relies heavily on red/green distinctions.
- Tiny fonts and low-contrast secondary text likely hinder accessibility.
- No keyboard-focused interaction model documented for critical controls.

## Responsiveness/use of real estate

- Designed around large desktop sizes (2560x1440 defaults).
- Cards scale but no adaptive layout strategy for smaller operator screens.
- Significant space spent on static text instead of time-series and change context.

## Consistency of visual language

- Card visual style is mostly consistent.
- Semantic consistency is weak: some badges are live-derived, others static placeholders but look identical.

## Trading suitability (critical-state clarity)

Current UI does not robustly surface, in one glance:

- data freshness/staleness by domain,
- confidence in displayed state provenance,
- risk guardrail breach timeline,
- per-sleeve execution lifecycle and failure impact,
- live vs paper blast radius boundaries.

### Specific confusion/overload/under-explained findings

- **Confusing:** Static status labels (“Active”, “Armed”, pipeline “Ready”) look operationally authoritative.
- **Hidden but important:** No explicit “last updated / source timestamp” next to critical numbers.
- **Visually overloaded:** Many equal-weight cards with similar color/shape; little prioritization by severity.
- **Under-explained:** Correlation metrics and thresholds lack period/context.
- **Redundant:** NAV/PnL/positions repeated in header, overview, and portfolio cards with little incremental value.
- **Should be regrouped:** guardrails, alerts, and execution mode should be one cohesive risk cockpit.

### Better representation recommendations (widget-level)

- Replace static circuit-breaker list with **live guardrail matrix table** (state, threshold, current, last breach, auto-action).
- Replace allocation static deviation rows with **target vs actual bar/heatmap + drift trend sparkline**.
- Replace monolithic alerts count with **alert timeline + root-cause tree + ack workflow**.
- Add **detail drawer** for any sleeve row/card showing model health, recent decisions, fills, and risk signals.
- Use **status badges with data age** (`fresh/stale/offline`) across all major panels.

---

## 6) Component Inventory

### Structural layout components

| Component | Purpose | Inputs | Outputs | Stateful | Reuse quality | Debt |
|---|---|---|---|---|---|---|
| `main.qml` root window | Global shell | `GlobalStore`, `AlertEngine`, context props | none | medium | low | Monolithic; mixed infra/UI logic |
| `OverviewTab.qml` etc | Per-domain pages | global context props | user actions in-tab | medium | low | No shared panel primitives |

### Navigation components

| Component | Purpose | Issues |
|---|---|---|
| `TabBar` in `main.qml` | Domain navigation | No deep links or route state, all tabs peers regardless of urgency |

### Data display components

| Component | Purpose | Inputs | Debt |
|---|---|---|---|
| KPI labels/cards across tabs | quick metrics | `GlobalStore` fields | duplicated templates; no common formatting util |
| `TableView` in Portfolio | positions display | `PositionsGrid` model | no sort/filter/column controls |
| Jobs `ListView` | job schedule/status | local `jobModel` from XHR | bypasses global data architecture |

### Forms/controls

| Component | Purpose | Inputs/events | Debt |
|---|---|---|---|
| Risk kill switches | sleeve halt/resume | `KillSwitch` invokables | local switch state not reconciled with backend truth |
| Lab promote button | submit promote request | `FidoGateway` + XHR | placeholder hash; weak success feedback |

### Charts/visualizations

| Component | Purpose | Data source | Reusability | Debt |
|---|---|---|---|---|
| `FanChartItem` | forecast fan | Arrow batch pointer | medium | feed path uncertain |
| `SankeyDiagramItem` | capital flow | setLayout vectors | medium | upstream wiring uncertain |
| `ParallelCoordinatesItem` | expert routing | setDistributionData | medium | appears not bound to live data |

### Stateful orchestration components

| Component | Purpose | Debt |
|---|---|---|
| `GlobalStore` | central state | notifier/latching design not fully integrated |
| `UiSynchronizer` | queue drain loop | does not call `commitFrameUpdates` |
| `StateReconciler` | startup dedupe | callback/bootstrap integration incomplete |
| `RestClient` | polling API | stringly typed payload signals |

### Utility/wrapper components

| Component | Purpose | Debt |
|---|---|---|
| `KillSwitchBridge` | QML adapter for SHM kill switch | no pull-sync to initialize controls |
| `WorkspaceManager` | layout persistence | not meaningfully exposed in UI workflows |
| `BuildEnvironment` | build metadata to QML | registration ambiguity |

---

## 7) State Management and Reactivity Audit

### State mechanisms currently in use

1. C++ RCU-like global state (`GlobalStore` shared_ptr snapshots).
2. Qt signal/slot notifications.
3. QML local state properties and models.
4. Ad hoc QML XMLHttpRequest side effects.

### Observed issues

- **Potential stale UI risk:** `UiSynchronizer::drainQueue` routes payloads but does not invoke `GlobalStore::commitFrameUpdates()`, while store comments imply this should drive NOTIFY emissions.
- **Duplication:** same concepts updated from multiple planes (ZMQ + C++ REST + QML REST).
- **State divergence risk:** kill switch UI toggles not initialized from authoritative halt mask.
- **Stale closure/async risks:** QML XHR callbacks silently swallow errors; no request cancellation on tab unload.
- **Over-fetch / duplicate polling:** REST polling in C++ + QML Jobs polling + startup broker checks.

### Where frontend can diverge from backend truth

- Static status badges that are not backend-bound (signals/risk/lab).
- Jobs tab stale when request fails silently.
- Kill switch visual state not reconciled with backend on startup.
- Potentially stale `GlobalStore`-bound values if notifier cadence not functioning as intended.

### Cleaner target state architecture

- Single “frontend data runtime” in C++:
  - typed endpoint adapters
  - central cache with timestamps
  - source-of-truth metadata (`source=stream|poll`, `asof`, `freshness`)
- QML becomes mostly declarative view, no business XHR.
- Introduce per-domain view-model objects (OverviewVM, RiskVM, PortfolioVM, etc.) fed from centralized store.
- Explicit stale/error/loading states standardized across panels.

---

## 8) Backend Integration Audit

### Frontend-to-backend surface map

| Frontend surface | Endpoint(s) | Integration style |
|---|---|---|
| Global control state | `/api/control/state` | C++ RestClient polling |
| Portfolio summary/holdings | `/api/control/portfolio-summary` | C++ RestClient polling |
| Broker autoconnect | `/api/control/broker-status`, `/api/control/broker/connect` | QML XHR |
| Jobs | `/api/control/job-graph` | QML XHR polling |
| Promotion action | `/api/control/promote` | QML XHR POST |
| Risk checks (client exists) | `/api/orchestrator/risk-checks` | C++ method exists; not surfaced in QML |
| Streaming telemetry/control/grid | ZMQ 5556/5557 | C++ ZmqReceiver |

### Contract and typing issues

- REST signals in C++ emit raw JSON strings (`std::string`) then parse manually.
- QML XHR parses dynamic JSON without schema validation.
- UI assumes keys exist (`resp.jobs`, `last_status`, etc.) with quiet fallback defaults.
- Endpoint contracts are not reflected as typed frontend DTOs.

### Silent failure modes

- Jobs parsing errors swallowed.
- Broker connect errors mostly text statuses, no structured UI retry diagnostics.
- Promotion POST success path lacks robust UI confirmation and audit display.

### Frontend compensating for backend inconsistency

- fallback values/defaults all over QML indicate defensive handling of missing fields.
- status strings in UI likely compensate for unavailable backend state.

### Contract tightening before redesign

- Define canonical schemas for:
  - control snapshot
  - portfolio summary + holdings rows
  - job graph node
  - sleeve status/health snapshot
  - guardrail state report
- Add `asof_ts`, `source_ts`, and `version` fields to key payloads.
- Introduce typed client adapters (even in C++/Qt context) with strict validation + error surfacing.

---

## 9) Visual Presentation Audit for a Trading App

### Coverage of required operator dimensions

| Dimension | Current coverage | Assessment |
|---|---|---|
| Account health | Header + overview KPIs | partial, no freshness trends |
| Sleeve status | Static-heavy cards | weak reliability |
| Model status | Minimal confidence percentages | weak; lacks drift/latency/backtest context |
| Capital allocation | Sankey + simple table | partial, no drift analytics |
| Execution status | Mode + portfolio table | partial, lacks order/fill pipeline state |
| Risk controls | Risk cards + static breaker list | weak; not clearly live-backed |
| Guardrails | Mentioned in text rows | weak visibility/trust |
| Alerts/failures | Count only, little causality UI | weak |
| Paper vs live separation | mode badge + optional sim watermark | partial; needs stronger boundary cues |
| Historical performance | largely absent in UI | missing |
| Intraday vs EOD distinction | minimal | weak |
| Orchestrator/runtime op status | Jobs tab exists | decent start, but isolated and low-fidelity |

### Better IA for serious operator dashboard

Proposed top-level structure:

1. **Command Center (default)**
   - Global health strip (freshness-aware)
   - Risk sentinel panel (breaches, alerts, halted sleeves)
   - Execution sentinel (orders/fills/rejections latency)
   - Allocation drift + capital utilization
   - “Needs Action Now” queue

2. **Sleeves Workspace**
   - Table of sleeves with status badges (Live/Paper/Halted/Degraded)
   - Clicking sleeve opens detail drawer with model, signals, recent decisions, risk, performance

3. **Risk & Guardrails**
   - Guardrail matrix (threshold/current/trend/action)
   - Alert timeline with root-cause grouping and ack ownership

4. **Execution & Portfolio**
   - Position table + order/fill timeline + slippage analytics
   - clear separation between inventory and execution pipeline

5. **Research/Lab**
   - Promotion pipeline with strong audit trail and cryptographic approval history

6. **Operations**
   - Job DAG status, runtime dependencies, health checks, incident log

---

## 10) Redundancy, Dead Code, Structural Problems

## Unused / weakly-used pieces

- `ExecutionGrid` injected but not used in QML.
- `ArrowBuilderWorker` implemented but no construction/wiring in `main.cpp`.
- `GlobalStore::setArrowWorker` never called.
- `StateReconciler` flush callback/bootstrap integration appears absent.
- `VulkanChartItem` compiled but not represented in views.

## Duplicate patterns

- KPI cards repeated across multiple tabs with near-identical styling and formulas.
- Similar status badges handcrafted in many files rather than a reusable component.

## Inconsistent abstractions

- Some data via C++ services; other data via direct QML XHR.
- Some controls are real backend actions (kill switch), others presentation placeholders.

## Over-engineered areas

- Complex low-level rendering/perf scaffolding compared with current largely static informational UI.

## Under-engineered areas

- Data contract validation, freshness/latency semantics, and operator-grade alerting UX.
- Design system and reusable componentization.

## Misleading naming/separation issues

- Comments/documentation imply robust frame-based diff signaling, but wiring appears incomplete.
- “Settings -> RHI Vulkan” text may not match runtime backend selection.

## Styling inconsistencies

- Hardcoded color and typography values everywhere; no centralized token source.

## Legacy code paths

- Tauri shell with missing web frontend references suggests obsolete/parallel architecture.

## Technical debt hotspots

1. Data-plane duality (ZMQ + C++ REST + QML XHR).
2. Partial wiring of state reconciliation/notification pipeline.
3. Static placeholder content looking operational.
4. Lack of component/design system primitives.

---

## 11) Screen-by-Screen Breakdown

## Overview

- **Purpose now:** high-level account and health summary.
- **Currently shows:** NAV, P&L, positions, health fields, alerts count, correlation.
- **Should show instead:** freshness-aware KPI trend mini-sparklines + incident banner + active risk breaches.
- **Remove:** duplicated low-value labels.
- **Elevate:** “data stale”, “mode risk”, “halted sleeves”, “critical alerts”.
- **Missing interactions:** click-through to details for each KPI/alert.
- **Visual change:** convert static grid cards to prioritized severity lanes.

## Signals

- **Purpose now:** sleeve signal confidence display.
- **Currently shows:** one custom chart + static sleeve cards.
- **Should show instead:** real per-sleeve live signal snapshot table + uncertainty + last model inference time.
- **Remove:** static status text not tied to backend.
- **Elevate:** signal freshness, veto flags, confidence trend.
- **Missing interactions:** symbol/sleeve drilldown.

## Risk

- **Purpose now:** risk metrics + manual kill switch.
- **Currently shows:** correlation, vol regime, alerts count, static breaker list, toggle switches.
- **Should show instead:** true guardrail engine state board with actions and audit log.
- **Remove:** static breaker entries unless data-backed.
- **Elevate:** halt state provenance + auto/manual source + timestamp.
- **Missing interactions:** acknowledge/resolve/override workflows.

## Allocation

- **Purpose now:** meta-allocation view.
- **Currently shows:** static blend metadata, sankey visualization, target/current rows.
- **Should show instead:** target/current/drift with rebalance recommendations and constraints.
- **Remove:** fixed deviation `0.0%` placeholder.
- **Elevate:** drift breaches and expected rebalance impact.

## Portfolio

- **Purpose now:** holdings and aggregate metrics.
- **Currently shows:** NAV/PnL/positions/mode + table.
- **Should show instead:** institutional-grade table (sort/filter/group/export), age badges, per-row risk tags.
- **Remove:** duplicated KPI strip if already in global header (or keep but contextualized).
- **Elevate:** execution pipeline and fills relationship.

## Lab

- **Purpose now:** shadow promotion workflow.
- **Currently shows:** counters, static pipeline steps, FIDO prompt.
- **Should show instead:** promotion queue with verifiable artifacts, model lineage, test evidence.
- **Remove:** placeholder model hash usage.
- **Elevate:** approval audit and rollback safety checks.

## Jobs

- **Purpose now:** orchestrator jobs status.
- **Currently shows:** counts + job rows.
- **Should show instead:** dependency graph health + SLA breaches + retry controls.
- **Remove:** silent-fail behavior.
- **Elevate:** stale/error state and last successful poll age.

## Settings

- **Purpose now:** environment and infra metadata.
- **Currently shows:** endpoints, hardware labels, build details.
- **Should show instead:** verified runtime diagnostics pulled from live runtime introspection.
- **Remove:** hardcoded/assumed infra claims.
- **Elevate:** actionable diagnostics and support bundle export.

---

## 12) Target Redesign Recommendations

## Recommended folder structure (frontend)

```text
native_frontend/src/
  app/
    bootstrap/
    routing/
    shell/
  domain/
    overview/
    sleeves/
    risk/
    allocation/
    execution/
    lab/
    ops/
  components/
    layout/
    panels/
    badges/
    tables/
    charts/
    forms/
  data/
    api/
    stream/
    contracts/
    adapters/
    cache/
  state/
    store/
    selectors/
    viewmodels/
  styling/
    tokens/
    themes/
    typography/
  platform/
    hardware/
    windowing/
```

## Component layering

- **Layer 1:** platform adapters (ZMQ, REST, FIDO, SHM)
- **Layer 2:** typed data contracts + normalization
- **Layer 3:** central state/viewmodels with freshness metadata
- **Layer 4:** reusable UI primitives and domain panels
- **Layer 5:** screen composition

## State/data architecture target

- One normalized store with domain slices.
- Every entity stamped with `source_ts`, `ingest_ts`, `freshness_state`.
- No direct QML XHR; actions dispatched via command services.

## API boundary design

- Typed DTOs and validation for each endpoint.
- Explicit error taxonomy surfaced to UI.
- Contract versioning.

## Design system approach

- Tokenized palette/spacing/typography.
- Severity semantics (info/warn/critical) consistently applied.
- Reusable components: `StatusBadge`, `MetricCard`, `GuardrailRow`, `DataAgeChip`, `PanelHeader`.

## Charting strategy

- Keep custom QSG only where needed for high-frequency rendering.
- Else use standardized chart wrappers with consistent axes/legends/empty states.
- Require source timestamp + sample window in each chart header.

## Status/alert system

- Global incident rail + per-domain alert facets.
- Root-cause grouped timeline with ack and ownership.
- Stale-data alerts first-class.

## Reusable panel system

- Standard panel scaffold:
  - title, subtitle, source badge, refresh age, action menu, body, empty/error/loading states.

## Table strategy

- Virtualized table component with sort/filter/group, pinned columns, row drilldown drawer.

## Drilldown/detail architecture

- Side drawer for sleeve/symbol/order/alert details.
- Preserve context while drilling down.

## Live updates & refresh

- Stream first; poll for reconciliation and low-frequency domains.
- Distinct visual markers for stream-disconnected, poll-failed, stale snapshots.

## Paper/live environment separation

- Persistent environment banner + color accent + explicit mode lock icon.
- Hard guards for live-mutating actions with dual confirmation and hardware signature when required.

---

## 13) Prioritized Action Plan

## Phase 0 — immediate cleanup / no-risk

**Goals**
- Remove obvious stale/placeholder ambiguity.
- Improve trust with freshness labels and error badges.

**Likely files**
- `native_frontend/src/qml/main.qml`
- `native_frontend/src/qml/views/*.qml`

**Why it matters**
- Immediate operator confidence and reduced misread risk.

**Dependencies**
- Minimal; mostly presentation and small data flags.

**Risk**: Low

**Validation**
- UI smoke run with backend disconnected/connected scenarios.
- Confirm stale/error indicators appear.

## Phase 1 — structural refactor

**Goals**
- Introduce reusable panel/components and design tokens.
- Decompose monolithic tab files.

**Likely files/modules**
- `native_frontend/src/qml/components/*` (new)
- existing tab files refactored to compose components

**Why**
- Enables consistent UX and faster future iteration.

**Dependencies**
- Phase 0 semantics for status labels.

**Risk**: Medium

**Validation**
- Visual regression checklist per tab.
- QML performance sanity check.

## Phase 2 — data correctness/state reliability

**Goals**
- Unify data pipeline, remove direct QML XHR business fetches.
- Fix notifier/reconciler wiring gaps.

**Likely files/modules**
- `engine/UiSynchronizer.*`
- `engine/GlobalStore.*`
- `engine/StateReconciler.*`
- `network/RestClient.*`
- tabs that currently fetch directly (`JobsTab.qml`, `main.qml`, `LabTab.qml`)

**Why**
- Prevent stale/misleading operational state.

**Dependencies**
- Light component cleanup from Phase 1.

**Risk**: High (touches core runtime behavior)

**Validation/testing**
- Unit tests for store diff/update behavior.
- Integration tests for simulated ZMQ+REST race and partition recovery.
- Latency/freshness telemetry checks.

## Phase 3 — UX/UI redesign

**Goals**
- Implement target operator IA and dashboard workflows.

**Likely files/modules**
- all qml screens + new domain modules + chart wrappers.

**Why**
- Transform into operator-grade interface.

**Dependencies**
- Stable data architecture from Phase 2.

**Risk**: Medium-High

**Validation**
- Scenario-based usability testing (incident, halt, reconnect, promotion).

## Phase 4 — advanced operator features

**Goals**
- Alert ownership, action journal, drilldown timelines, what-if allocation, multi-workspace role profiles.

**Likely files/modules**
- new ops/risk/sleeves feature modules, backend contract extensions.

**Why**
- High leverage for real operational excellence.

**Dependencies**
- Foundation complete in phases 1-3.

**Risk**: Medium

**Validation**
- End-to-end workflow acceptance tests with replayed market sessions.

---

## 14) Deliverables Format (conformance)

This document includes:

- Narrative architectural explanation ✅
- Frontend inventory ✅
- Route/screen inventory ✅
- Data flow map ✅
- Deficiency list ✅
- Proposed target architecture ✅
- Prioritized implementation plan ✅

---

## A) What ChatGPT should know before redesigning this frontend

1. This is a **Qt/QML native desktop frontend**, not React/Vue web.
2. Data currently comes from a **hybrid of ZMQ stream + REST poll + QML XHR**, and must be unified.
3. Some UI sections are **static placeholders** that currently look live.
4. There are signs of **partial/unfinished plumbing** (store notifier cadence, reconciler bootstrap, arrow worker wiring).
5. Redesign should prioritize **operator trust, freshness visibility, and safety-state clarity** before aesthetics.

## B) Highest-risk frontend weaknesses

1. Potential stale/misleading state due to partial update wiring.
2. Mixed live/static UI semantics that can mislead trading operators.
3. Direct QML networking with silent failures and no standardized error states.
4. Weakly typed API contracts and schema drift risk.
5. Live/paper and guardrail status not consolidated into an unambiguous command center.

## C) Fastest wins with highest visual impact

1. Add panel-level freshness badges and global stale-data warning rail.
2. Replace static “breaker/status” rows with explicit “mock/unwired” badges or real backend data.
3. Introduce consistent severity badges + typography scale.
4. Convert duplicated KPI cards into reusable metric component with trend + last update.
5. Improve jobs and promotion feedback with visible success/error toasts and timestamps.

## D) Best target dashboard structure for this application

**Recommended primary layout:**

- Top: Global safety strip (mode, broker, paused, data freshness, critical alerts)
- Left main: Command Center timeline (alerts/guardrails/execution anomalies)
- Center: Sleeve matrix (status, allocation drift, confidence, risk)
- Right: Action queue (halts, overrides, promotions pending approval)
- Bottom tabs/drawers: positions/execution details, job runtime details, model/lab details

This structure optimizes for the core operator question:

> “What is wrong right now, how severe is it, what is the safest action, and what is the blast radius?”
