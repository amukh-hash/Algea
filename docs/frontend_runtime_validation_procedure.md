# Frontend Runtime Truth Validation Procedure (Qt-capable environment)

This is the narrow smoke procedure for closing Stage 0/1 runtime-proof criteria.

## Prerequisites
- Qt 6.8+ SDK installed and discoverable by CMake (`Qt6Config.cmake`)
- Native frontend build dependencies available (same as `native_frontend/CMakeLists.txt`)
- Python 3.10+

## 1) Build native frontend

```bash
cmake -S native_frontend -B build/native_frontend
cmake --build build/native_frontend -j
```

Expected: build succeeds and executable is produced (`Algae_Live` or `Algae_Sim`).

## 2) Static runtime-truth regression check

```bash
python scripts/frontend_runtime_truth_check.py
```

Expected: all checks pass.

## 3) Runtime smoke: backend unavailable startup

1. Ensure no backend service is running on configured `Algae_BACKEND_URL`.
2. Launch frontend executable.
3. Verify:
   - app starts without crash,
   - header shows `BACKEND DOWN`,
   - freshness chip shows `disconnected`.

## 4) Runtime smoke: reconnect to healthy backend

1. Start backend.
2. Wait for poll cycle (or initial startup poll).
3. Verify:
   - header `BACKEND UP` appears,
   - freshness transitions to `fresh`.

## 5) Runtime smoke: stale/degraded transitions

1. Stop backend after app reached `fresh`.
2. Wait >35s and <90s: verify `stale`.
3. Wait >90s: verify `degraded` or `disconnected` depending on network error timing.

## 6) Malformed payload validation (job-graph)

Use a local mock backend that returns malformed jobs payload for `/api/control/job-graph`:

```json
{"jobs": "not-an-array"}
```

Verify:
- app does not crash,
- operations summary warns stale/degraded,
- logs include `jobGraphReceived: missing jobs array`.

## 7) Malformed payload validation (broker-status)

Return malformed broker payload without `connected` field:

```json
{"status": "ok"}
```

Verify:
- app does not crash,
- header still renders safely,
- logs include `brokerStatusReceived: missing connected field`.

## 8) QRC packaged load validation

Run frontend from built artifact location (not source tree).
Verify all tabs open without missing-qrc errors.

## 9) Evidence capture checklist

Capture for review:
- app startup log excerpt (backend unavailable)
- app startup log excerpt (backend healthy)
- screenshot of header disconnected state
- screenshot of header fresh state
- screenshot of command center degraded/disconnected banner
- screenshot of operations stale/degraded warning
- malformed payload log lines for jobs/broker checks



## Added runtime checks for guardrails + typed jobs

When running on a Qt-capable machine, also validate:

1. `GET /api/control/guardrails/status` returns schema `guardrails_status.v1` and exactly 5 guardrail rows.
2. Risk screen table displays all five rows and shows `UNKNOWN/UNWIRED` rather than fake healthy values when metrics are missing.
3. Operations table renders row-level jobs from `JobsModel` and still shows stale/degraded banners when freshness drops.
4. Malformed guardrail payload causes degraded state (warning log + freshness degradation) rather than silent failure.
