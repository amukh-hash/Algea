#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

checks = []

def expect(cond, msg):
    checks.append((cond, msg))

main_cpp = (ROOT / "native_frontend/src/main.cpp").read_text(encoding="utf-8")
ui_sync = (ROOT / "native_frontend/src/engine/UiSynchronizer.cpp").read_text(encoding="utf-8")
global_store = (ROOT / "native_frontend/src/engine/GlobalStore.h").read_text(encoding="utf-8")
rest_h = (ROOT / "native_frontend/src/network/RestClient.h").read_text(encoding="utf-8")
rest_cpp = (ROOT / "native_frontend/src/network/RestClient.cpp").read_text(encoding="utf-8")
main_qml = (ROOT / "native_frontend/src/qml/main.qml").read_text(encoding="utf-8")
cmd_qml = (ROOT / "native_frontend/src/qml/views/CommandCenterTab.qml").read_text(encoding="utf-8")
jobs_qml = (ROOT / "native_frontend/src/qml/views/JobsTab.qml").read_text(encoding="utf-8")
risk_qml = (ROOT / "native_frontend/src/qml/views/RiskTab.qml").read_text(encoding="utf-8")
lab_qml = (ROOT / "native_frontend/src/qml/views/LabTab.qml").read_text(encoding="utf-8")
qrc = (ROOT / "native_frontend/src/qml/qml.qrc").read_text(encoding="utf-8")

# Runtime truth wiring
expect("receiver.get(), store, reconciler, &app" in main_cpp, "UiSynchronizer is wired with StateReconciler")
expect("m_store->commitFrameUpdates();" in ui_sync, "UiSynchronizer drains call commitFrameUpdates()")
expect("setFlushCallback" in main_cpp and "routePayload" in main_cpp, "StateReconciler flush callback routes to GlobalStore")
expect("reconciler->onBootstrapComplete(doc);" in main_cpp, "Control-state callback triggers reconciler bootstrap completion")

# Parsing resilience and freshness
expect("markDataDegraded" in global_store, "GlobalStore exposes markDataDegraded()")
expect("control_last_update_ms == 0 || s.portfolio_last_update_ms == 0" in global_store, "Freshness uses critical-domain timestamp gating")
expect("jobGraphReceived: missing jobs array" in main_cpp, "Malformed jobs payload is detected and degraded")
expect("brokerStatusReceived: missing connected field" in main_cpp, "Malformed broker payload is detected and degraded")
expect("guardrailStatusReceived: invalid guardrail contract" in main_cpp, "Malformed guardrail payload is detected and degraded")
expect("markBackendDisconnected" in main_cpp, "REST network error path marks backend disconnected")

# New backend bindings
expect("getGuardrailStatus" in rest_h and "/api/control/guardrails/status" in rest_cpp, "RestClient exposes guardrail status endpoint")
expect("ctx->setContextProperty(\"JobsModel\", jobsModel);" in main_cpp, "Typed JobsModel is exposed to QML")

# No critical QML business XHR
for rel, body in {
    "main.qml": main_qml,
    "views/CommandCenterTab.qml": cmd_qml,
    "views/JobsTab.qml": jobs_qml,
    "views/RiskTab.qml": risk_qml,
    "views/LabTab.qml": lab_qml,
}.items():
    expect("XMLHttpRequest" not in body, f"{rel} has no direct business XMLHttpRequest path")

# Critical truth indicators
expect("BACKEND" in main_qml and "BROKER" in main_qml, "Header contains distinct backend and broker badges")
expect("SeverityBanner" in cmd_qml and "degraded/stale" in cmd_qml, "Command Center renders degraded/disconnected warning banner")
expect("modelData: JobsModel" in jobs_qml, "Jobs view binds to typed C++ jobs model")
expect("GlobalStore.guardrails" in risk_qml, "Risk guardrail table is store-backed")
expect("Promotion blocked" in lab_qml, "Lab promotion flow remains explicitly blocked")

# QRC completeness
required_qrc_entries = [
    "views/CommandCenterTab.qml",
    "views/SignalsTab.qml",
    "views/RiskTab.qml",
    "views/PortfolioTab.qml",
    "views/LabTab.qml",
    "views/JobsTab.qml",
    "views/SettingsTab.qml",
    "components/StatusBadge.qml",
    "components/DataAgeChip.qml",
    "components/MetricCard.qml",
    "components/PanelScaffold.qml",
    "components/AlertRail.qml",
    "components/GuardrailMatrix.qml",
    "components/ActionQueuePanel.qml",
    "components/PositionsTable.qml",
    "components/DetailDrawer.qml",
    "components/SleeveStatusRow.qml",
    "components/SectionHeader.qml",
    "components/SeverityBanner.qml",
    "components/OperatorTable.qml",
]
for entry in required_qrc_entries:
    expect(entry in qrc, f"qrc contains {entry}")

for entry in required_qrc_entries:
    qml_file = ROOT / "native_frontend/src/qml" / entry
    expect(qml_file.exists(), f"qrc entry exists on disk: {entry}")

failed = [m for ok, m in checks if not ok]
for ok, msg in checks:
    print(("PASS" if ok else "FAIL") + " - " + msg)

if failed:
    print(f"\n{len(failed)} check(s) failed.")
    sys.exit(1)

print(f"\nAll {len(checks)} runtime-truth checks passed.")
