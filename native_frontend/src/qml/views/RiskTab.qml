import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8
import "../components"

Rectangle {
    color: "#090C14"

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 14
        spacing: 10

        SectionHeader { title: "Risk & Guardrails"; subtitle: "Live bindings only. Unknown/unwired states are explicit." }

        SeverityBanner {
            tone: GlobalStore.backendReachable ? "warn" : "danger"
            message: GlobalStore.dataFreshness === "fresh" ? "" : (GlobalStore.backendReachable ? "Risk data stale/degraded" : "Risk data disconnected")
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            MetricCard {
                Layout.fillWidth: true
                height: 88
                title: "StatArb Correlation"
                value: GlobalStore.statArbCorrelation.toFixed(4)
                subtitle: "Threshold <= 2.0"
                valueColor: GlobalStore.statArbCorrelation > 2.0 ? "#EF4444" : "#22C55E"
            }
            MetricCard {
                Layout.fillWidth: true
                height: 88
                title: "Vol Regime"
                value: GlobalStore.volRegimeOverride.length > 0 ? GlobalStore.volRegimeOverride : "UNAVAILABLE"
                subtitle: GlobalStore.volRegimeOverride.length > 0 ? "Control state" : "No backend vol-regime feed"
                valueColor: GlobalStore.volRegimeOverride.length > 0 ? "#E8EDF8" : "#F59E0B"
            }
            MetricCard {
                Layout.fillWidth: true
                height: 88
                title: "Active Alerts"
                value: AlertEngine.activeAlertCount.toString()
                subtitle: "Alert DAG"
                valueColor: AlertEngine.activeAlertCount > 0 ? "#EF4444" : "#22C55E"
            }
        }

        OperatorTable {
            Layout.fillWidth: true
            Layout.fillHeight: true
            title: "Guardrail Matrix"
            subtitle: "Source: /api/control/guardrails/status (breached/unknown rows emphasized)"
            modelData: ListModel {
                id: guardrailModel
            }
            columns: [
                {"title": "Guardrail", "role": "label", "width": 180},
                {"title": "Status", "role": "status", "width": 95},
                {"title": "Value", "role": "valueText", "width": 90},
                {"title": "Threshold", "role": "thresholdText", "width": 90},
                {"title": "Reason", "role": "reasonText", "width": 140},
                {"title": "Source", "role": "source_job", "width": 130}
            ]
            emptyText: "Guardrail contract unavailable"
        }

        PanelScaffold {
            Layout.fillWidth: true
            Layout.preferredHeight: 200
            title: "Kill Switch"
            subtitle: "Authoritative shared-memory state"

            Repeater {
                model: ["core", "selector", "vrp"]
                Rectangle {
                    Layout.fillWidth: true
                    height: 36
                    radius: 5
                    color: "#121C2F"
                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 8
                        Label { text: modelData.toUpperCase(); color: "#E2E8F0"; Layout.preferredWidth: 90 }
                        StatusBadge {
                            text: KillSwitch.isSleeveHalted(modelData) ? "HALTED" : "RUNNING"
                            tone: KillSwitch.isSleeveHalted(modelData) ? "danger" : "ok"
                        }
                        Item { Layout.fillWidth: true }
                        Button {
                            text: KillSwitch.isSleeveHalted(modelData) ? "Resume" : "Halt"
                            onClicked: KillSwitch.toggleSleeve(modelData, !KillSwitch.isSleeveHalted(modelData))
                        }
                    }
                }
            }
        }
    }

    function formatNumber(v) {
        if (v === undefined || v === null || v === "") return "—";
        const n = Number(v);
        if (isNaN(n)) return "—";
        return n.toFixed(4);
    }

    function reloadGuardrails() {
        guardrailModel.clear();
        const rows = GlobalStore.guardrails || [];
        for (let i = 0; i < rows.length; ++i) {
            const r = rows[i];
            guardrailModel.append({
                "label": r.label || r.id || "unknown",
                "status": (r.status || "unknown").toUpperCase(),
                "valueText": formatNumber(r.value),
                "thresholdText": formatNumber(r.threshold),
                "reasonText": r.reason ? String(r.reason) : "—",
                "source_job": r.source_job || "—"
            });
        }
    }

    Component.onCompleted: reloadGuardrails()
    Connections {
        target: GlobalStore
        function onHealthStatusChanged() { reloadGuardrails(); }
    }
}
