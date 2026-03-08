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

        SectionHeader { title: "Command Center"; subtitle: "Operator home: runtime truth, urgency, and action" }

        SeverityBanner {
            tone: GlobalStore.backendReachable ? "warn" : "danger"
            message: GlobalStore.dataFreshness === "fresh"
                     ? ""
                     : (GlobalStore.backendReachable ? "Live data degraded/stale" : "Backend disconnected; showing last known values")
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            MetricCard { Layout.fillWidth: true; height: 92; title: "Active Alerts"; value: AlertEngine.activeAlertCount.toString(); subtitle: AlertEngine.activeAlertCount > 0 ? "Needs immediate review" : "No active incidents"; valueColor: AlertEngine.activeAlertCount > 0 ? "#EF4444" : "#22C55E" }
            MetricCard { Layout.fillWidth: true; height: 92; title: "Jobs Failed"; value: GlobalStore.jobFailed.toString(); subtitle: GlobalStore.jobFailed > 0 ? "Action queue has blockers" : "No current blockers"; valueColor: GlobalStore.jobFailed > 0 ? "#EF4444" : "#22C55E" }
            MetricCard { Layout.fillWidth: true; height: 92; title: "Runtime"; value: GlobalStore.systemPaused ? "PAUSED" : "RUNNING"; subtitle: GlobalStore.systemPaused ? "Execution halted by control" : "Execution enabled"; valueColor: GlobalStore.systemPaused ? "#F59E0B" : "#22C55E" }
            MetricCard { Layout.fillWidth: true; height: 92; title: "Mode"; value: GlobalStore.executionMode.toUpperCase(); subtitle: "Paper vs live venue"; valueColor: GlobalStore.executionMode === "ibkr" ? "#F97316" : "#7C8CFF" }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 8

            AlertRail {
                Layout.fillWidth: true
                Layout.fillHeight: true
                activeAlerts: AlertEngine.activeAlertCount
            }

            ActionQueuePanel {
                Layout.fillWidth: true
                Layout.fillHeight: true
                failedJobs: GlobalStore.jobFailed
                paused: GlobalStore.systemPaused
            }

            GuardrailMatrix {
                Layout.fillWidth: true
                Layout.fillHeight: true
            }
        }
    }
}
