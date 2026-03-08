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

        SectionHeader { title: "Operations"; subtitle: "Typed C++ jobs model from /api/control/job-graph" }

        SeverityBanner {
            tone: GlobalStore.backendReachable ? "warn" : "danger"
            message: GlobalStore.dataFreshness === "fresh" ? "" : (GlobalStore.backendReachable ? "Jobs data stale/degraded" : "Jobs data disconnected")
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            MetricCard { Layout.fillWidth: true; height: 85; title: "Scheduled"; value: GlobalStore.jobTotal.toString(); subtitle: "Total jobs"; valueColor: "#7C8CFF" }
            MetricCard { Layout.fillWidth: true; height: 85; title: "Running"; value: GlobalStore.jobRunning.toString(); subtitle: "Current"; valueColor: GlobalStore.jobRunning > 0 ? "#38BDF8" : "#22C55E" }
            MetricCard { Layout.fillWidth: true; height: 85; title: "Failed"; value: GlobalStore.jobFailed.toString(); subtitle: "Needs action"; valueColor: GlobalStore.jobFailed > 0 ? "#EF4444" : "#22C55E" }
        }

        OperatorTable {
            Layout.fillWidth: true
            Layout.fillHeight: true
            title: "Job Runtime"
            subtitle: "Typed C++ model; failures and stale rows are emphasized"
            modelData: JobsModel
            columns: [
                {"title": "Job", "role": "name", "width": 170},
                {"title": "Status", "role": "status", "width": 85},
                {"title": "Last Run", "role": "lastRun", "width": 170},
                {"title": "Dur(s)", "role": "durationSeconds", "width": 60},
                {"title": "Session", "role": "sessions", "width": 110},
                {"title": "Deps", "role": "dependencyCount", "width": 45},
                {"title": "Error", "role": "errorSummary", "width": 220}
            ]
            emptyText: GlobalStore.backendReachable ? "No jobs returned" : "Backend unavailable"
        }
    }
}
