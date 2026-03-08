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

        SectionHeader { title: "Lab"; subtitle: "Model promotion remains blocked until candidate-hash backend contract exists" }

        SeverityBanner { tone: "warn"; message: "Promotion blocked: no authoritative candidate-hash contract is available." }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            MetricCard { Layout.fillWidth: true; height: 84; title: "Shadow Runs"; value: GlobalStore.shadowRuns.toString(); subtitle: "Store backed"; valueColor: "#38BDF8" }
            MetricCard { Layout.fillWidth: true; height: 84; title: "Pending Promotions"; value: GlobalStore.pendingPromotions.toString(); subtitle: "Store backed"; valueColor: "#F59E0B" }
            MetricCard { Layout.fillWidth: true; height: 84; title: "Promoted Models"; value: GlobalStore.promotedModels.toString(); subtitle: "Historical"; valueColor: "#22C55E" }
        }

        PanelScaffold {
            Layout.fillWidth: true
            Layout.fillHeight: true
            title: "Promotion Controls"
            subtitle: "Truthful fail-closed behavior"

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                text: "UNAVAILABLE: candidate model hash and authorization chain are not wired from backend in this build."
                color: "#FBBF24"
            }
            Button {
                text: "Promote Candidate"
                enabled: false
            }
        }
    }
}
