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

        SectionHeader { title: "Sleeves"; subtitle: "Sleeve wording distinguishes live metrics, partial feeds, and unwired state" }

        SeverityBanner {
            tone: GlobalStore.backendReachable ? "warn" : "danger"
            message: GlobalStore.dataFreshness === "fresh" ? "" : "Sleeve metrics may be stale/degraded"
        }

        PanelScaffold {
            Layout.fillWidth: true
            Layout.fillHeight: true
            title: "Sleeve Runtime Summary"
            subtitle: "Weights/confidence are live-backed; execution/health remains explicit when unavailable"

            SleeveStatusRow {
                Layout.fillWidth: true
                sleeveName: "Core"
                status: GlobalStore.dataFreshness === "fresh" ? "WEIGHT/CONF" : "DEGRADED"
                note: "Live: weight/confidence only | Weight " + GlobalStore.coocWeight.toFixed(2) + " | Confidence " + GlobalStore.coocConfidence.toFixed(2)
            }
            SleeveStatusRow {
                Layout.fillWidth: true
                sleeveName: "Selector"
                status: GlobalStore.dataFreshness === "fresh" ? "WEIGHT/CONF" : "DEGRADED"
                note: "Live: weight/confidence only | Weight " + GlobalStore.statarbWeight.toFixed(2) + " | Confidence " + GlobalStore.statarbConfidence.toFixed(2)
            }
            SleeveStatusRow {
                Layout.fillWidth: true
                sleeveName: "VRP"
                status: "UNWIRED"
                note: "No live per-sleeve status/health contract in current payload"
            }
            SleeveStatusRow {
                Layout.fillWidth: true
                sleeveName: "Kronos"
                status: GlobalStore.dataFreshness === "fresh" ? "WEIGHT/CONF" : "DEGRADED"
                note: "Live: weight/confidence only | Weight " + GlobalStore.kronosWeight.toFixed(2) + " | Confidence " + GlobalStore.kronosConfidence.toFixed(2)
            }
        }
    }
}
