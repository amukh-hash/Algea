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

        SectionHeader { title: "Settings / Diagnostics"; subtitle: "Runtime introspection and environment metadata" }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            MetricCard { Layout.fillWidth: true; height: 84; title: "Backend"; value: GlobalStore.backendReachable ? "UP" : "DOWN"; subtitle: "REST connectivity"; valueColor: GlobalStore.backendReachable ? "#22C55E" : "#EF4444" }
            MetricCard { Layout.fillWidth: true; height: 84; title: "Broker"; value: GlobalStore.brokerConnected ? "UP" : "DOWN"; subtitle: "Execution gateway"; valueColor: GlobalStore.brokerConnected ? "#22C55E" : "#EF4444" }
            MetricCard { Layout.fillWidth: true; height: 84; title: "Freshness"; value: GlobalStore.dataFreshness.toUpperCase(); subtitle: "Control + portfolio timestamps"; valueColor: GlobalStore.dataFreshness === "fresh" ? "#22C55E" : "#F59E0B" }
        }

        PanelScaffold {
            Layout.fillWidth: true
            Layout.fillHeight: true
            title: "Build & Runtime"
            subtitle: "Binary identity and compile metadata"

            GridLayout {
                Layout.fillWidth: true
                columns: 2
                rowSpacing: 8
                columnSpacing: 14

                Label { text: "Build"; color: "#94A3B8" }
                Label { text: BuildEnvironment.buildVersion; color: "#E8EDF8" }
                Label { text: "Compiler"; color: "#94A3B8" }
                Label { text: BuildEnvironment.compilerId + " " + BuildEnvironment.cxxStandard; color: "#E8EDF8" }
                Label { text: "Arrow"; color: "#94A3B8" }
                Label { text: BuildEnvironment.arrowVersion; color: "#E8EDF8" }
                Label { text: "Protobuf"; color: "#94A3B8" }
                Label { text: BuildEnvironment.protobufVersion; color: "#E8EDF8" }
                Label { text: "FIDO"; color: "#94A3B8" }
                Label { text: FidoGateway.available ? FidoGateway.deviceName : "Unavailable"; color: FidoGateway.available ? "#22C55E" : "#F59E0B" }
            }
        }
    }
}
