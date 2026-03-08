import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8
import QtQuick.Window 6.8
import algae.Rendering 1.0
import "components"

ApplicationWindow {
    id: rootWindow
    visible: true
    width: 2560
    height: 1440
    title: isSimulation ? "Algae [SIMULATION]" : "Algae Operator Console"
    color: "#090C14"

    header: Rectangle {
        color: "#101A2A"
        border.color: "#273449"
        border.width: 1
        implicitHeight: 78

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 12
            spacing: 7

            RowLayout {
                Layout.fillWidth: true
                Label { text: "ALGAE OPERATOR CONSOLE"; color: "#E8EDF8"; font.pixelSize: 16; font.bold: true; font.letterSpacing: 1.2 }
                Item { Layout.fillWidth: true }
                Label { text: "Session: " + GlobalStore.currentSession.toUpperCase(); color: "#95A3BF"; font.pixelSize: 11 }
                Label { text: "|"; color: "#3B4A66"; font.pixelSize: 11 }
                Label { text: "Seq " + GlobalStore.lastSequenceId; color: "#7C8CFF"; font.pixelSize: 11 }
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 10

                Rectangle {
                    Layout.preferredHeight: 28
                    Layout.fillWidth: true
                    radius: 6
                    color: "#0D1525"
                    border.color: "#25344F"

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 4
                        spacing: 6
                        StatusBadge { text: GlobalStore.executionMode.toUpperCase(); tone: GlobalStore.executionMode === "ibkr" ? "danger" : "info" }
                        StatusBadge { text: GlobalStore.systemPaused ? "PAUSED" : "RUNNING"; tone: GlobalStore.systemPaused ? "warn" : "ok" }
                        StatusBadge { text: GlobalStore.backendReachable ? "BACKEND UP" : "BACKEND DOWN"; tone: GlobalStore.backendReachable ? "ok" : "danger" }
                        StatusBadge { text: GlobalStore.brokerConnected ? "BROKER UP" : "BROKER DOWN"; tone: GlobalStore.brokerConnected ? "ok" : "danger" }
                        DataAgeChip { freshness: GlobalStore.dataFreshness }
                        Label {
                            visible: GlobalStore.dataLossFlag
                            text: "ZMQ OVERFLOW"
                            color: "#F87171"
                            font.pixelSize: 11
                            font.bold: true
                        }
                    }
                }

                Rectangle {
                    Layout.preferredHeight: 28
                    Layout.preferredWidth: 280
                    radius: 6
                    color: "#0D1525"
                    border.color: "#25344F"

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 8
                        spacing: 8
                        Label { text: "NAV"; color: "#95A3BF"; font.pixelSize: 10 }
                        Label { text: "$" + GlobalStore.totalPortfolioValue.toFixed(2); color: "#E8EDF8"; font.pixelSize: 13; font.bold: true }
                        Rectangle { width: 1; height: 14; color: "#2B3A55" }
                        Label { text: "PnL"; color: "#95A3BF"; font.pixelSize: 10 }
                        Label { text: (GlobalStore.totalPnl >= 0 ? "+$" : "-$") + Math.abs(GlobalStore.totalPnl).toFixed(2); color: GlobalStore.totalPnl >= 0 ? "#22C55E" : "#EF4444"; font.pixelSize: 13; font.bold: true }
                    }
                }
            }
        }
    }

    Rectangle {
        anchors.fill: parent
        anchors.topMargin: header.height
        color: "#090C14"

        RowLayout {
            anchors.fill: parent
            spacing: 0

            Rectangle {
                Layout.preferredWidth: 230
                Layout.fillHeight: true
                color: "#0F1728"
                border.color: "#25324A"
                border.width: 1

                ListView {
                    anchors.fill: parent
                    anchors.margins: 8
                    spacing: 6
                    model: ["Command Center", "Sleeves", "Risk & Guardrails", "Portfolio & Execution", "Lab", "Operations", "Settings / Diagnostics"]
                    delegate: Rectangle {
                        width: ListView.view.width
                        height: 40
                        radius: 6
                        color: index === navTabs.currentIndex ? "#24344E" : "#131E33"
                        border.color: index === navTabs.currentIndex ? "#4B658D" : "#1F2C45"

                        Label {
                            anchors.centerIn: parent
                            text: modelData
                            color: index === navTabs.currentIndex ? "#E8EDF8" : "#9AAAC5"
                            font.pixelSize: 11
                            font.bold: index === navTabs.currentIndex
                        }

                        MouseArea {
                            anchors.fill: parent
                            onClicked: navTabs.currentIndex = index
                        }
                    }
                }
            }

            StackLayout {
                id: navTabs
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: 0

                Loader { active: navTabs.currentIndex === 0; source: "views/CommandCenterTab.qml" }
                Loader { active: navTabs.currentIndex === 1; source: "views/SignalsTab.qml" }
                Loader { active: navTabs.currentIndex === 2; source: "views/RiskTab.qml" }
                Loader { active: navTabs.currentIndex === 3; source: "views/PortfolioTab.qml" }
                Loader { active: navTabs.currentIndex === 4; source: "views/LabTab.qml" }
                Loader { active: navTabs.currentIndex === 5; source: "views/JobsTab.qml" }
                Loader { active: navTabs.currentIndex === 6; source: "views/SettingsTab.qml" }
            }
        }
    }
}
