import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8

Rectangle {
    color: "#0F111A"
    GridLayout {
        anchors.fill: parent; anchors.margins: 16; columns: 3; columnSpacing: 12; rowSpacing: 12
        Rectangle { Layout.fillWidth: true; height: 120; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
            ColumnLayout { anchors.fill: parent; anchors.margins: 16; spacing: 8
                Label { text: "PORTFOLIO VALUE"; color: "#64748B"; font.pixelSize: 11; font.letterSpacing: 1 }
                Label { text: "$" + GlobalStore.totalPortfolioValue.toFixed(2); color: "#6366F1"; font.pixelSize: 28; font.bold: true }
                Label { text: "Total NAV"; color: "#64748B"; font.pixelSize: 10 }
            }
        }
        Rectangle { Layout.fillWidth: true; height: 120; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
            ColumnLayout { anchors.fill: parent; anchors.margins: 16; spacing: 8
                Label { text: "P&L TODAY"; color: "#64748B"; font.pixelSize: 11; font.letterSpacing: 1 }
                Label { text: (GlobalStore.totalPnl >= 0 ? "+$" : "-$") + Math.abs(GlobalStore.totalPnl).toFixed(2); color: GlobalStore.totalPnl >= 0 ? "#22C55E" : "#EF4444"; font.pixelSize: 28; font.bold: true }
                Label { text: "Unrealized + Realized"; color: "#64748B"; font.pixelSize: 10 }
            }
        }
        Rectangle { Layout.fillWidth: true; height: 120; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
            ColumnLayout { anchors.fill: parent; anchors.margins: 16; spacing: 8
                Label { text: "POSITIONS"; color: "#64748B"; font.pixelSize: 11; font.letterSpacing: 1 }
                Label { text: GlobalStore.positionCount.toString(); color: "#06B6D4"; font.pixelSize: 28; font.bold: true }
                Label { text: "Active positions"; color: "#64748B"; font.pixelSize: 10 }
            }
        }
        Rectangle { Layout.fillWidth: true; Layout.fillHeight: true; Layout.columnSpan: 2; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
            ColumnLayout { anchors.fill: parent; anchors.margins: 16; spacing: 12
                Label { text: "SYSTEM HEALTH"; color: "#64748B"; font.pixelSize: 11; font.weight: Font.DemiBold; font.letterSpacing: 2 }
                GridLayout { columns: 2; columnSpacing: 24; rowSpacing: 10; Layout.fillWidth: true
                    Label { text: "Broker"; color: "#94A3B8" } Row { spacing: 6; Rectangle { width: 8; height: 8; radius: 4; color: GlobalStore.brokerConnected ? "#22C55E" : "#EF4444"; anchors.verticalCenter: parent.verticalCenter } Label { text: GlobalStore.brokerConnected ? "Connected" : "Disconnected"; color: GlobalStore.brokerConnected ? "#22C55E" : "#EF4444" } }
                    Label { text: "Session"; color: "#94A3B8" } Label { text: GlobalStore.currentSession; color: "#06B6D4"; font.bold: true }
                    Label { text: "Mode"; color: "#94A3B8" } Label { text: GlobalStore.executionMode; color: GlobalStore.executionMode === "ibkr" ? "#EF4444" : "#F59E0B"; font.bold: true }
                    Label { text: "Paused"; color: "#94A3B8" } Label { text: GlobalStore.systemPaused ? "YES" : "No"; color: GlobalStore.systemPaused ? "#F59E0B" : "#22C55E" }
                    Label { text: "Data Loss"; color: "#94A3B8" } Label { text: GlobalStore.dataLossFlag ? "ACTIVE" : "None"; color: GlobalStore.dataLossFlag ? "#EF4444" : "#22C55E" }
                    Label { text: "Sequence"; color: "#94A3B8" } Label { text: GlobalStore.lastSequenceId.toString(); color: "#E2E8F0"; font.family: "Consolas" }
                    Label { text: "Vol Regime"; color: "#94A3B8" } Label { text: GlobalStore.volRegimeOverride || "Normal"; color: "#E2E8F0" }
                }
                Item { Layout.fillHeight: true }
            }
        }
        Rectangle { Layout.fillWidth: true; Layout.fillHeight: true; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
            ColumnLayout { anchors.fill: parent; anchors.margins: 16; spacing: 12
                Label { text: "ALERTS"; color: "#64748B"; font.pixelSize: 11; font.weight: Font.DemiBold; font.letterSpacing: 2 }
                Rectangle { Layout.fillWidth: true; height: 80; radius: 6; color: "#1A1D2E"
                    ColumnLayout { anchors.centerIn: parent; spacing: 4
                        Label { text: AlertEngine.activeAlertCount.toString(); color: AlertEngine.activeAlertCount > 0 ? "#F59E0B" : "#22C55E"; font.pixelSize: 36; font.bold: true; Layout.alignment: Qt.AlignHCenter }
                        Label { text: "Active Alerts"; color: "#64748B"; font.pixelSize: 10; Layout.alignment: Qt.AlignHCenter }
                    }
                }
                Rectangle { Layout.fillWidth: true; height: 60; radius: 6; color: GlobalStore.statArbCorrelation > 2.0 ? "#4A0000" : "#1A1D2E"
                    ColumnLayout { anchors.centerIn: parent; spacing: 4
                        Label { text: GlobalStore.statArbCorrelation.toFixed(4); color: GlobalStore.statArbCorrelation > 2.0 ? "#EF4444" : "#E2E8F0"; font.pixelSize: 24; font.bold: true; font.family: "Consolas"; Layout.alignment: Qt.AlignHCenter }
                        Label { text: "StatArb Corr"; color: "#64748B"; font.pixelSize: 10; Layout.alignment: Qt.AlignHCenter }
                    }
                }
                Item { Layout.fillHeight: true }
            }
        }
    }
}
