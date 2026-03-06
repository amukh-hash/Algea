import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8

Rectangle {
    color: "#0F111A"
    ColumnLayout {
        anchors.fill: parent; anchors.margins: 16; spacing: 12
        Label { text: "PORTFOLIO & EXECUTION"; color: "#64748B"; font.pixelSize: 13; font.weight: Font.DemiBold; font.letterSpacing: 2 }
        RowLayout { Layout.fillWidth: true; spacing: 12
            Rectangle { Layout.fillWidth: true; height: 90; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
                ColumnLayout { anchors.fill: parent; anchors.margins: 12; spacing: 6; Label { text: "NAV"; color: "#64748B"; font.pixelSize: 9 } Label { text: "$" + GlobalStore.totalPortfolioValue.toFixed(2); color: "#6366F1"; font.pixelSize: 22; font.bold: true } } }
            Rectangle { Layout.fillWidth: true; height: 90; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
                ColumnLayout { anchors.fill: parent; anchors.margins: 12; spacing: 6; Label { text: "DAY P&L"; color: "#64748B"; font.pixelSize: 9 } Label { text: (GlobalStore.totalPnl >= 0 ? "+" : "") + "$" + GlobalStore.totalPnl.toFixed(2); color: GlobalStore.totalPnl >= 0 ? "#22C55E" : "#EF4444"; font.pixelSize: 22; font.bold: true } } }
            Rectangle { Layout.fillWidth: true; height: 90; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
                ColumnLayout { anchors.fill: parent; anchors.margins: 12; spacing: 6; Label { text: "POSITIONS"; color: "#64748B"; font.pixelSize: 9 } Label { text: GlobalStore.positionCount.toString(); color: "#06B6D4"; font.pixelSize: 22; font.bold: true } } }
            Rectangle { Layout.fillWidth: true; height: 90; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
                ColumnLayout { anchors.fill: parent; anchors.margins: 12; spacing: 6; Label { text: "MODE"; color: "#64748B"; font.pixelSize: 9 } Label { text: GlobalStore.executionMode.toUpperCase(); color: GlobalStore.executionMode === "ibkr" ? "#EF4444" : "#F59E0B"; font.pixelSize: 22; font.bold: true } } }
        }
        // Arrow IPC Positions Grid
        Rectangle { Layout.fillWidth: true; Layout.fillHeight: true; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
            ColumnLayout { anchors.fill: parent; anchors.margins: 16; spacing: 8
                RowLayout { Layout.fillWidth: true
                    Label { text: "POSITIONS GRID — ARROW IPC"; color: "#64748B"; font.pixelSize: 11; font.weight: Font.DemiBold; font.letterSpacing: 2 }
                    Item { Layout.fillWidth: true }
                    Label { text: "Rows: " + PositionsGrid.currentRowCount; color: "#64748B"; font.pixelSize: 10; font.family: "Consolas" }
                }

                TableView {
                    Layout.fillWidth: true; Layout.fillHeight: true
                    model: PositionsGrid
                    clip: true
                    columnWidthProvider: function (column) { return [100, 70, 60, 80, 100, 100, 120, 140][column] || 100 }

                    delegate: Rectangle {
                        implicitHeight: 32; color: row % 2 === 0 ? "transparent" : "#1A1D2E"
                        Label { anchors.fill: parent; anchors.margins: 4; text: display || ""; color: "#E2E8F0"; font.pixelSize: 11; font.family: "Consolas"; verticalAlignment: Text.AlignVCenter }
                    }
                }
            }
        }
    }
}
