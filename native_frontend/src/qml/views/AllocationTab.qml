import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8
import algae.Rendering 1.0

Rectangle {
    color: "#0F111A"
    ColumnLayout {
        anchors.fill: parent; anchors.margins: 16; spacing: 12
        Label { text: "META-ALLOCATION"; color: "#64748B"; font.pixelSize: 13; font.weight: Font.DemiBold; font.letterSpacing: 2 }
        Rectangle { Layout.fillWidth: true; height: 70; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
            RowLayout { anchors.fill: parent; anchors.margins: 16; spacing: 24
                Column { spacing: 4; Label { text: "BLEND MODE"; color: "#64748B"; font.pixelSize: 9 } Label { text: "Risk-Parity Weighted"; color: "#6366F1"; font.pixelSize: 14; font.bold: true } }
                Column { spacing: 4; Label { text: "REBALANCE"; color: "#64748B"; font.pixelSize: 9 } Label { text: "Daily @ 09:20 EST"; color: "#E2E8F0"; font.pixelSize: 14 } }
                Column { spacing: 4; Label { text: "TOTAL LEVERAGE"; color: "#64748B"; font.pixelSize: 9 } Label { text: GlobalStore.totalLeverage.toFixed(2) + "x"; color: "#E2E8F0"; font.pixelSize: 14; font.family: "Consolas" } }
                Item { Layout.fillWidth: true }
            }
        }

        // C++ Sankey Diagram
        Rectangle { Layout.fillWidth: true; Layout.fillHeight: true; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
            ColumnLayout { anchors.fill: parent; anchors.margins: 16; spacing: 8
                Label { text: "CAPITAL FLOW — SANKEY DIAGRAM"; color: "#64748B"; font.pixelSize: 11; font.weight: Font.DemiBold; font.letterSpacing: 2 }
                Label { text: "Cubic Bézier triangle strip mesh — flow thickness ∝ capital weight"; color: "#64748B"; font.pixelSize: 9 }
                SankeyDiagramItem {
                    Layout.fillWidth: true; Layout.fillHeight: true
                }
            }
        }

        // Weight table
        Rectangle { Layout.fillWidth: true; height: 160; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
            ColumnLayout { anchors.fill: parent; anchors.margins: 12; spacing: 6
                RowLayout { Layout.fillWidth: true
                    Label { text: "Sleeve"; color: "#64748B"; font.pixelSize: 10; font.bold: true; Layout.preferredWidth: 200 }
                    Label { text: "Target"; color: "#64748B"; font.pixelSize: 10; font.bold: true; Layout.preferredWidth: 100 }
                    Label { text: "Current"; color: "#64748B"; font.pixelSize: 10; font.bold: true; Layout.preferredWidth: 100 }
                    Label { text: "Deviation"; color: "#64748B"; font.pixelSize: 10; font.bold: true; Layout.preferredWidth: 100 }
                    Item { Layout.fillWidth: true }
                }
                Rectangle { Layout.fillWidth: true; height: 1; color: "#2A2D3E" }
                Repeater { model: ["Kronos (PatchTST)", "CO→OC Reversal", "StatArb Pairs", "TFT Forecaster"]
                    Rectangle { Layout.fillWidth: true; height: 28; radius: 3; color: index % 2 === 0 ? "transparent" : "#1A1D2E"
                        RowLayout { anchors.fill: parent; spacing: 4
                            Label { text: modelData; color: "#E2E8F0"; font.pixelSize: 11; Layout.preferredWidth: 200 }
                            Label { text: (index === 0 ? GlobalStore.kronosWeight : index === 1 ? GlobalStore.coocWeight : index === 2 ? GlobalStore.statarbWeight : GlobalStore.tftWeight) * 100 + "%"; color: "#6366F1"; font.pixelSize: 11; font.family: "Consolas"; Layout.preferredWidth: 100 }
                            Label { text: (index === 0 ? GlobalStore.kronosWeight : index === 1 ? GlobalStore.coocWeight : index === 2 ? GlobalStore.statarbWeight : GlobalStore.tftWeight) * 100 + "%"; color: "#E2E8F0"; font.pixelSize: 11; font.family: "Consolas"; Layout.preferredWidth: 100 }
                            Label { text: "0.0%"; color: "#94A3B8"; font.pixelSize: 11; font.family: "Consolas"; Layout.preferredWidth: 100 }
                            Item { Layout.fillWidth: true }
                        }
                    }
                }
            }
        }
    }
}
