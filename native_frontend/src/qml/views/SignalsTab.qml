import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8
import algae.Rendering 1.0

Rectangle {
    color: "#0F111A"
    ColumnLayout {
        anchors.fill: parent; anchors.margins: 16; spacing: 12
        Label { text: "SIGNAL GENERATION"; color: "#64748B"; font.pixelSize: 13; font.weight: Font.DemiBold; font.letterSpacing: 2 }
        RowLayout { Layout.fillWidth: true; Layout.fillHeight: true; spacing: 12

            // Kronos — C++ FanChartItem (Vulkan QSGGeometryNode)
            Rectangle { Layout.fillWidth: true; Layout.fillHeight: true; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
                ColumnLayout { anchors.fill: parent; anchors.margins: 16; spacing: 8
                    RowLayout { Label { text: "Kronos — PatchTST Forecaster"; color: "#E2E8F0"; font.pixelSize: 14; font.bold: true } Item { Layout.fillWidth: true } Rectangle { radius: 3; color: "#14532D"; implicitWidth: 60; implicitHeight: 20; Label { anchors.centerIn: parent; text: "Active"; color: "#22C55E"; font.pixelSize: 9; font.bold: true } } }
                    Label { text: "Quantile Fan Chart — P10/P25/P50/P75/P90 uncertainty bands"; color: "#6366F1"; font.pixelSize: 10 }

                    FanChartItem {
                        Layout.fillWidth: true; Layout.fillHeight: true
                    }

                    RowLayout { spacing: 16
                        Column { spacing: 2; Label { text: "TARGET WT"; color: "#64748B"; font.pixelSize: 9 } Label { text: (GlobalStore.kronosWeight * 100).toFixed(0) + "%"; color: "#E2E8F0"; font.pixelSize: 14; font.family: "Consolas" } }
                        Column { spacing: 2; Label { text: "CONFIDENCE"; color: "#64748B"; font.pixelSize: 9 } Label { text: (GlobalStore.kronosConfidence * 100).toFixed(1) + "%"; color: "#E2E8F0"; font.pixelSize: 14; font.family: "Consolas" } }
                        Column { spacing: 2; Label { text: "HORIZON"; color: "#64748B"; font.pixelSize: 9 } Label { text: "5m bars"; color: "#64748B"; font.pixelSize: 11 } }
                    }
                }
            }

            // Right column — remaining sleeves
            ColumnLayout { Layout.preferredWidth: 400; Layout.fillHeight: true; spacing: 8
                Repeater { model: ["CO→OC Reversal|Statistical Mean-Reversion|Active", "StatArb Pairs|Cointegration Pairs|Monitoring", "TFT Forecaster|Temporal Fusion Transformer|Training"]
                    Rectangle { Layout.fillWidth: true; Layout.fillHeight: true; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
                        ColumnLayout { anchors.fill: parent; anchors.margins: 12; spacing: 6
                            RowLayout { Label { text: modelData.split("|")[0]; color: "#E2E8F0"; font.pixelSize: 13; font.bold: true } Item { Layout.fillWidth: true } Rectangle { radius: 3; color: modelData.split("|")[2] === "Active" ? "#14532D" : modelData.split("|")[2] === "Training" ? "#78350F" : "#222538"; implicitWidth: 70; implicitHeight: 18; Label { anchors.centerIn: parent; text: modelData.split("|")[2]; color: modelData.split("|")[2] === "Active" ? "#22C55E" : modelData.split("|")[2] === "Training" ? "#F59E0B" : "#64748B"; font.pixelSize: 8; font.bold: true } } }
                            Label { text: modelData.split("|")[1]; color: "#6366F1"; font.pixelSize: 10 }
                            RowLayout { spacing: 12
                                Column { spacing: 2; Label { text: "WT"; color: "#64748B"; font.pixelSize: 8 } Label { text: index === 0 ? (GlobalStore.coocWeight * 100).toFixed(0) + "%" : index === 1 ? (GlobalStore.statarbWeight * 100).toFixed(0) + "%" : (GlobalStore.tftWeight * 100).toFixed(0) + "%"; color: "#E2E8F0"; font.pixelSize: 12; font.family: "Consolas" } }
                                Column { spacing: 2; Label { text: "CONF"; color: "#64748B"; font.pixelSize: 8 } Label { text: index === 0 ? (GlobalStore.coocConfidence * 100).toFixed(1) + "%" : index === 1 ? (GlobalStore.statarbConfidence * 100).toFixed(1) + "%" : (GlobalStore.tftConfidence * 100).toFixed(1) + "%"; color: "#E2E8F0"; font.pixelSize: 12; font.family: "Consolas" } }
                            }
                        }
                    }
                }
            }
        }
    }
}
