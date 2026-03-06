import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8

Rectangle {
    color: GlobalStore.statArbCorrelation > 2.0 ? "#200000" : "#0F111A"
    Behavior on color { ColorAnimation { duration: 300 } }
    ColumnLayout {
        anchors.fill: parent; anchors.margins: 16; spacing: 12
        Label { text: "RISK & SAFEGUARDS"; color: "#64748B"; font.pixelSize: 13; font.weight: Font.DemiBold; font.letterSpacing: 2 }
        RowLayout { Layout.fillWidth: true; spacing: 12
            Rectangle { Layout.fillWidth: true; height: 100; radius: 8; color: GlobalStore.statArbCorrelation > 2.0 ? "#3B0000" : "#141622"; border.color: GlobalStore.statArbCorrelation > 2.0 ? "#EF4444" : "#2A2D3E"; border.width: 1
                ColumnLayout { anchors.fill: parent; anchors.margins: 12; spacing: 6
                    Label { text: "STATARB CORRELATION"; color: "#64748B"; font.pixelSize: 10; font.letterSpacing: 1 }
                    Label { text: GlobalStore.statArbCorrelation.toFixed(4); color: GlobalStore.statArbCorrelation > 2.0 ? "#EF4444" : "#E2E8F0"; font.pixelSize: 22; font.bold: true; font.family: "Consolas" }
                    Label { text: "Threshold: < 2.0"; color: "#64748B"; font.pixelSize: 9 }
                }
            }
            Rectangle { Layout.fillWidth: true; height: 100; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
                ColumnLayout { anchors.fill: parent; anchors.margins: 12; spacing: 6
                    Label { text: "VOL REGIME"; color: "#64748B"; font.pixelSize: 10; font.letterSpacing: 1 }
                    Label { text: GlobalStore.volRegimeOverride || "Normal"; color: "#E2E8F0"; font.pixelSize: 22; font.bold: true }
                    Label { text: "Auto-detected"; color: "#64748B"; font.pixelSize: 9 }
                }
            }
            Rectangle { Layout.fillWidth: true; height: 100; radius: 8; color: AlertEngine.activeAlertCount > 0 ? "#3B0000" : "#141622"; border.color: AlertEngine.activeAlertCount > 0 ? "#EF4444" : "#2A2D3E"; border.width: 1
                ColumnLayout { anchors.fill: parent; anchors.margins: 12; spacing: 6
                    Label { text: "ACTIVE ALERTS"; color: "#64748B"; font.pixelSize: 10; font.letterSpacing: 1 }
                    Label { text: AlertEngine.activeAlertCount.toString(); color: AlertEngine.activeAlertCount > 0 ? "#EF4444" : "#22C55E"; font.pixelSize: 22; font.bold: true }
                    Label { text: "Threshold: 0"; color: "#64748B"; font.pixelSize: 9 }
                }
            }
        }
        // OOB Kill Switch — Per-Sleeve Interactive Toggles
        Rectangle { Layout.fillWidth: true; height: 200; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
            ColumnLayout { anchors.fill: parent; anchors.margins: 16; spacing: 8
                Label { text: "OOB KILL SWITCH — SHARED MEMORY CONTROL PLANE"; color: "#EF4444"; font.pixelSize: 11; font.weight: Font.DemiBold; font.letterSpacing: 2 }
                Label { text: "Writes atomically to AlgaeControlPlane SHM. Bypasses TCP/REST — sub-microsecond latency."; color: "#64748B"; font.pixelSize: 10 }
                Repeater { model: ["0|Kronos (PatchTST)|kronos", "1|CO→OC Reversal|cooc", "2|StatArb Pairs|statarb", "3|TFT Forecaster|tft"]
                    Rectangle { Layout.fillWidth: true; height: 36; radius: 4; color: "#1A1D2E"
                        RowLayout { anchors.fill: parent; anchors.margins: 8; spacing: 12
                            Rectangle { width: 8; height: 8; radius: 4; color: killSw.checked ? "#22C55E" : "#EF4444" }
                            Label { text: modelData.split("|")[1]; color: "#E2E8F0"; font.pixelSize: 12; Layout.fillWidth: true }
                            Label { text: killSw.checked ? "ARMED" : "HALTED"; color: killSw.checked ? "#22C55E" : "#EF4444"; font.pixelSize: 9; font.bold: true }
                            Switch { id: killSw; checked: true; palette.highlight: "#22C55E" }
                            MouseArea {
                                anchors.fill: killSw
                                onClicked: {
                                    var sleeveId = parseInt(modelData.split("|")[0])
                                    if (killSw.checked) {
                                        KillSwitch.haltSleeve(sleeveId, "Manual PM Override via Tab 3")
                                        killSw.checked = false
                                    } else {
                                        KillSwitch.resumeSleeve(sleeveId)
                                        killSw.checked = true
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Circuit Breakers
        Rectangle { Layout.fillWidth: true; Layout.fillHeight: true; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
            ColumnLayout { anchors.fill: parent; anchors.margins: 16; spacing: 8
                Label { text: "CIRCUIT BREAKERS"; color: "#64748B"; font.pixelSize: 11; font.weight: Font.DemiBold; font.letterSpacing: 2 }
                Repeater { model: ["ECE Tracker — Expected Calibration Error halt|Armed", "MMD LiveGuard — Covariate shift detector|Armed", "Max Drawdown — 2% intraday limit|Armed", "Gap Risk Filter — Overnight VIX protection|Active", "Slippage Monitor — Fill quality tracking|Monitoring"]
                    Rectangle { Layout.fillWidth: true; height: 38; radius: 4; color: "#1A1D2E"
                        RowLayout { anchors.fill: parent; anchors.margins: 10; spacing: 12
                            Rectangle { width: 8; height: 8; radius: 4; color: modelData.split("|")[1] === "Armed" ? "#22C55E" : modelData.split("|")[1] === "Active" ? "#06B6D4" : "#F59E0B" }
                            Label { text: modelData.split("|")[0]; color: "#E2E8F0"; font.pixelSize: 11; Layout.fillWidth: true }
                            Label { text: modelData.split("|")[1]; color: modelData.split("|")[1] === "Armed" ? "#22C55E" : "#F59E0B"; font.pixelSize: 9; font.bold: true }
                        }
                    }
                }
                Item { Layout.fillHeight: true }
            }
        }
    }
}
