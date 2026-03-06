import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8
import algae.Rendering 1.0

Rectangle {
    color: "#0F111A"
    ColumnLayout {
        anchors.fill: parent; anchors.margins: 16; spacing: 12
        Label { text: "THE LAB — SHADOW MODE"; color: "#64748B"; font.pixelSize: 13; font.weight: Font.DemiBold; font.letterSpacing: 2 }
        RowLayout { Layout.fillWidth: true; spacing: 12
            Rectangle { Layout.fillWidth: true; height: 100; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
                ColumnLayout { anchors.fill: parent; anchors.margins: 12; spacing: 6; Label { text: "SHADOW RUNS"; color: "#64748B"; font.pixelSize: 9 } Label { text: GlobalStore.shadowRuns.toString(); color: "#6366F1"; font.pixelSize: 28; font.bold: true } Label { text: "Parallel non-execution"; color: "#64748B"; font.pixelSize: 10 } } }
            Rectangle { Layout.fillWidth: true; height: 100; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
                ColumnLayout { anchors.fill: parent; anchors.margins: 12; spacing: 6; Label { text: "PENDING PROMOTIONS"; color: "#64748B"; font.pixelSize: 9 } Label { text: GlobalStore.pendingPromotions.toString(); color: "#6366F1"; font.pixelSize: 28; font.bold: true } Label { text: "Awaiting FIDO2 auth"; color: "#64748B"; font.pixelSize: 10 } } }
            Rectangle { Layout.fillWidth: true; height: 100; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
                ColumnLayout { anchors.fill: parent; anchors.margins: 12; spacing: 6; Label { text: "PROMOTED MODELS"; color: "#64748B"; font.pixelSize: 9 } Label { text: GlobalStore.promotedModels.toString(); color: "#6366F1"; font.pixelSize: 28; font.bold: true } Label { text: "Passed all gates"; color: "#64748B"; font.pixelSize: 10 } } }
        }

        // Promotion Pipeline with FIDO2 gate
        Rectangle { Layout.fillWidth: true; Layout.fillHeight: true; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
            ColumnLayout { anchors.fill: parent; anchors.margins: 16; spacing: 8
                Label { text: "PROMOTION PIPELINE — FIDO2 CRYPTOGRAPHIC GATE"; color: "#64748B"; font.pixelSize: 11; font.weight: Font.DemiBold; font.letterSpacing: 2 }
                Repeater { model: ["1|Universe Expansion|Expand training data", "2|Model Training|Cross-validated training", "3|Shadow Evaluation|30-day shadow trades", "4|Statistical Gates|Sharpe>1.2 DA>55% MaxDD<3%"]
                    Rectangle { Layout.fillWidth: true; height: 48; radius: 6; color: "#1A1D2E"
                        RowLayout { anchors.fill: parent; anchors.margins: 12; spacing: 12
                            Rectangle { width: 28; height: 28; radius: 14; color: "#222538"; Label { anchors.centerIn: parent; text: modelData.split("|")[0]; color: "#94A3B8"; font.pixelSize: 12; font.bold: true } }
                            Column { spacing: 2; Layout.fillWidth: true; Label { text: modelData.split("|")[1]; color: "#E2E8F0"; font.pixelSize: 12; font.bold: true } Label { text: modelData.split("|")[2]; color: "#64748B"; font.pixelSize: 10 } }
                            Rectangle { radius: 3; color: "#222538"; implicitWidth: 50; implicitHeight: 20; Label { anchors.centerIn: parent; text: "Ready"; color: "#64748B"; font.pixelSize: 9; font.bold: true } }
                        }
                    }
                }

                // FIDO2 Promotion Gate — Stage 5
                Rectangle { Layout.fillWidth: true; height: 80; radius: 6; color: "#1A1D2E"; border.color: "#6366F1"; border.width: 1
                    RowLayout { anchors.fill: parent; anchors.margins: 12; spacing: 12
                        Rectangle { width: 28; height: 28; radius: 14; color: "#6366F1"; Label { anchors.centerIn: parent; text: "5"; color: "white"; font.pixelSize: 12; font.bold: true } }
                        Column { spacing: 2; Layout.fillWidth: true
                            Label { text: "Promotion Decision — CTAP2 Hardware Auth Required"; color: "#E2E8F0"; font.pixelSize: 12; font.bold: true }
                            Label { text: FidoGateway.available ? ("FIDO2 device: " + FidoGateway.deviceName) : "No FIDO2 device detected (stubbed)"; color: FidoGateway.available ? "#22C55E" : "#F59E0B"; font.pixelSize: 10 }
                        }
                        Button {
                            text: FidoGateway.waiting ? "Touch your key..." : "Promote to Live"
                            enabled: !FidoGateway.waiting
                            palette.button: "#6366F1"; palette.buttonText: "white"
                            font.pixelSize: 11; font.bold: true
                            onClicked: {
                                FidoGateway.requestSignature("sha256:model_hash_placeholder")
                            }
                        }
                    }
                }

                // FIDO2 callback
                Connections {
                    target: FidoGateway
                    function onTouchRequired() { yubikeyModal.open() }
                    function onSignatureComplete(success, sig, error) {
                        yubikeyModal.close()
                        if (success) {
                            console.log("FIDO2 promotion authorized: " + sig)
                            var xhr = new XMLHttpRequest()
                            xhr.open("POST", "http://127.0.0.1:8000/api/control/promote")
                            xhr.setRequestHeader("Content-Type", "application/json")
                            xhr.send(JSON.stringify({
                                model_hash: "sha256:model_hash_placeholder",
                                signature: sig
                            }))
                        } else {
                            console.log("FIDO2 promotion failed: " + error)
                        }
                    }
                }

                Popup {
                    id: yubikeyModal
                    modal: true; focus: true
                    anchors.centerIn: Overlay.overlay
                    closePolicy: Popup.NoAutoClose
                    width: 380; height: 140
                    background: Rectangle { color: "#1A1D27"; radius: 12; border.color: "#6366F1"; border.width: 2 }
                    RowLayout {
                        anchors.fill: parent; anchors.margins: 20; spacing: 16
                        BusyIndicator { running: true; Layout.preferredWidth: 40; Layout.preferredHeight: 40 }
                        Column { spacing: 4; Layout.fillWidth: true
                            Label { text: "AWAITING HARDWARE KEY"; color: "#E2E8F0"; font.pixelSize: 13; font.bold: true; font.letterSpacing: 1 }
                            Label { text: "Touch your FIDO2 token to authorize\nstrategy promotion to live capital."; color: "#94A3B8"; font.pixelSize: 11 }
                        }
                    }
                }
                Item { Layout.fillHeight: true }

                // SMoE Expert Routing Visualization
                Label { text: "EXPERT ROUTING — PARALLEL COORDINATES"; color: "#64748B"; font.pixelSize: 11; font.weight: Font.DemiBold; font.letterSpacing: 2 }
                Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 200; radius: 8; color: "#1A1D2E"; border.color: "#2A2D3E"; border.width: 1
                    ParallelCoordinatesItem {
                        anchors.fill: parent; anchors.margins: 8
                    }
                }
            }
        }
    }
}
