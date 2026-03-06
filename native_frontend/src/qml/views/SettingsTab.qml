import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8

Rectangle {
    color: "#0F111A"
    ColumnLayout {
        anchors.fill: parent; anchors.margins: 16; spacing: 12
        Label { text: "SETTINGS & OPERATIONS"; color: "#64748B"; font.pixelSize: 13; font.weight: Font.DemiBold; font.letterSpacing: 2 }
        GridLayout { Layout.fillWidth: true; Layout.fillHeight: true; columns: 2; columnSpacing: 12; rowSpacing: 12
            Rectangle { Layout.fillWidth: true; Layout.fillHeight: true; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
                ColumnLayout { anchors.fill: parent; anchors.margins: 16; spacing: 10
                    Label { text: "CONNECTIONS"; color: "#64748B"; font.pixelSize: 11; font.weight: Font.DemiBold; font.letterSpacing: 2 }
                    GridLayout { columns: 2; columnSpacing: 16; rowSpacing: 8; Layout.fillWidth: true
                        Label { text: "Backend API"; color: "#94A3B8" } Label { text: "http://localhost:8000"; color: "#22C55E"; font.family: "Consolas"; font.pixelSize: 11 }
                        Label { text: "ZMQ Events"; color: "#94A3B8" } Label { text: "tcp://localhost:5556"; color: "#22C55E"; font.family: "Consolas"; font.pixelSize: 11 }
                        Label { text: "ZMQ Grid"; color: "#94A3B8" } Label { text: "tcp://localhost:5557"; color: "#22C55E"; font.family: "Consolas"; font.pixelSize: 11 }
                        Label { text: "Kill Switch SHM"; color: "#94A3B8" } Label { text: "AlgaeControlPlane"; color: "#22C55E"; font.family: "Consolas"; font.pixelSize: 11 }
                        Label { text: "Broker"; color: "#94A3B8" } Label { text: GlobalStore.brokerConnected ? "IBKR TWS Connected" : "Not connected"; color: GlobalStore.brokerConnected ? "#22C55E" : "#EF4444"; font.pixelSize: 11 }
                    }
                    Item { Layout.fillHeight: true }
                }
            }
            Rectangle { Layout.fillWidth: true; Layout.fillHeight: true; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
                ColumnLayout { anchors.fill: parent; anchors.margins: 16; spacing: 10
                    Label { text: "HARDWARE PARTITIONING"; color: "#64748B"; font.pixelSize: 11; font.weight: Font.DemiBold; font.letterSpacing: 2 }
                    GridLayout { columns: 2; columnSpacing: 16; rowSpacing: 8; Layout.fillWidth: true
                        Label { text: "GPU 0 (iGPU)"; color: "#94A3B8" } Label { text: "Qt Vulkan RHI / Scene Graph"; color: "#06B6D4"; font.pixelSize: 11 }
                        Label { text: "GPU 1 (CUDA:0)"; color: "#94A3B8" } Label { text: "PatchTST Inference"; color: "#6366F1"; font.pixelSize: 11 }
                        Label { text: "GPU 2 (CUDA:1)"; color: "#94A3B8" } Label { text: "TFT Training / Shadow"; color: "#6366F1"; font.pixelSize: 11 }
                        Label { text: "Render Loop"; color: "#94A3B8" } Label { text: "Threaded (16ms budget)"; color: "#E2E8F0"; font.pixelSize: 11 }
                        Label { text: "FIDO2"; color: "#94A3B8" } Label { text: FidoGateway.available ? FidoGateway.deviceName : "Not detected (stubbed)"; color: FidoGateway.available ? "#22C55E" : "#F59E0B"; font.pixelSize: 11 }
                    }
                    Item { Layout.fillHeight: true }
                }
            }
            Rectangle { Layout.fillWidth: true; Layout.columnSpan: 2; height: 70; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
                RowLayout { anchors.fill: parent; anchors.margins: 16; spacing: 24
                    Column { spacing: 2; Label { text: "BUILD"; color: "#64748B"; font.pixelSize: 9 } Label { text: "Algae v" + BuildEnvironment.buildVersion; color: "#E2E8F0"; font.pixelSize: 12 } }
                    Column { spacing: 2; Label { text: "COMPILER"; color: "#64748B"; font.pixelSize: 9 } Label { text: BuildEnvironment.compilerId + " " + BuildEnvironment.cxxStandard; color: "#E2E8F0"; font.pixelSize: 12 } }
                    Column { spacing: 2; Label { text: "RHI"; color: "#64748B"; font.pixelSize: 9 } Label { text: "Vulkan 1.3 (RHI)"; color: "#E2E8F0"; font.pixelSize: 12 } }
                    Column { spacing: 2; Label { text: "ARROW"; color: "#64748B"; font.pixelSize: 9 } Label { text: "Arrow " + BuildEnvironment.arrowVersion; color: "#E2E8F0"; font.pixelSize: 12 } }
                    Column { spacing: 2; Label { text: "PROTOBUF"; color: "#64748B"; font.pixelSize: 9 } Label { text: "Protobuf " + BuildEnvironment.protobufVersion; color: "#E2E8F0"; font.pixelSize: 12 } }
                    Item { Layout.fillWidth: true }
                }
            }
        }
    }
}
