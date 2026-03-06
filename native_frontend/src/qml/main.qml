import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8
import QtQuick.Window 6.8
import algae.Rendering 1.0

ApplicationWindow {
    id: rootWindow
    visible: true
    width: 2560
    height: 1440
    title: isSimulation ? "Algae [SIMULATION]" : "Algae Trading Orchestrator"
    color: GlobalStore.dataLossFlag ? "#2A0000" : "#0F111A"
    Behavior on color { ColorAnimation { duration: 300 } }

    // ── IBKR Auto-Connect ────────────────────────────────────────
    // Phase 1 bootstrap: check broker status (GET), then POST /connect if needed.
    // All ongoing state polling is handled by C++ RestClient (30s timer).
    property int _brokerRetryCount: 0
    property string _brokerStatus: "Checking..."

    Timer {
        id: brokerAutoConnect
        interval: 3000; running: true; repeat: false
        onTriggered: {
            if (GlobalStore.brokerConnected) {
                rootWindow._brokerStatus = "Connected"
                return
            }
            var statusXhr = new XMLHttpRequest()
            statusXhr.open("GET", "http://127.0.0.1:8000/api/control/broker-status")
            statusXhr.onreadystatechange = function() {
                if (statusXhr.readyState === XMLHttpRequest.DONE) {
                    try {
                        var statusResp = JSON.parse(statusXhr.responseText)
                        if (statusResp.connected) {
                            rootWindow._brokerStatus = "Connected (" + (statusResp.account_id || "") + ")"
                            GlobalStore.setBrokerConnected(true)
                            return
                        }
                    } catch (e) {}

                    rootWindow._brokerStatus = "Connecting..."
                    var xhr = new XMLHttpRequest()
                    xhr.open("POST", "http://127.0.0.1:8000/api/control/broker/connect")
                    xhr.setRequestHeader("Content-Type", "application/json")
                    xhr.onreadystatechange = function() {
                        if (xhr.readyState === XMLHttpRequest.DONE) {
                            try {
                                var resp = JSON.parse(xhr.responseText)
                                if (resp.connected) {
                                    rootWindow._brokerStatus = "Connected (" + (resp.account || "") + ")"
                                    GlobalStore.setBrokerConnected(true)
                                } else {
                                    rootWindow._brokerRetryCount++
                                    rootWindow._brokerStatus = "Failed: " + (resp.error || "unknown")
                                    if (rootWindow._brokerRetryCount < 5) brokerRetryTimer.start()
                                    else rootWindow._brokerStatus = "Not available (retries exhausted)"
                                }
                            } catch (e) {
                                rootWindow._brokerRetryCount++
                                rootWindow._brokerStatus = "Backend unreachable"
                                if (rootWindow._brokerRetryCount < 5) brokerRetryTimer.start()
                            }
                        }
                    }
                    xhr.send("{}")
                }
            }
            statusXhr.send()
        }
    }

    Timer {
        id: brokerRetryTimer
        interval: 15000; running: false; repeat: false
        onTriggered: {
            if (!GlobalStore.brokerConnected) {
                brokerAutoConnect.start()
            }
        }
    }

    // NOTE: Duplicate broker/calendar XHR polling REMOVED.
    // All ongoing state polling is now handled exclusively by C++ RestClient
    // at 30s intervals → GlobalStore → QML bindings (Unidirectional Data Flow).

    // ── Sim Watermark ────────────────────────────────────────────
    Rectangle {
        visible: isSimulation
        anchors.fill: parent; z: 999999; color: "transparent"
        Repeater {
            model: 8
            Text {
                x: (index % 4) * (rootWindow.width / 3) - 100
                y: Math.floor(index / 4) * (rootWindow.height / 2) + 100
                text: "SIMULATED / NON-PRODUCTION"
                color: "#30ff4444"; font.pixelSize: 32; font.bold: true; rotation: -30; opacity: 0.3
            }
        }
    }

    // ── Status Bar ───────────────────────────────────────────────
    header: ToolBar {
        height: 40
        background: Rectangle { color: "#12141E" }
        RowLayout {
            anchors.fill: parent; anchors.margins: 8; spacing: 16
            Label { text: "Algae"; color: "#6366F1"; font.pixelSize: 13; font.weight: Font.Black; font.letterSpacing: 3 }
            Rectangle { width: 1; height: 20; color: "#2A2D3E" }
            Rectangle { width: 8; height: 8; radius: 4; color: GlobalStore.brokerConnected ? "#22C55E" : "#EF4444" }
            Label { text: GlobalStore.brokerConnected ? "Connected" : "Disconnected"; color: GlobalStore.brokerConnected ? "#22C55E" : "#EF4444"; font.pixelSize: 11 }
            Rectangle { radius: 3; color: "#222538"; implicitWidth: sessL.implicitWidth + 12; implicitHeight: 20
                Label { id: sessL; anchors.centerIn: parent; text: GlobalStore.currentSession.toUpperCase(); color: "#06B6D4"; font.pixelSize: 9; font.bold: true } }
            Rectangle { radius: 3; color: GlobalStore.executionMode === "ibkr" ? "#7F1D1D" : "#78350F"; implicitWidth: modeL.implicitWidth + 12; implicitHeight: 20
                Label { id: modeL; anchors.centerIn: parent; text: GlobalStore.executionMode.toUpperCase(); color: "white"; font.pixelSize: 9; font.bold: true } }
            Label { visible: GlobalStore.systemPaused; text: "⏸ PAUSED"; color: "#F59E0B"; font.pixelSize: 11; font.bold: true }
            Label { visible: GlobalStore.dataLossFlag; text: "⚠ ZMQ OVERFLOW"; color: "#EF4444"; font.bold: true; font.pixelSize: 11 }
            Item { Layout.fillWidth: true }
            Label { text: "Portfolio"; color: "#64748B"; font.pixelSize: 10 }
            Label { text: "$" + GlobalStore.totalPortfolioValue.toFixed(2); color: "#E2E8F0"; font.pixelSize: 13; font.bold: true }
            Rectangle { width: 1; height: 20; color: "#2A2D3E" }
            Label { text: "PnL"; color: "#64748B"; font.pixelSize: 10 }
            Label { text: (GlobalStore.totalPnl >= 0 ? "+$" : "-$") + Math.abs(GlobalStore.totalPnl).toFixed(2); color: GlobalStore.totalPnl >= 0 ? "#22C55E" : "#EF4444"; font.pixelSize: 13; font.bold: true }
        }
    }

    // ── Tab Bar ──────────────────────────────────────────────────
    TabBar {
        id: domainTabs
        anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right
        background: Rectangle { color: "#141622" }
        Repeater {
            model: ["Overview", "Signals", "Risk", "Allocation", "Portfolio", "Lab", "Jobs", "Settings"]
            TabButton {
                text: modelData; font.pixelSize: 11; width: implicitWidth + 20
                palette.button: "#141622"
                palette.buttonText: domainTabs.currentIndex === index ? "#E2E8F0" : "#64748B"
            }
        }
    }

    // ── Content — Lazy-loaded per-tab components ─────────────────
    // Tab 0 (Overview) loads synchronously for instant boot visibility.
    // All other tabs use asynchronous: true, deferring Vulkan context
    // creation and heavy mesh generation until the user navigates there.
    StackLayout {
        anchors.top: domainTabs.bottom; anchors.left: parent.left; anchors.right: parent.right; anchors.bottom: parent.bottom
        currentIndex: domainTabs.currentIndex

        Loader { active: domainTabs.currentIndex === 0; source: "views/OverviewTab.qml";    asynchronous: false }
        Loader { active: domainTabs.currentIndex === 1; source: "views/SignalsTab.qml";      asynchronous: true }
        Loader { active: domainTabs.currentIndex === 2; source: "views/RiskTab.qml";         asynchronous: true }
        Loader { active: domainTabs.currentIndex === 3; source: "views/AllocationTab.qml";   asynchronous: true }
        Loader { active: domainTabs.currentIndex === 4; source: "views/PortfolioTab.qml";    asynchronous: true }
        Loader { active: domainTabs.currentIndex === 5; source: "views/LabTab.qml";          asynchronous: true }
        Loader { active: domainTabs.currentIndex === 6; source: "views/JobsTab.qml";         asynchronous: true }
        Loader { active: domainTabs.currentIndex === 7; source: "views/SettingsTab.qml";     asynchronous: true }
    }
}
