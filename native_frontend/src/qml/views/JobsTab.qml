import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8

Rectangle {
    color: "#0F111A"

    // Job Data Model
    ListModel { id: jobModel }
    property int jobTotal: 0
    property int jobRunning: 0
    property int jobFailed: 0

    function refreshJobs() {
        var xhr = new XMLHttpRequest()
        xhr.open("GET", "http://127.0.0.1:8000/api/control/job-graph")
        xhr.onreadystatechange = function() {
            if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
                try {
                    var resp = JSON.parse(xhr.responseText)
                    var jobs = resp.jobs || []
                    jobModel.clear()
                    var running = 0
                    var failed = 0
                    for (var i = 0; i < jobs.length; i++) {
                        var j = jobs[i]
                        var status = j.last_status || "idle"
                        var sessions = (j.sessions || []).join(", ")
                        var lastStarted = j.last_started || ""
                        var duration = j.last_duration_s != null ? j.last_duration_s + "s" : ""
                        var lastError = j.last_error || ""
                        if (status === "failed") failed++
                        if (status === "running") running++
                        var timeStr = ""
                        if (lastStarted) {
                            var parts = lastStarted.split("T")
                            if (parts.length > 1) timeStr = parts[1].split("-")[0].split("+")[0].substring(0, 8)
                        }
                        jobModel.append({
                            "jobName": j.name || "",
                            "jobSessions": sessions,
                            "jobStatus": status,
                            "jobLastRun": timeStr,
                            "jobDuration": duration,
                            "jobError": lastError
                        })
                    }
                    jobTotal = jobs.length
                    jobRunning = running
                    jobFailed = failed
                } catch (e) {}
            }
        }
        xhr.send()
    }

    Timer {
        id: jobPollTimer
        interval: 30000; running: true; repeat: true
        onTriggered: parent.refreshJobs()
    }
    Component.onCompleted: refreshJobs()

    ColumnLayout {
        anchors.fill: parent; anchors.margins: 16; spacing: 12
        Label { text: "JOBS & ORCHESTRATOR"; color: "#64748B"; font.pixelSize: 13; font.weight: Font.DemiBold; font.letterSpacing: 2 }
        RowLayout { Layout.fillWidth: true; spacing: 12
            Rectangle { Layout.fillWidth: true; height: 90; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
                ColumnLayout { anchors.fill: parent; anchors.margins: 12; spacing: 6; Label { text: "SCHEDULED"; color: "#64748B"; font.pixelSize: 9 } Label { text: jobTotal.toString(); color: "#6366F1"; font.pixelSize: 24; font.bold: true } Label { text: "Total jobs"; color: "#64748B"; font.pixelSize: 10 } } }
            Rectangle { Layout.fillWidth: true; height: 90; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
                ColumnLayout { anchors.fill: parent; anchors.margins: 12; spacing: 6; Label { text: "RUNNING"; color: "#64748B"; font.pixelSize: 9 } Label { text: jobRunning.toString(); color: jobRunning > 0 ? "#06B6D4" : "#22C55E"; font.pixelSize: 24; font.bold: true } Label { text: "Currently executing"; color: "#64748B"; font.pixelSize: 10 } } }
            Rectangle { Layout.fillWidth: true; height: 90; radius: 8; color: jobFailed > 0 ? "#1C0F0F" : "#141622"; border.color: jobFailed > 0 ? "#7F1D1D" : "#2A2D3E"; border.width: 1
                ColumnLayout { anchors.fill: parent; anchors.margins: 12; spacing: 6; Label { text: "FAILED"; color: "#64748B"; font.pixelSize: 9 } Label { text: jobFailed.toString(); color: jobFailed > 0 ? "#EF4444" : "#22C55E"; font.pixelSize: 24; font.bold: true } Label { text: "Current cycle"; color: "#64748B"; font.pixelSize: 10 } } }
        }
        Rectangle { Layout.fillWidth: true; Layout.fillHeight: true; radius: 8; color: "#141622"; border.color: "#2A2D3E"; border.width: 1
            ColumnLayout { anchors.fill: parent; anchors.margins: 16; spacing: 8
                Label { text: "JOB SCHEDULE"; color: "#64748B"; font.pixelSize: 11; font.weight: Font.DemiBold; font.letterSpacing: 2 }
                RowLayout { Layout.fillWidth: true
                    Label { text: "Job"; color: "#64748B"; font.pixelSize: 10; font.bold: true; Layout.preferredWidth: 280 }
                    Label { text: "Sessions"; color: "#64748B"; font.pixelSize: 10; font.bold: true; Layout.preferredWidth: 200 }
                    Label { text: "Last Run"; color: "#64748B"; font.pixelSize: 10; font.bold: true; Layout.preferredWidth: 100 }
                    Label { text: "Duration"; color: "#64748B"; font.pixelSize: 10; font.bold: true; Layout.preferredWidth: 80 }
                    Label { text: "Status"; color: "#64748B"; font.pixelSize: 10; font.bold: true; Layout.preferredWidth: 80 }
                    Item { Layout.fillWidth: true }
                }
                Rectangle { Layout.fillWidth: true; height: 1; color: "#2A2D3E" }
                ListView {
                    Layout.fillWidth: true; Layout.fillHeight: true
                    model: jobModel
                    clip: true
                    spacing: 2
                    delegate: Rectangle {
                        width: ListView.view.width; height: 36; radius: 4
                        color: index % 2 === 0 ? "transparent" : "#1A1D2E"
                        RowLayout { anchors.fill: parent; anchors.margins: 4
                            Label { text: jobName; color: "#E2E8F0"; font.pixelSize: 11; font.family: "Consolas"; Layout.preferredWidth: 280 }
                            Label { text: jobSessions; color: "#94A3B8"; font.pixelSize: 10; Layout.preferredWidth: 200; elide: Text.ElideRight }
                            Label { text: jobLastRun; color: "#94A3B8"; font.pixelSize: 11; font.family: "Consolas"; Layout.preferredWidth: 100 }
                            Label { text: jobDuration; color: "#94A3B8"; font.pixelSize: 11; font.family: "Consolas"; Layout.preferredWidth: 80 }
                            Rectangle {
                                radius: 3; implicitWidth: 70; implicitHeight: 18
                                color: jobStatus === "success" ? "#14532D" : jobStatus === "failed" ? "#7F1D1D" : jobStatus === "skipped" ? "#78350F" : "#222538"
                                Label {
                                    anchors.centerIn: parent; font.pixelSize: 9; font.bold: true
                                    text: jobStatus.charAt(0).toUpperCase() + jobStatus.slice(1)
                                    color: jobStatus === "success" ? "#22C55E" : jobStatus === "failed" ? "#EF4444" : jobStatus === "skipped" ? "#F59E0B" : "#64748B"
                                }
                            }
                            Item { Layout.fillWidth: true }
                        }
                        ToolTip.visible: jobError && ma.containsMouse
                        ToolTip.text: jobError
                        MouseArea { id: ma; anchors.fill: parent; hoverEnabled: true }
                    }
                }
            }
        }
    }
}
