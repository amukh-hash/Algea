import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8

PanelScaffold {
    property int failedJobs: 0
    property bool paused: false
    title: "ACTION QUEUE"
    subtitle: "Operator priorities derived from live state"

    ColumnLayout {
        Layout.fillWidth: true
        spacing: 6
        SleeveStatusRow { sleeveName: "System Pause"; status: paused ? "ACTION" : "HEALTHY"; note: paused ? "Resume when safe" : "No action" }
        SleeveStatusRow { sleeveName: "Failed Jobs"; status: failedJobs > 0 ? "ACTION" : "HEALTHY"; note: failedJobs > 0 ? failedJobs + " failed" : "None" }
        SleeveStatusRow { sleeveName: "Alerts"; status: AlertEngine.activeAlertCount > 0 ? "ACTION" : "HEALTHY"; note: AlertEngine.activeAlertCount + " active" }
    }
}
