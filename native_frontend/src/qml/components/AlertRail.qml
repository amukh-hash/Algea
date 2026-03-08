import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8

PanelScaffold {
    property int activeAlerts: 0
    title: "ALERT RAIL"
    subtitle: "Live alert count from AlertEngine"

    RowLayout {
        Layout.fillWidth: true
        spacing: 12
        StatusBadge { text: activeAlerts > 0 ? "ACTION REQUIRED" : "NO ACTIVE ALERTS"; tone: activeAlerts > 0 ? "danger" : "ok" }
        Label { text: activeAlerts.toString() + " active"; color: "#E2E8F0"; font.pixelSize: 16; font.bold: true }
    }
}
