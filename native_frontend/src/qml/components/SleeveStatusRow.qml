import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8

Rectangle {
    id: root
    property string sleeveName: ""
    property string status: "UNKNOWN"
    property string note: ""
    radius: 4
    height: 36
    color: "#1A1D2E"

    RowLayout {
        anchors.fill: parent
        anchors.margins: 8
        Label { text: sleeveName; color: "#E2E8F0"; font.pixelSize: 11; font.bold: true; Layout.preferredWidth: 90 }
        StatusBadge {
            text: status;
            tone: (status === "HEALTHY" || status === "LIVE" || status === "ARMED" || status === "RUNNING") ? "ok"
                  : (status === "HALTED" || status === "BREACHED" || status === "DISCONNECTED") ? "danger"
                  : (status === "DEGRADED" || status === "MONITORING" || status === "ACTION") ? "warn"
                  : (status === "WEIGHT/CONF") ? "info"
                  : "neutral"
        }
        Label {
            Layout.fillWidth: true
            text: note
            color: "#94A3B8"
            font.pixelSize: 10
            elide: Text.ElideRight
            ToolTip.visible: note.length > 72 && ma.containsMouse
            ToolTip.text: note
            MouseArea { id: ma; anchors.fill: parent; hoverEnabled: true; acceptedButtons: Qt.NoButton }
        }
    }
}
