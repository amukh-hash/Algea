import QtQuick 6.8
import QtQuick.Controls 6.8

Rectangle {
    id: root
    property string text: "UNKNOWN"
    property string tone: "neutral" // neutral|ok|warn|danger|info
    radius: 4
    implicitHeight: 22
    implicitWidth: label.implicitWidth + 14
    color: tone === "ok" ? "#14532D" : tone === "warn" ? "#78350F" : tone === "danger" ? "#7F1D1D" : tone === "info" ? "#083344" : "#222538"

    Label {
        id: label
        anchors.centerIn: parent
        text: root.text
        color: tone === "ok" ? "#22C55E" : tone === "warn" ? "#F59E0B" : tone === "danger" ? "#EF4444" : tone === "info" ? "#06B6D4" : "#94A3B8"
        font.pixelSize: 10
        font.bold: true
    }
}
