import QtQuick 6.8
import QtQuick.Controls 6.8

Rectangle {
    id: root
    property string freshness: "disconnected"
    radius: 4
    implicitHeight: 22
    implicitWidth: label.implicitWidth + 14
    color: freshness === "fresh" ? "#14532D" : freshness === "stale" ? "#78350F" : freshness === "degraded" ? "#7F1D1D" : "#3F1D1D"

    Label {
        id: label
        anchors.centerIn: parent
        text: "DATA " + freshness.toUpperCase()
        color: freshness === "fresh" ? "#22C55E" : freshness === "stale" ? "#F59E0B" : "#EF4444"
        font.pixelSize: 10
        font.bold: true
    }
}
