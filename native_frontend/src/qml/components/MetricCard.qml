import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8

Rectangle {
    id: root
    property string title: ""
    property string value: ""
    property string subtitle: ""
    property color valueColor: "#E2E8F0"
    radius: 8
    color: "#141622"
    border.color: "#2A2D3E"
    border.width: 1

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 6
        Label { text: root.title; color: "#64748B"; font.pixelSize: 9; font.letterSpacing: 1 }
        Label { text: root.value; color: root.valueColor; font.pixelSize: 24; font.bold: true }
        Label { text: root.subtitle; color: "#64748B"; font.pixelSize: 10 }
    }
}
