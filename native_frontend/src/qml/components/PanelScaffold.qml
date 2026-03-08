import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8

Rectangle {
    id: root
    property string title: ""
    property string subtitle: ""
    default property alias contentData: body.data
    radius: 8
    color: "#141622"
    border.color: "#2A2D3E"
    border.width: 1

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 8
        Label { text: root.title; color: "#64748B"; font.pixelSize: 11; font.weight: Font.DemiBold; font.letterSpacing: 1 }
        Label { visible: root.subtitle.length > 0; text: root.subtitle; color: "#64748B"; font.pixelSize: 10 }
        ColumnLayout { id: body; Layout.fillWidth: true; Layout.fillHeight: true }
    }
}
