import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8

PanelScaffold {
    title: "POSITIONS"
    subtitle: "Arrow-backed model"

    TableView {
        Layout.fillWidth: true
        Layout.fillHeight: true
        model: PositionsGrid
        clip: true
        columnWidthProvider: function (column) { return [100, 70, 60, 80, 100, 100, 120, 140][column] || 100 }
        delegate: Rectangle {
            implicitHeight: 30
            color: row % 2 === 0 ? "transparent" : "#1A1D2E"
            Label {
                anchors.fill: parent
                anchors.margins: 4
                text: display || ""
                color: "#E2E8F0"
                font.pixelSize: 10
                font.family: "Consolas"
                verticalAlignment: Text.AlignVCenter
            }
        }
    }
}
