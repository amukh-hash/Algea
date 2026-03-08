import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8

Popup {
    id: root
    property string title: "DETAIL"
    width: 420
    height: 320
    modal: false
    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
    background: Rectangle { color: "#141622"; border.color: "#2A2D3E"; radius: 8 }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        Label { text: root.title; color: "#E2E8F0"; font.pixelSize: 14; font.bold: true }
        Label { text: "Detailed drilldown wiring is pending backend contract consolidation."; color: "#94A3B8"; wrapMode: Text.WordWrap }
    }
}
