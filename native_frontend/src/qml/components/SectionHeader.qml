import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8

RowLayout {
    property string title: ""
    property string subtitle: ""
    Layout.fillWidth: true
    spacing: 8

    ColumnLayout {
        spacing: 2
        Label { text: title; color: "#E8EDF8"; font.pixelSize: 15; font.bold: true }
        Label { text: subtitle; color: "#95A3BF"; font.pixelSize: 10; visible: subtitle.length > 0 }
    }
    Item { Layout.fillWidth: true }
}
