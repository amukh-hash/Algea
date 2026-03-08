import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8

Rectangle {
    property string tone: "warn"
    property string message: ""
    visible: message.length > 0
    Layout.fillWidth: true
    height: visible ? 34 : 0
    radius: 6
    color: tone === "danger" ? "#3A161A" : tone === "warn" ? "#38280F" : "#1D3242"
    border.color: tone === "danger" ? "#7F1D1D" : tone === "warn" ? "#7C4A03" : "#1D4E89"

    Label {
        anchors.centerIn: parent
        text: message
        color: tone === "danger" ? "#FCA5A5" : tone === "warn" ? "#FCD34D" : "#7DD3FC"
        font.pixelSize: 11
        font.bold: true
    }
}
