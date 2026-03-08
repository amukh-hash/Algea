import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8

PanelScaffold {
    property var modelData
    property var columns: []
    property string emptyText: "No rows"

    ColumnLayout {
        Layout.fillWidth: true
        Layout.fillHeight: true
        spacing: 6

        Rectangle {
            Layout.fillWidth: true
            height: 30
            radius: 4
            color: "#1B2940"
            RowLayout {
                anchors.fill: parent
                anchors.margins: 6
                spacing: 8
                Repeater {
                    model: columns
                    Label {
                        Layout.preferredWidth: modelData.width
                        text: modelData.title
                        color: "#94A3B8"
                        font.pixelSize: 10
                        font.bold: true
                    }
                }
            }
        }

        ListView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            model: modelData
            delegate: Rectangle {
                width: ListView.view.width
                height: 34
                readonly property string statusValue: (model.status || "").toString().toLowerCase()
                readonly property bool isCritical: statusValue === "failed" || statusValue === "breached" || statusValue === "halted" || statusValue === "disconnected"
                readonly property bool isWarn: statusValue === "degraded" || statusValue === "stale" || statusValue === "unknown" || statusValue === "monitoring"

                color: isCritical ? "#2A121A" : (isWarn ? "#221C10" : (index % 2 === 0 ? "#121C2F" : "#0F1728"))
                border.width: isCritical || isWarn ? 1 : 0
                border.color: isCritical ? "#7F1D1D" : (isWarn ? "#7C4A03" : "transparent")

                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 6
                    spacing: 8
                    Repeater {
                        model: columns
                        Label {
                            Layout.preferredWidth: modelData.width
                            text: {
                                const key = modelData.role;
                                const val = model[key];
                                return (val === undefined || val === null || val === "") ? "—" : val;
                            }
                            color: modelData.role === "status"
                                   ? (isCritical ? "#FCA5A5" : (isWarn ? "#FCD34D" : "#86EFAC"))
                                   : (isCritical ? "#FEE2E2" : "#E2E8F0")
                            font.pixelSize: 11
                            font.bold: modelData.role === "status" || isCritical
                            elide: Text.ElideRight
                            ToolTip.visible: hover.containsMouse && text.length > 20
                            ToolTip.text: text
                            HoverHandler { id: hover }
                        }
                    }
                }
            }

            Label {
                anchors.centerIn: parent
                visible: modelData.count === 0
                text: emptyText
                color: "#94A3B8"
                font.pixelSize: 11
            }
        }
    }
}
