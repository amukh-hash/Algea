import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8
import "../components"

Rectangle {
    color: "#090C14"

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 14
        spacing: 10

        SectionHeader { title: "Portfolio & Execution"; subtitle: "Arrow-backed positions with explicit freshness" }

        SeverityBanner {
            tone: GlobalStore.backendReachable ? "warn" : "danger"
            message: GlobalStore.dataFreshness === "fresh" ? "" : "Portfolio values may be stale/degraded"
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            MetricCard { Layout.fillWidth: true; height: 84; title: "Portfolio Value"; value: "$" + GlobalStore.totalPortfolioValue.toFixed(2); subtitle: "Control API"; valueColor: "#E8EDF8" }
            MetricCard { Layout.fillWidth: true; height: 84; title: "PnL"; value: (GlobalStore.totalPnl >= 0 ? "+$" : "-$") + Math.abs(GlobalStore.totalPnl).toFixed(2); subtitle: "Unrealized + realized"; valueColor: GlobalStore.totalPnl >= 0 ? "#22C55E" : "#EF4444" }
            MetricCard { Layout.fillWidth: true; height: 84; title: "Positions"; value: GlobalStore.positionCount.toString(); subtitle: "Live holdings"; valueColor: "#7C8CFF" }
        }

        PositionsTable {
            Layout.fillWidth: true
            Layout.fillHeight: true
        }
    }
}
