import QtQuick 6.8
import QtQuick.Controls 6.8
import QtQuick.Layouts 6.8

PanelScaffold {
    title: "Guardrail Summary"
    subtitle: "Derived from guardrails_status.v1"

    function guardrailStatus(id) {
        const g = GlobalStore.guardrailById(id);
        if (!g || !g.status) return "UNKNOWN";
        return String(g.status).toUpperCase();
    }

    ColumnLayout {
        Layout.fillWidth: true
        spacing: 6
        SleeveStatusRow { Layout.fillWidth: true; sleeveName: "ECE Tracker"; status: guardrailStatus("ece_tracker"); note: "Calibration" }
        SleeveStatusRow { Layout.fillWidth: true; sleeveName: "MMD LiveGuard"; status: guardrailStatus("mmd_liveguard"); note: "Drift" }
        SleeveStatusRow { Layout.fillWidth: true; sleeveName: "Max Drawdown"; status: guardrailStatus("max_drawdown"); note: "Risk" }
        SleeveStatusRow { Layout.fillWidth: true; sleeveName: "Gap Risk Filter"; status: guardrailStatus("gap_risk_filter"); note: "Overnight" }
        SleeveStatusRow { Layout.fillWidth: true; sleeveName: "Slippage Monitor"; status: guardrailStatus("slippage_monitor"); note: "Execution" }
    }
}
