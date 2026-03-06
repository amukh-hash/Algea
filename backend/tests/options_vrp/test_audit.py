"""
Tests for VRP daily audit artifact.
"""
import json
import tempfile
from pathlib import Path

from algae.eval.vrp_audit import VRPDailyAudit


class TestVRPDailyAudit:
    def test_required_fields_present(self):
        audit = VRPDailyAudit(
            as_of_date="2024-07-15",
            regime="normal_carry",
            regime_score_components={"vix": 18.0, "drawdown": -0.02},
            forecast_inputs={"rv10_pred_p95": 0.22, "health_score": 0.92},
            gating_decisions=[{"underlying": "SPY", "allowed": True}],
            size_scaler_components={"pnl_scaler": 1.1, "proxy_scaler": 0.9},
            scenario_estimates={"worst_case": -3000, "top_contributor": "p1"},
            exit_actions=[],
            danger_zone_results=[],
        )
        d = audit.to_dict()
        assert "regime" in d
        assert "forecast_inputs" in d
        assert "gating_decisions" in d
        assert "scenario_estimates" in d
        assert "exit_actions" in d
        assert "danger_zone_results" in d
        assert "constraints_before" in d
        assert "constraints_after" in d

    def test_serializes_to_json(self):
        audit = VRPDailyAudit(as_of_date="2024-07-15", regime="crash_risk")
        j = audit.to_json()
        parsed = json.loads(j)
        assert parsed["regime"] == "crash_risk"

    def test_save_and_load(self):
        audit = VRPDailyAudit(
            as_of_date="2024-07-15",
            regime="caution",
            forecast_inputs={"rv10_pred_p90": 0.30},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = audit.save(root)
            assert path.exists()
            loaded = VRPDailyAudit.load(root, "2024-07-15")
            assert loaded.regime == "caution"
            assert loaded.forecast_inputs["rv10_pred_p90"] == 0.30

    def test_no_trade_day_audit(self):
        """Audit on no-trade day should still be valid."""
        audit = VRPDailyAudit(
            as_of_date="2024-07-16",
            regime="normal_carry",
            gating_decisions=[],
            exit_actions=[],
        )
        j = audit.to_json()
        assert "normal_carry" in j
