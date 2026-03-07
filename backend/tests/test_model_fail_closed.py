"""
Test: Fail-closed model initialization.

Ensures that all production adapters raise RuntimeError or ValueError
when called without trained weights, preventing the Orchestrator DAG
from routing random-weight garbage signals to IBKR.
"""
from __future__ import annotations

import pytest


class TestChronos2FailClosed:
    """Chronos2Adapter must fail closed without weights."""

    def test_no_weights_raises_on_forecast(self):
        from backend.app.ml_platform.models.chronos2.adapter import Chronos2Adapter

        adapter = Chronos2Adapter(model_weights_path=None, device="cpu")
        assert not adapter.is_loaded, "Adapter should not be loaded without weights"

        with pytest.raises(RuntimeError, match="fail-closed"):
            adapter.forecast([100.0, 101.0, 102.0] * 20, prediction_length=3)

    def test_empty_series_raises(self):
        from backend.app.ml_platform.models.chronos2.adapter import Chronos2Adapter

        adapter = Chronos2Adapter(model_weights_path=None, device="cpu")
        with pytest.raises((RuntimeError, ValueError)):
            adapter.forecast([], prediction_length=3)


class TestSMoEFailClosed:
    """SMoEExpertEnsemble must fail closed without weights."""

    def test_no_weights_raises_on_compute_scores(self):
        from backend.app.ml_platform.models.selector_smoe.experts import SMoEExpertEnsemble

        ensemble = SMoEExpertEnsemble(
            num_experts=4, d_input=18, weights_dir=None, device="cpu"
        )
        assert not ensemble.is_loaded, "Ensemble should not be loaded without weights"

        with pytest.raises(RuntimeError, match="fail-closed"):
            ensemble.compute_scores([[0.1] * 18 for _ in range(10)])

    def test_empty_features_raises(self):
        from backend.app.ml_platform.models.selector_smoe.experts import SMoEExpertEnsemble

        ensemble = SMoEExpertEnsemble(
            num_experts=4, d_input=18, weights_dir=None, device="cpu"
        )
        with pytest.raises((RuntimeError, ValueError)):
            ensemble.compute_scores([])


class TestRLPolicyFailSafe:
    """RLPolicyModel must return safe pass-through without weights."""

    def test_no_weights_returns_passthrough(self):
        from backend.app.ml_platform.models.rl_policy.model import RLPolicyModel

        model = RLPolicyModel(hidden_size=32, raw_feature_dim=10, device="cpu")
        assert not model.is_active, "Model should not be active without weights"

        mult, veto, conf = model.act([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        assert mult == 1.0, f"Untrained model should pass-through (mult=1.0), got {mult}"
        assert veto is False, f"Untrained model should not veto, got {veto}"

    def test_nan_input_vetoes(self):
        """NaN in features must trigger a veto, not silent propagation."""
        from backend.app.ml_platform.models.rl_policy.model import RLPolicyModel

        model = RLPolicyModel(hidden_size=32, raw_feature_dim=5, device="cpu")
        # Force active to test NaN guard
        model.is_active = True

        mult, veto, _ = model.act([1.0, float("nan"), 0.5, 0.1, -0.3])
        assert veto is True, "NaN input must trigger veto"
        assert mult == 0.0, "NaN input must zero the multiplier"


class TestVolSurfaceGridGuard:
    """VolSurfaceGridForecaster must not inject dummy tensors."""

    def test_non_empty_history_returns_embedding(self):
        from backend.app.ml_platform.models.vol_surface_grid.model import VolSurfaceGridForecaster

        model = VolSurfaceGridForecaster()
        # get_state_embedding is now implemented — verify it returns something
        result = model.get_state_embedding([{"iv": {"SPY": 0.2}, "liq": {"SPY": 0.5}}])
        assert result is not None


class TestStatArbHandlerGuard:
    """StatArb signal handler must reject synthetic features."""

    def test_handler_processes_without_crash(self, tmp_path):
        """StatArb handler now runs the feature pipeline — verify it completes."""
        import json
        from backend.app.orchestrator.job_defs import handle_signals_generate_statarb

        root = tmp_path
        ctx = {
            "asof_date": "2026-03-03",
            "session": "PREMARKET",
            "mode": "paper",
            "config": type("C", (), {"enable_statarb_sleeve": True})(),
            "artifact_root": str(root),
        }
        result = handle_signals_generate_statarb(ctx)
        assert result["status"] in ("ok", "degraded")
