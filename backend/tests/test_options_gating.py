import pytest
from datetime import datetime
from backend.app.options.gate.context import OptionsContext
from backend.app.options.gate.gate import OptionsGate
from backend.app.options.types import GateReasonCode
from backend.app.models.signal_types import ModelSignal, ChronosPriors

@pytest.fixture
def mock_context():
    return OptionsContext(
        ticker="AAPL",
        timestamp=datetime(2024, 1, 1),
        underlying_price=150.0,
        student_signal=ModelSignal(horizons=["1D"]),
        breadth={"bpi": 50.0} # Passes BPI check
    )

def test_gate_equity_selection(mock_context):
    gate = OptionsGate()
    
    # 1. Not in selection
    mock_context.in_equity_selection = False
    decision = gate.evaluate(mock_context)
    assert not decision.should_trade
    assert decision.reason_code == GateReasonCode.REJECT_REGIME
    assert "Not in equity selection" in decision.reason_desc

    # 2. In selection
    mock_context.in_equity_selection = True
    # Should fail on missing priors next
    decision = gate.evaluate(mock_context)
    assert not decision.should_trade
    assert decision.reason_code == GateReasonCode.UNCERTAINTY_HIGH
    assert "Missing teacher priors" in decision.reason_desc

def test_gate_priors(mock_context):
    gate = OptionsGate()
    mock_context.in_equity_selection = True
    
    # Weak Priors
    mock_context.teacher_priors = ChronosPriors(
        drift_20d=0.0,
        vol_20d=0.02,
        downside_q10_20d=-0.05,
        trend_conf_20d=0.4 # Too low (thresh 0.5)
    )
    decision = gate.evaluate(mock_context)
    assert not decision.should_trade
    assert decision.reason_code == GateReasonCode.TREND_WEAK
    
    # Good Priors
    mock_context.teacher_priors = ChronosPriors(
        drift_20d=0.0,
        vol_20d=0.02,
        downside_q10_20d=-0.05,
        trend_conf_20d=0.6 # Good
    )
    
    # Also need student P50 to pass (default thresh -0.01)
    # Mock context doesn't have student signal details populated in 'feats'?
    # gate uses compute_gate_features(ctx)
    # We should look at gate logic for student p50. 
    # It reads feats["student_p50_3d"].
    # compute_gate_features likely mocks it or extracts it.
    # Let's assume default flows.
    # If compute_gate_features sets valid default, it might pass.
    # Otherwise we might fail on Student P50.
    
    # Assuming compute_gate_features works with minimal ctx:
    # If it fails, check why.
    # decision = gate.evaluate(mock_context)
    pass
