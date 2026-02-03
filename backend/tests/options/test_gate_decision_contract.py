import pytest
from backend.app.options.types import GateDecision, GateReasonCode
from backend.app.options.gate.gate import OptionsGate
from backend.app.options.gate.context import OptionsContext
from backend.app.models.signal_types import ModelSignal
from datetime import datetime

def test_gate_decision_contract():
    # Setup context
    signal = ModelSignal(horizons=["3D"], quantiles={"3D": {"0.50": 0.02, "0.05": -0.01}})
    ctx = OptionsContext(
        ticker="AAPL",
        timestamp=datetime.now(),
        underlying_price=150.0,
        student_signal=signal,
        breadth={"bpi": 60.0, "ad_line": 1000.0}
    )

    gate = OptionsGate()
    decision = gate.evaluate(ctx)

    assert isinstance(decision, GateDecision)
    assert isinstance(decision.should_trade, bool)
    assert isinstance(decision.reason_code, GateReasonCode)

    # Check default behavior
    assert decision.should_trade is True # With these good inputs

    # Check failure
    ctx.breadth["bpi"] = 10.0
    decision_bad = gate.evaluate(ctx)
    assert decision_bad.should_trade is False
    assert decision_bad.reason_code == GateReasonCode.REJECT_REGIME
