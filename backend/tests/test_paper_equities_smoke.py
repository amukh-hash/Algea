import pytest
import polars as pl
import pandas as pd
from unittest.mock import MagicMock, patch
from backend.app.engine.equity_pod import EquityPod
from backend.app.risk.types import ActionType, RiskDecision
from backend.app.models.signal_types import ModelSignal

@patch("backend.app.engine.equity_pod.StudentRunner")
def test_equity_pod_smoke(mock_runner_cls):
    # Setup Mock Runner
    mock_runner = MagicMock()
    mock_runner_cls.return_value = mock_runner

    # Mock Inference Output
    mock_signal = ModelSignal(
        horizons=["1D", "3D"],
        direction_probs={"3D": 0.9}, # Bullish
        quantiles={"3D": {"0.50": 0.01}}
    )
    mock_runner.infer.return_value = mock_signal

    # Init Pod
    # Model/Preproc paths are dummy since we mocked Runner class
    pod = EquityPod("AAPL", "dummy_model", "dummy_preproc")

    # Mock Schedule to allow trading
    pod.scheduler.get_window = MagicMock(return_value="EARLY_ENTRY")
    pod.scheduler.get_allowed_actions = MagicMock(return_value=["BUY", "SELL"])

    # Mock Buffer to be full enough
    # We need to feed 'lookback' ticks.
    # Let's just manually populate buffer.

    # Create dummy tick
    ts = pd.Timestamp("2023-01-01 10:00:00", tz="UTC")
    tick = {
        "timestamp": ts,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0
    }
    breadth = {"ad_line": 0.0, "bpi": 50.0}

    # Feed 1 tick
    decision = pod.on_tick(tick, breadth)

    # Buffer should have 1 row.
    assert pod.buffer.height == 1

    # Buffer < lookback (128) -> Should return None (no inference)
    assert decision is None

    # Fill buffer
    # We cheat and just set the buffer
    # Create DF with 128 rows
    pod.buffer = pl.DataFrame({
        "timestamp": [ts] * 128,
        "open": [100.0] * 128,
        "high": [100.0] * 128,
        "low": [100.0] * 128,
        "close": [100.0] * 128,
        "volume": [1000.0] * 128,
        "ad_line": [0.0] * 128,
        "bpi": [50.0] * 128
    })

    # Now feed one more tick -> Should trigger inference
    decision = pod.on_tick(tick, breadth)

    # Should have called infer
    mock_runner.infer.assert_called()

    # Should have a decision
    assert decision is not None
    assert decision.ticker == "AAPL"
    # Bullish signal (0.9) -> Should Buy (if logic holds)
    # Check RiskManager logic (normal posture + prob > 0.6 -> BUY)
    assert decision.action == ActionType.BUY

    # Execute
    pod.execute_decision(decision)
    assert "AAPL" in pod.portfolio.positions
    assert pod.portfolio.positions["AAPL"].quantity > 0
