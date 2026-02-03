import pytest
import os
from unittest.mock import MagicMock
from backend.app.engine.options_pod import OptionsPod
from backend.app.options.gate.context import OptionsContext
from backend.app.models.signal_types import ModelSignal
from backend.app.core import config
from datetime import datetime

def test_monitor_determinism_smoke(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_OPTIONS", True)
    monkeypatch.setattr(config, "OPTIONS_MODE", "monitor")
    monkeypatch.setattr(config, "OPTIONS_SEED", 42)

    # Init Pod
    pod = OptionsPod()

    # Mock context
    ctx = OptionsContext(
        ticker="AAPL",
        timestamp=datetime(2023, 1, 1),
        underlying_price=150.0,
        student_signal=ModelSignal(horizons=["3D"], quantiles={"3D": {"0.50": 0.02, "0.05": -0.01}}),
        breadth={"bpi": 60.0, "ad_line": 1000.0}
    )

    # Run twice
    dec1 = pod.on_signal(ctx)
    dec2 = pod.on_signal(ctx)

    # Should be identical
    if dec1 and dec2:
        assert dec1.candidate == dec2.candidate
        assert dec1.reason == dec2.reason
    elif dec1 is None and dec2 is None:
        pass
    else:
        assert False, "Non-deterministic output"

    # Execute (should be no-op)
    if dec1:
        pod.execute(dec1) # Should not raise
