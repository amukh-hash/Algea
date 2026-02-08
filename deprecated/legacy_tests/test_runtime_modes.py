import sys
from unittest.mock import MagicMock, patch
import pytest
import logging
import polars as pl
from backend.app.core import config
from backend.app.models.feature_contracts import compute_contract_hash, validate_contract

# Mock calendar to avoid ImportError if dependencies missing
sys.modules["backend.app.data.calendar"] = MagicMock()
sys.modules["backend.app.data.universe"] = MagicMock() # Often imported by runners
sys.modules["exchange_calendars"] = MagicMock()

from backend.app.engine.equity_pod import EquityPod

def test_feature_contracts():
    cols = ["open", "high", "low", "close", "volume"]
    h = compute_contract_hash(cols)
    assert len(h) == 16
    
    df = pl.DataFrame({"open": [1], "close": [2], "high": [3], "low": [4], "volume": [5]})
    assert validate_contract(df, h)
    
    # Extra col -> Hash should differ (strict)
    df2 = df.with_columns(pl.lit(0).alias("extra"))
    assert not validate_contract(df2, h)

def test_config_mock_inference():
    # Just ensure it doesn't crash
    val = config.NIGHTLY_MOCK_INFERENCE
    assert isinstance(val, bool)

def test_legacy_mode_warning(caplog):
    # Mock config.EXECUTION_MODE to LEGACY locally?
    # Config is imported at module level in EquityPod.
    # We need to patch the CLASS attribute or config import.
    from unittest.mock import patch
    
    with caplog.at_level(logging.WARNING):
        # We need to patch config used inside equity_pod
        with patch("backend.app.engine.equity_pod.EXECUTION_MODE", "LEGACY"):
             # We also mock StudentRunner etc to avoid loading real models
             with patch("backend.app.engine.equity_pod.StudentRunner"):
                 pod = EquityPod("AAPL", "dummy", "dummy")
                 assert "running in LEGACY mode" in caplog.text
