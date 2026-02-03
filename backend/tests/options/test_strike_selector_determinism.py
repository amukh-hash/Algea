import pytest
from datetime import datetime
from backend.app.options.strategy.strike_selector import StrikeSelector
from backend.app.options.data.types import OptionChainSnapshot, OptionRow
from backend.app.models.types import DistributionForecast

def test_strike_selector_determinism():
    selector = StrikeSelector()

    # Setup chain
    rows = [
        OptionRow(90, "put", 1.0, 1.1, 0.2),
        OptionRow(95, "put", 2.0, 2.1, 0.2),
        OptionRow(100, "put", 3.0, 3.1, 0.2),
    ]
    chain = OptionChainSnapshot("AAPL", datetime.now(), "2023-01-01", 30, rows)

    # Setup forecast
    forecast = DistributionForecast("Mock", ["3D"], {"3D": {"0.50": 0.0}})

    # Underlying 100
    cand1 = selector.select_best_spread(100.0, chain, forecast)
    cand2 = selector.select_best_spread(100.0, chain, forecast)

    if cand1:
        assert cand1 == cand2
        assert cand1.short_strike < 100.0
        assert cand1.strategy_type == "put_credit_spread"

def test_strike_selector_none_valid():
    selector = StrikeSelector()
    rows = [OptionRow(90, "put", 0.01, 0.02, 0.2)] # Too cheap
    chain = OptionChainSnapshot("AAPL", datetime.now(), "2023-01-01", 30, rows)
    forecast = DistributionForecast("Mock", ["3D"], {"3D": {"0.50": 0.0}})

    cand = selector.select_best_spread(100.0, chain, forecast)
    assert cand is None
