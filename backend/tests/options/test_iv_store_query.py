import pytest
from datetime import datetime
from backend.app.options.data.providers.mock import MockIVProvider
from backend.app.options.data.iv_store import ProviderIVStore, get_iv_store

def test_mock_provider_deterministic():
    p1 = MockIVProvider(seed=42)
    p2 = MockIVProvider(seed=42)

    ts = datetime(2023, 1, 1, 10, 0)
    iv1 = p1.get_iv("AAPL", ts, 30)
    iv2 = p2.get_iv("AAPL", ts, 30)

    assert iv1.atm_iv == iv2.atm_iv
    assert iv1.iv_rank == iv2.iv_rank
    assert iv1.ticker == "AAPL"

    # Different seed
    p3 = MockIVProvider(seed=43)
    iv3 = p3.get_iv("AAPL", ts, 30)
    assert iv3.atm_iv != iv1.atm_iv # Highly likely different

def test_iv_store_provider_mode():
    store = get_iv_store(mode="provider", provider=MockIVProvider(seed=123))
    ts = datetime(2023, 6, 15, 9, 30)
    snap = store.get_iv("TSLA", ts, 7)

    assert snap is not None
    assert snap.ticker == "TSLA"
    assert snap.dte == 7
    assert 0.0 < snap.atm_iv < 5.0
    assert 0.0 <= snap.iv_rank <= 1.0

def test_mock_history():
    p = MockIVProvider()
    start = datetime(2023, 1, 1)
    end = datetime(2023, 1, 10)
    history = p.get_iv_history("NVDA", start, end, 30)

    # weekends excluded in mock logic
    assert len(history) > 0
    assert len(history) <= 10
    assert all(h.ticker == "NVDA" for h in history)
