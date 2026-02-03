import pytest
import os
import polars as pl
from datetime import datetime
from backend.app.options.data.iv_store import ParquetIVStore

@pytest.fixture
def temp_iv_dir(tmp_path):
    d = tmp_path / "iv"
    d.mkdir()
    return str(d)

def test_parquet_iv_store_roundtrip(temp_iv_dir):
    # 1. Create dummy parquet file
    ticker = "SPY"
    timestamps = [
        datetime(2023, 1, 1, 9, 30),
        datetime(2023, 1, 1, 10, 0),
        datetime(2023, 1, 1, 10, 30)
    ]

    df = pl.DataFrame({
        "timestamp": timestamps,
        "ticker": [ticker]*3,
        "dte": [30]*3,
        "atm_iv": [0.15, 0.155, 0.16],
        "iv_rank": [0.2, 0.22, 0.25],
        "iv_percentile": [0.2, 0.22, 0.25]
    })

    path = os.path.join(temp_iv_dir, f"{ticker}.parquet")
    df.write_parquet(path)

    # 2. Init Store
    store = ParquetIVStore(data_dir=temp_iv_dir)

    # 3. Query exact match
    snap = store.get_iv(ticker, timestamps[1], 30)
    assert snap is not None
    assert snap.atm_iv == 0.155
    assert snap.timestamp == timestamps[1]

    # 4. Query backward search (asof)
    # Query at 10:15 -> should get 10:00 (since 10:30 is future)
    query_ts = datetime(2023, 1, 1, 10, 15)
    snap_asof = store.get_iv(ticker, query_ts, 30)
    assert snap_asof is not None
    assert snap_asof.timestamp == timestamps[1] # 10:00
    assert snap_asof.atm_iv == 0.155

    # 5. Missing ticker
    assert store.get_iv("UNKNOWN", timestamps[0], 30) is None

    # 6. Missing DTE
    assert store.get_iv(ticker, timestamps[0], 999) is None
