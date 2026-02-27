from backend.app.data.options.snapshot_writer import write_snapshot_atomic


def _row():
    return {
        "asof": "2026-01-01T00:00:00Z",
        "underlying_symbol": "SPY",
        "expiry": "2026-02-01",
        "dte": 30,
        "option_type": "C",
        "strike": 100.0,
        "bid": 1.0,
        "ask": 1.2,
        "mid": 1.1,
        "implied_vol": 0.2,
        "delta": 0.25,
        "gamma": 0.01,
        "vega": 0.1,
        "theta": -0.01,
        "spot": 100.0,
    }


def test_snapshot_atomic_write(tmp_path):
    path = write_snapshot_atomic(tmp_path, [_row()], "s1")
    assert path.exists()
    assert not (tmp_path / "s1.tmp.json").exists()
