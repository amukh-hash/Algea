from backend.app.data.options.snapshot_writer import snapshot_id


def test_snapshot_id_determinism():
    a = snapshot_id("SPY", "2026-01-01", "params")
    b = snapshot_id("SPY", "2026-01-01", "params")
    assert a == b
