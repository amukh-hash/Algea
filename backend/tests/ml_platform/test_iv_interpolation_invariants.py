from backend.app.ml_platform.feature_store.builders.vol_surface import build_vol_surface_dataset


def test_iv_interpolation_invariants():
    rows = [{"dte": 7, "strike": 100, "spot": 100, "implied_vol": 0.2, "option_type": "C", "delta": 0.25}]
    close = [100 + i for i in range(100)]
    ds = build_vol_surface_dataset("x", "2026-01-01", "SPY", rows, close, idx=50)
    assert 7 in ds["features"]
