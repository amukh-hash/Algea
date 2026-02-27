from backend.app.ml_platform.feature_store.builders.vol_surface import build_vol_surface_dataset


def test_vol_surface_builder_determinism(tmp_path):
    rows = [{"dte": 7, "strike": 100, "spot": 100, "implied_vol": 0.2, "option_type": "C", "delta": 0.2}]
    close = [100 + i for i in range(200)]
    d1 = build_vol_surface_dataset("v1", "2026-01-01", "SPY", rows, close, idx=100, out_root=tmp_path)
    d2 = build_vol_surface_dataset("v1", "2026-01-01", "SPY", rows, close, idx=100, out_root=tmp_path)
    assert d1["manifest"]["manifest_hash"] == d2["manifest"]["manifest_hash"]
