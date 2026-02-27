from backend.app.ml_platform.feature_store.builders.vol_surface_grid import build_vol_surface_grid_dataset


def test_vol_surface_grid_builder_determinism(tmp_path):
    rows = [{"ts": "2026-01-01T00:00:00", "iv": {"7:ATM": 0.2}, "liq": {"7:ATM": 1.0}, "ret": {"7:ATM": 0.01}, "target": {"7:ATM": 0.21}}]
    a = build_vol_surface_grid_dataset(rows, tmp_path)
    b = build_vol_surface_grid_dataset(rows, tmp_path)
    assert a["dataset_id"] == b["dataset_id"]
