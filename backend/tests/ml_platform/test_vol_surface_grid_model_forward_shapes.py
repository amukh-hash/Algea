from backend.app.ml_platform.models.vol_surface_grid.model import VolSurfaceGridForecaster


def test_vol_surface_grid_model_forward_shapes():
    m = VolSurfaceGridForecaster()
    pred, unc, drift = m.forecast([{"iv": {"7:ATM": 0.2}, "liq": {"7:ATM": 1.0}, "ret": 0.1}])
    assert isinstance(pred, dict)
    assert isinstance(unc, float)
    assert isinstance(drift, float)
