from backend.app.ml_platform.models.vol_surface.model import VolSurfaceForecaster


def test_vol_surface_forward_shapes():
    model = VolSurfaceForecaster()
    out = model.forecast({7: [{"rv_hist_20": 0.2}] * 5, 30: [{"rv_hist_20": 0.3}] * 5}, [0.1, 0.5, 0.9])
    assert 7 in out and 30 in out
    assert "0.50" in out[7]
