from backend.app.ml_platform.drift.detectors import tenor_drift_score


def test_vol_surface_drift_scoring():
    hist = {7: [{"rv_hist_20": 0.2}, {"rv_hist_20": 0.3}]}
    baseline = {"7": {"rv_hist_20_mean": 0.2, "rv_hist_20_std": 0.1}}
    assert tenor_drift_score(hist, baseline) > 0
