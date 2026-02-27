from datetime import datetime, timedelta

from backend.app.ml_platform.training.datasets.tsfm_windows import build_tsfm_windows


def test_chronos_downsampling_determinism():
    series = list(range(30))
    base = datetime(2026, 1, 1, 9, 30)
    timestamps = [(base + timedelta(minutes=i)).isoformat() for i in range(len(series))]
    w1 = build_tsfm_windows(series, context_length=4, prediction_length=2, timestamps=timestamps, downsample_freq="5min")
    w2 = build_tsfm_windows(series, context_length=4, prediction_length=2, timestamps=timestamps, downsample_freq="5min")
    assert w1 == w2
