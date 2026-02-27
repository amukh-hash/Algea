from datetime import datetime, timedelta

from backend.app.ml_platform.training.datasets.tsfm_windows import build_tsfm_windows


def test_chronos_time_aware_downsampling_integration():
    base = datetime(2026, 1, 1, 9, 30)
    series = list(range(20))
    timestamps = [(base + timedelta(minutes=i)).isoformat() for i in range(20)]
    windows = build_tsfm_windows(series, context_length=2, prediction_length=1, timestamps=timestamps, downsample_freq="5min")
    # Downsample keeps last in each 5-min bin => [4,9,14,19]
    assert windows == [([4.0, 9.0], [14.0]), ([9.0, 14.0], [19.0])]
