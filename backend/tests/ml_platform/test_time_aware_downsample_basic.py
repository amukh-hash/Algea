from datetime import datetime

import pandas as pd

from backend.app.ml_platform.utils.downsample import parse_duration_to_timedelta, time_aware_downsample


def test_time_aware_downsample_basic():
    idx = pd.date_range(datetime(2026, 1, 1, 9, 30), periods=12, freq="1min")
    df = pd.DataFrame({"value": list(range(12))}, index=idx)
    out = time_aware_downsample(df, parse_duration_to_timedelta("5min"))
    assert list(out.index) == [pd.Timestamp("2026-01-01 09:30:00"), pd.Timestamp("2026-01-01 09:35:00"), pd.Timestamp("2026-01-01 09:40:00")]
    assert out["value"].tolist() == [4, 9, 11]
