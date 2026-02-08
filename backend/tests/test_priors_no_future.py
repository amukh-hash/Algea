from datetime import date

import pandas as pd

from algaie.data.priors.build import build_priors


def test_priors_ignore_future_data():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "ticker": ["AAA", "AAA", "AAA"],
            "open": [10.0, 11.0, 100.0],
            "high": [12.0, 12.0, 100.0],
            "low": [9.0, 10.0, 100.0],
            "close": [11.0, 12.0, 100.0],
            "volume": [100.0, 110.0, 120.0],
        }
    )
    asof = date.fromisoformat("2024-01-02")
    priors_full = build_priors(df, asof)
    priors_truncated = build_priors(df[df["date"] <= "2024-01-02"], asof)
    pd.testing.assert_frame_equal(priors_full.reset_index(drop=True), priors_truncated.reset_index(drop=True))
