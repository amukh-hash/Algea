import pandas as pd

from algea.data.features.build import build_features


def test_feature_build_deterministic():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "ticker": ["AAA", "AAA"],
            "open": [10.0, 11.0],
            "high": [12.0, 12.0],
            "low": [9.0, 10.0],
            "close": [11.0, 12.0],
            "volume": [100.0, 110.0],
        }
    )
    first = build_features(df)
    second = build_features(df)
    pd.testing.assert_frame_equal(first, second)
