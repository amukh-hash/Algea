import pandas as pd

from algaie.training.datasets import WindowConfig, build_valid_windows


def test_valid_windows_exclude_invalid_returns():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "ticker": ["AAA", "AAA", "AAA", "AAA"],
            "open": [10.0, 10.0, 10.0, 10.0],
            "high": [10.0, 10.0, 10.0, 10.0],
            "low": [10.0, 10.0, 10.0, 10.0],
            "close": [10.0, -1.0, 10.0, 10.0],
            "volume": [100.0, 100.0, 100.0, 100.0],
        }
    )
    index = build_valid_windows(df, WindowConfig(context=1, horizon=1))
    assert index["AAA"] == []
