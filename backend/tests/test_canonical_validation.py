import pandas as pd

from algae.data.canonical.validate import validate_canonical_daily


def test_canonical_validation_flags_close_and_duplicates():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "open": [10.0, 11.0],
            "high": [12.0, 12.0],
            "low": [9.0, 10.0],
            "close": [0.0, 11.0],
            "volume": [100.0, 100.0],
        }
    )
    issues = validate_canonical_daily(df)
    messages = {issue.message for issue in issues}
    assert "close <= 0" in messages
    assert "duplicate dates" in messages
