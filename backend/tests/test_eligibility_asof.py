from datetime import date

import pandas as pd

from algae.data.eligibility.build import build_eligibility


def test_eligibility_asof_stable():
    df = pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
            ],
            "ticker": ["AAA", "AAA", "AAA", "AAA"],
            "close": [6.0, 6.0, 6.0, 1.0],
            "volume": [1000, 1000, 1000, 1000],
            "open": [6.0, 6.0, 6.0, 1.0],
            "high": [6.0, 6.0, 6.0, 1.0],
            "low": [6.0, 6.0, 6.0, 1.0],
        }
    )
    asof = date.fromisoformat("2024-01-03")
    full = build_eligibility(df, asof)
    truncated = build_eligibility(df[df["date"] <= "2024-01-03"], asof)
    assert full.equals(truncated)
