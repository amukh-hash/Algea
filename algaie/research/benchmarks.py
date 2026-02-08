from __future__ import annotations

import pandas as pd


def spy_returns(prices: pd.DataFrame) -> pd.Series:
    if "close" not in prices.columns:
        raise KeyError("close column missing in benchmark prices")
    return prices.set_index("date")["close"].pct_change().dropna()
