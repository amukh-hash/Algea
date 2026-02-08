from __future__ import annotations

import pandas as pd


def topk_basket(signals: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    return signals.sort_values(["date", "rank"]).groupby("date").head(k).copy()
