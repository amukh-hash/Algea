from __future__ import annotations

import pandas as pd


def rank_ic(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    return frame["score"].corr(frame["rank"], method="spearman")
