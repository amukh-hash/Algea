from __future__ import annotations

import pandas as pd


def build_equity_curve(snapshots: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(snapshots)
