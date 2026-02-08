from __future__ import annotations

import pandas as pd


def build_trade_log(trades: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(trades)
