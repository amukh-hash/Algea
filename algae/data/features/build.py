from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from algae.data.common import ensure_datetime, write_dataframe
from algae.data.features.validate import validate_feature_frame


def build_features(canonical_daily: pd.DataFrame) -> pd.DataFrame:
    df = ensure_datetime(canonical_daily.copy())
    df = df.sort_values(["ticker", "date"]).copy()
    df["ret_1d"] = df.groupby("ticker")["close"].pct_change()
    df["ret_5d"] = df.groupby("ticker")["close"].pct_change(5)
    df["vol_20d"] = (
        df.groupby("ticker")["ret_1d"].rolling(20).std().reset_index(level=0, drop=True)
    )
    df["dollar_vol"] = df["close"] * df["volume"]
    features = df[["date", "ticker", "ret_1d", "ret_5d", "vol_20d", "dollar_vol"]].copy()
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    validate_feature_frame(features, strict=False)
    return features


def write_features(frame: pd.DataFrame, destination: Path) -> None:
    write_dataframe(frame, destination)
