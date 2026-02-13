from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureRow:
    values: dict[str, float]
    feature_timestamp_end: datetime
    decision_timestamp: datetime

    def assert_no_leakage(self) -> None:
        if not self.feature_timestamp_end < self.decision_timestamp:
            raise AssertionError("feature_timestamp_end must be strictly before decision_timestamp")


def compute_core_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    out = df.copy()
    out["sigma_co"] = out["r_co"].rolling(lookback, min_periods=3).std().replace(0, np.nan)
    out["z_co"] = out["r_co"] / out["sigma_co"]
    for col in ("r_co", "z_co"):
        out[f"{col}_winsor"] = out[col].clip(out[col].quantile(0.05), out[col].quantile(0.95))
    out["r_co_cs_demean"] = out["r_co"] - out.groupby("date")["r_co"].transform("mean")
    out["r_co_rank_pct"] = out.groupby("date")["r_co"].rank(pct=True)
    out["r_oc_mean_l"] = out.groupby("instrument")["r_oc"].transform(lambda s: s.rolling(lookback, min_periods=3).mean())
    out["r_oc_vol_l"] = out.groupby("instrument")["r_oc"].transform(lambda s: s.rolling(lookback, min_periods=3).std())
    out["r_co_mean_l"] = out.groupby("instrument")["r_co"].transform(lambda s: s.rolling(lookback, min_periods=3).mean())
    out["r_co_vol_l"] = out.groupby("instrument")["r_co"].transform(lambda s: s.rolling(lookback, min_periods=3).std())
    return out


def micro_features(window: pd.DataFrame) -> dict[str, float]:
    spread_rel = ((window["ask"] - window["bid"]) / ((window["ask"] + window["bid"]) / 2.0)).mean()
    depth = float((window["bid_size"] + window["ask_size"]).mean())
    imbalance = float(((window["bid_size"] - window["ask_size"]) / (window["bid_size"] + window["ask_size"]).replace(0, np.nan)).mean())
    rv = float(np.log(window["price"]).diff().std())
    return {
        "spread_rel": float(spread_rel),
        "depth": depth,
        "book_imbalance": imbalance,
        "short_rv": rv,
    }
