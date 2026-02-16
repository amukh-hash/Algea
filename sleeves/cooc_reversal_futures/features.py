"""Runtime feature computation — delegates to features_core for parity.

This module provides the ``FeatureRow`` wrapper and ``compute_core_features``
that the sleeve uses at inference time.  All actual feature math lives in
``features_core.py`` to guarantee exact match with training.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from .features_core import (
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_V1,
    FEATURE_SCHEMA_V2,
    FeatureConfig,
    compute_core_features as _compute_core,
)


@dataclass(frozen=True)
class FeatureRow:
    values: dict[str, float]
    feature_timestamp_end: datetime
    decision_timestamp: datetime

    def assert_no_leakage(self) -> None:
        if not self.feature_timestamp_end < self.decision_timestamp:
            raise AssertionError("feature_timestamp_end must be strictly before decision_timestamp")


def compute_core_features(
    df: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """Compute canonical features for the runtime sleeve.

    Delegates to ``features_core.compute_core_features`` (shared with training).

    Parameters
    ----------
    df
        Gold-like frame with columns:
        ``instrument`` (or ``root``), ``trading_day``,
        ``r_co`` (or ``ret_co``), ``r_oc`` (or ``ret_oc``),
        ``volume``, ``days_to_expiry``, ``roll_window_flag``.
    lookback
        Rolling window size.

    Returns
    -------
    DataFrame with ``FEATURE_SCHEMA`` (V2) columns computed.
    """
    cfg = FeatureConfig(lookback=lookback, schema_version=2)
    return _compute_core(df, cfg=cfg)


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
