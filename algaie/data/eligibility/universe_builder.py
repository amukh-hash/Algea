"""
Universe builder -- enriched eligibility with tier/weight assignment.

Ported from deprecated/backend_app_snapshot/data/universe_frame.py.
Builds on top of the basic eligibility mask in ``algaie.data.eligibility.build``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from algaie.data.common import ensure_datetime, get_close_column


@dataclass
class UniverseConfig:
    """Configuration for universe construction."""

    min_price: float = 5.0
    min_dollar_vol: float = 1_000_000.0
    min_history_days: int = 60
    # Tier thresholds (dollar volume breakpoints for tiers 1-3)
    tier_breakpoints: List[float] = field(default_factory=lambda: [50_000_000.0, 10_000_000.0])


class UniverseBuilder:
    """
    Constructs a *UniverseFrame* -- a panel DataFrame with:
      ``[date, symbol, is_observable, is_tradable, tier, weight]``
    """

    def __init__(self, config: UniverseConfig | None = None) -> None:
        self.config = config or UniverseConfig()

    def build(self, daily: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        df = ensure_datetime(daily.copy())

        close_col = get_close_column(df)
        dollar_vol = df[close_col].abs() * df["volume"]

        # is_observable: price > min_price AND enough history
        df = df.sort_values(["symbol", "date"])
        hist_days = df.groupby("symbol").cumcount() + 1
        is_observable = (df[close_col] > cfg.min_price) & (hist_days >= cfg.min_history_days)

        # is_tradable: observable AND sufficient dollar volume
        is_tradable = is_observable & (dollar_vol >= cfg.min_dollar_vol)

        # Tier assignment via pd.cut (simpler than np.select)
        bp = sorted(cfg.tier_breakpoints, reverse=True)
        bins = [float("inf")] + bp + [0.0]
        tier = pd.cut(dollar_vol, bins=list(reversed(bins)), labels=False, right=False)
        # pd.cut labels from 0; we want tier 1 = highest liquidity
        tier = len(bp) + 1 - tier.fillna(len(bp) + 1).astype(int)

        # Weight: inverse tier, normalised to mean=1 within each date
        max_tier = tier.max() if not df.empty else 1
        weight = (max_tier + 1 - tier).astype(float)
        date_mean = weight.groupby(df["date"]).transform("mean")
        weight = weight / date_mean.clip(lower=1e-8)

        return pd.DataFrame({
            "date": df["date"].values,
            "symbol": df["symbol"].values,
            "is_observable": is_observable.values,
            "is_tradable": is_tradable.values,
            "tier": tier.values,
            "weight": weight.values,
        })
