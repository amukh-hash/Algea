"""
SelectorFeatureFrame builder — deterministic, point-in-time safe feature engineering.

Ported from deprecated/backend_app_snapshot/features/selector_features_v2.py.
Decoupled from ``pathmap`` / ``run_recorder`` — operates on Polars DataFrames directly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import polars as pl

from algae.core.contracts import assert_schema, normalize_keys, SELECTOR_FEATURES_V2_REQUIRED_COLS

logger = logging.getLogger(__name__)


@dataclass
class SelectorFeatureConfig:
    """Configuration for selector feature generation."""

    min_breadth_train: int = 200
    horizon_days: int = 5
    lookback_window: int = 20
    volatility_window: int = 20
    relative_vol_window: int = 20
    tie_break_policy: str = "sort_value_then_symbol"


class SelectorFeatureBuilder:
    """
    Builds a normalised, cross-sectional feature frame for the ranking/selector model.

    Pipeline
    --------
    1. Compute raw causal features (log-returns, vol, relative volume)
    2. Join with universe (filter to tradable)
    3. Drop days with breadth < ``min_breadth_train``
    4. Cross-sectional rank normalisation → ``x_*`` features in ``[-1, 1]``
    5. Compute targets (``y_rank``, ``y_trade``)
    """

    def __init__(self, config: SelectorFeatureConfig | None = None) -> None:
        self.config = config or SelectorFeatureConfig()

    # ------------------------------------------------------------------
    def build(
        self,
        ohlcv: pl.LazyFrame | pl.DataFrame,
        universe: pl.LazyFrame | pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Parameters
        ----------
        ohlcv : DataFrame with ``[date, symbol, close, volume]``
        universe : DataFrame with ``[date, symbol, is_tradable, tier, weight]``

        Returns
        -------
        Feature frame with ``x_*`` normalised features + ``y_rank``, ``y_trade``
        """
        if isinstance(ohlcv, pl.DataFrame):
            ohlcv = ohlcv.lazy()
        if isinstance(universe, pl.DataFrame):
            universe = universe.lazy()

        # Normalise keys
        if "ticker" in ohlcv.collect_schema().names():
            ohlcv = ohlcv.rename({"ticker": "symbol"})
        if "ticker" in universe.collect_schema().names():
            universe = universe.rename({"ticker": "symbol"})

        # 1. Raw features
        features_lf = self._compute_raw_features(ohlcv)
        features_lf = features_lf.with_columns(pl.col("date").cast(pl.Date))

        # 2. Join + filter tradable
        combined_lf = features_lf.join(universe, on=["date", "symbol"], how="inner").filter(
            pl.col("is_tradable") == True  # noqa: E712
        )
        df = combined_lf.collect()

        # 3. Small-N filtering
        breadth = df.group_by("date").count()
        valid_days = breadth.filter(pl.col("count") >= self.config.min_breadth_train).select("date")
        dropped = len(breadth) - len(valid_days)
        if dropped:
            logger.info(f"Dropping {dropped} days with breadth < {self.config.min_breadth_train}")
        df = df.join(valid_days, on="date", how="inner")

        # 4. Rank normalisation
        df = self._apply_rank_normalization(df)

        # 5. Targets
        df = self._compute_targets(df)

        # 6. Validate
        df = normalize_keys(df)
        assert_schema(df, required_cols=SELECTOR_FEATURES_V2_REQUIRED_COLS)

        # Bounds check
        for col in [c for c in df.columns if c.startswith("x_")]:
            mn, mx = df[col].min(), df[col].max()
            if mn < -1.001 or mx > 1.001:
                raise ValueError(f"Feature {col} out of bounds: [{mn}, {mx}]")

        return df

    # ------------------------------------------------------------------
    def _compute_raw_features(self, ohlcv: pl.LazyFrame) -> pl.LazyFrame:
        cfg = self.config
        return (
            ohlcv.sort(["symbol", "date"])
            .with_columns(
                (pl.col("close") / pl.col("close").shift(1).over("symbol"))
                .log()
                .alias("log_return_1d")
            )
            .with_columns([
                pl.col("log_return_1d").rolling_sum(window_size=5).over("symbol").alias("log_return_5d"),
                pl.col("log_return_1d").rolling_sum(window_size=20).over("symbol").alias("log_return_20d"),
                pl.col("log_return_1d").rolling_std(window_size=cfg.volatility_window).over("symbol").alias("volatility_20d"),
                (pl.col("volume") / (
                    pl.col("volume").shift(1).rolling_median(window_size=cfg.relative_vol_window).over("symbol") + 1.0
                )).alias("relative_volume_20d"),
                (pl.col("close").shift(-cfg.horizon_days).over("symbol") / pl.col("close")).log().alias("target_return_raw"),
                pl.col("log_return_1d").rolling_std(window_size=cfg.horizon_days).shift(-cfg.horizon_days).over("symbol").alias("fwd_volatility"),
            ])
        )

    # ------------------------------------------------------------------
    def _apply_rank_normalization(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns((-pl.col("volatility_20d")).alias("vol_signal"))

        norm_features = ["log_return_1d", "log_return_5d", "log_return_20d", "vol_signal", "relative_volume_20d"]
        out_names = ["x_lr1", "x_lr5", "x_lr20", "x_vol", "x_relvol"]

        exprs = []
        for feat, out_name in zip(norm_features, out_names):
            r = pl.struct([feat, "symbol"]).rank("ordinal").over("date") - 1
            n_t = pl.col("symbol").count().over("date")
            norm = (2.0 * (r / (n_t - 1)) - 1.0).alias(out_name)
            exprs.append(norm)

        return df.with_columns(exprs)

    # ------------------------------------------------------------------
    def _compute_targets(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.rename({"target_return_raw": "y_rank"})

        fwd_vol = pl.coalesce([pl.col("fwd_volatility"), pl.col("volatility_20d")])
        risk_adj = (pl.col("y_rank") / (fwd_vol + 1e-8)).alias("risk_adj_return")
        df = df.with_columns(risk_adj)

        r = pl.struct(["risk_adj_return", "symbol"]).rank("ordinal").over("date") - 1
        n_t = pl.col("symbol").count().over("date")
        rank_pct = r / (n_t - 1)
        y_trade = (rank_pct >= 0.70).cast(pl.Int32).alias("y_trade")

        return df.with_columns([y_trade])
