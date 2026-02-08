from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from algaie.models.foundation.base import FoundationModel, FoundationModelOutput
from algaie.models.foundation.registry import register


@dataclass(frozen=True)
class FoundationModelConfig:
    enable_quantiles: bool = False
    context_length: int = 60
    horizon_short: int = 5
    horizon_long: int = 10


class SimpleChronos2(FoundationModel):
    def __init__(self, config: FoundationModelConfig) -> None:
        self.config = config
        if self.config.enable_quantiles:
            raise ValueError("Quantile heads are disabled until encoder pooling is verified")

    def infer(self, canonical_daily: pd.DataFrame) -> pd.DataFrame:
        df = canonical_daily.sort_values(["ticker", "date"]).copy()
        df["ret_1d"] = df.groupby("ticker")["close"].pct_change()
        df["p_mu5"] = (
            df.groupby("ticker")["ret_1d"]
            .rolling(self.config.horizon_short, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df["p_mu10"] = (
            df.groupby("ticker")["ret_1d"]
            .rolling(self.config.horizon_long, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df["p_sig5"] = (
            df.groupby("ticker")["ret_1d"]
            .rolling(self.config.horizon_short, min_periods=1)
            .std(ddof=0)
            .reset_index(level=0, drop=True)
        )
        df["p_sig10"] = (
            df.groupby("ticker")["ret_1d"]
            .rolling(self.config.horizon_long, min_periods=1)
            .std(ddof=0)
            .reset_index(level=0, drop=True)
        )
        df["p_pdown5"] = (df["p_mu5"] < 0).astype(float)
        df["p_pdown10"] = (df["p_mu10"] < 0).astype(float)
        df[["p_mu5", "p_mu10", "p_sig5", "p_sig10"]] = df[[
            "p_mu5",
            "p_mu10",
            "p_sig5",
            "p_sig10",
        ]].fillna(0)
        df[["p_sig5", "p_sig10"]] = df[["p_sig5", "p_sig10"]].replace(0, 1e-6)
        priors = df[
            [
                "date",
                "ticker",
                "p_mu5",
                "p_mu10",
                "p_sig5",
                "p_sig10",
                "p_pdown5",
                "p_pdown10",
            ]
        ].copy()
        return priors

    def infer_priors(self, canonical_df: pd.DataFrame, asof: pd.Timestamp | None = None) -> FoundationModelOutput:
        priors = self.infer(canonical_df)
        if asof is not None:
            priors = priors[priors["date"] == asof]
        return FoundationModelOutput(priors=priors)

    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame | None = None) -> str:
        _ = (train_data, val_data)
        return "chronos2_stub_checkpoint"


register("chronos2", SimpleChronos2)
