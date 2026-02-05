"""
Chronos-2 HF pipeline helpers (predict_df-based).
Builds long DataFrames with target + covariates and computes priors from forecast outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from chronos import ChronosPipeline
except ImportError:  # pragma: no cover - optional dependency
    ChronosPipeline = None


@dataclass(frozen=True)
class ChronosPredictConfig:
    model_id: str
    target_col: str
    covariate_cols: Sequence[str]
    id_col: str = "id"
    timestamp_col: str = "timestamp"


class Chronos2HFPredictor:
    def __init__(self, config: ChronosPredictConfig, device: str = "cpu"):
        if ChronosPipeline is None:
            raise ImportError("chronos package is required for predict_df usage.")
        self.config = config
        self.pipeline = ChronosPipeline.from_pretrained(
            config.model_id,
            device_map=device,
        )

    def build_context_df(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {self.config.id_col, self.config.timestamp_col, self.config.target_col}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        covariate_cols = [c for c in self.config.covariate_cols if c in df.columns]
        cols = [self.config.id_col, self.config.timestamp_col, self.config.target_col, *covariate_cols]
        context = df.loc[:, cols].copy()
        context = context.sort_values([self.config.id_col, self.config.timestamp_col])
        return context

    def build_future_df(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None:
            return None
        required = {self.config.id_col, self.config.timestamp_col}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for future_df: {sorted(missing)}")
        covariate_cols = [c for c in self.config.covariate_cols if c in df.columns]
        cols = [self.config.id_col, self.config.timestamp_col, *covariate_cols]
        future = df.loc[:, cols].copy()
        future = future.sort_values([self.config.id_col, self.config.timestamp_col])
        return future

    def predict_df(
        self,
        context_df: pd.DataFrame,
        prediction_length: int,
        future_df: Optional[pd.DataFrame] = None,
        num_samples: int = 20,
    ) -> pd.DataFrame:
        context = self.build_context_df(context_df)
        future = self.build_future_df(future_df)
        return self.pipeline.predict_df(
            context_df=context,
            prediction_length=prediction_length,
            num_samples=num_samples,
            future_df=future,
        )

    def summarize_priors(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        id_col = self.config.id_col
        if id_col not in forecast_df.columns:
            raise ValueError(f"Forecast output missing '{id_col}' column.")

        sample_cols = [c for c in forecast_df.columns if c.startswith("sample_")]
        quantile_cols = [c for c in forecast_df.columns if c.replace(".", "", 1).isdigit()]
        if "prediction" in forecast_df.columns:
            value_col = "prediction"
        elif "value" in forecast_df.columns:
            value_col = "value"
        else:
            value_col = None

        results = []
        for series_id, group in forecast_df.groupby(id_col):
            if sample_cols:
                samples = group[sample_cols].to_numpy().reshape(-1, len(sample_cols))
                terminal = samples[-1]
                drift = float(np.median(terminal))
                vol = float(np.std(terminal))
                downside = float(np.quantile(terminal, 0.1))
                trend_conf = float(np.mean(terminal > 0))
            elif quantile_cols:
                q_cols = sorted(quantile_cols, key=float)
                q10 = float(group[q_cols[0]].iloc[-1])
                q50 = float(group[q_cols[len(q_cols) // 2]].iloc[-1])
                q90 = float(group[q_cols[-1]].iloc[-1])
                drift = q50
                vol = float(q90 - q10)
                downside = q10
                trend_conf = float(q50 > 0)
            elif value_col:
                terminal = float(group[value_col].iloc[-1])
                drift = terminal
                vol = 0.0
                downside = terminal
                trend_conf = float(terminal > 0)
            else:
                raise ValueError("Forecast output has no usable prediction columns.")

            results.append(
                {
                    id_col: series_id,
                    "teacher_drift": drift,
                    "teacher_vol_forecast": vol,
                    "teacher_tail_risk": downside,
                    "teacher_trend_conf": trend_conf,
                }
            )

        return pd.DataFrame(results)
