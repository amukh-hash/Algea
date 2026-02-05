import numpy as np
import pandas as pd
from typing import List, Callable, Optional
from backend.app.models.chronos2_teacher import load_chronos_adapter


class ChronosRunner:
    def __init__(self, model_id: str, context_len: int, horizon: int):
        self.model_id = model_id
        self.context_len = context_len
        self.horizon = horizon
        self.model = None

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        try:
            self.model = load_chronos_adapter(self.model_id)
        except Exception:
            self.model = None

    def _fallback_priors(self, series_df: pd.DataFrame) -> dict:
        if series_df is None or series_df.empty:
            return {
                "prior_drift_20d": 0.0,
                "prior_vol_20d": 0.0,
                "prior_downside_q10": 0.0,
                "prior_trend_conf": 0.5
            }

        df = series_df.copy()
        if "close_adj" not in df.columns:
            raise ValueError("series_df must contain close_adj for priors fallback.")

        df = df.sort_values("date")
        df["ret_1d"] = df["close_adj"].pct_change()
        window = df["ret_1d"].dropna().tail(20)
        if window.empty:
            return {
                "prior_drift_20d": 0.0,
                "prior_vol_20d": 0.0,
                "prior_downside_q10": 0.0,
                "prior_trend_conf": 0.5
            }

        drift = float(window.mean())
        vol = float(window.std(ddof=0))
        downside_q10 = float(np.quantile(window, 0.1))
        trend_conf = float((window > 0).mean())

        return {
            "prior_drift_20d": drift,
            "prior_vol_20d": vol,
            "prior_downside_q10": downside_q10,
            "prior_trend_conf": trend_conf
        }

    def infer_one(self, symbol: str, asof_date, series_df, covariates_df) -> dict:
        self._ensure_model()
        return self._fallback_priors(series_df)

    def infer_batch(self, symbols: List[str], asof_date, load_fn: Callable[[str, Optional[str]], pd.DataFrame]) -> pd.DataFrame:
        data = []
        for s in symbols:
            series_df = load_fn(s, asof_date)
            priors = self._fallback_priors(series_df)
            row = {
                "symbol": s,
                **priors
            }
            data.append(row)
        return pd.DataFrame(data)
