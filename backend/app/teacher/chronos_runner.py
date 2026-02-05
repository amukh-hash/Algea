
import pandas as pd
from typing import List
from backend.app.models.chronos2_teacher import load_chronos_adapter

class ChronosRunner:
    def __init__(self, model_id: str, context_len: int, horizon: int):
        self.model_id = model_id
        self.context_len = context_len
        self.horizon = horizon
        self.model = None # Lazy load
        
    def infer_one(self, symbol: str, asof_date, series_df, covariates_df) -> dict:
        """
        Infer priors for a single symbol.
        """
        return {
            'prior_drift_20d': 0.005,
            'prior_vol_20d': 0.02,
            'prior_downside_q10': -0.05,
            'prior_trend_conf': 0.6
        }

    def infer_batch(self, symbols: List[str], asof_date, load_fn) -> pd.DataFrame:
        """
        Infer priors for a batch of symbols.
        """
        data = []
        for s in symbols:
            data.append({
                #'date': asof_date, # Often joined later or expected
                'symbol': s,
                'prior_drift_20d': 0.005,
                'prior_vol_20d': 0.02,
                'prior_downside_q10': -0.05,
                'prior_trend_conf': 0.6
            })
        return pd.DataFrame(data)
