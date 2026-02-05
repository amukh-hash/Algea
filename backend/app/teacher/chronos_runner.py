
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
        return {}

    def infer_batch(self, symbols: List[str], asof_date, load_fn) -> pd.DataFrame:
        """
        Infer priors for a batch of symbols.
        """
        return pd.DataFrame()
