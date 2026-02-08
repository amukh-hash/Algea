import random
import polars as pl
from typing import Dict
from backend.app.models.signal_types import ModelSignal, ModelMetadata
from backend.app.core.config import OPTIONS_SEED

class MockStudentRunner:
    def __init__(self, model_path: str = None, preproc_path: str = None, device: str = "cpu", seed: int = OPTIONS_SEED):
        self.seed = seed
        self.device = device
        self.metadata = ModelMetadata(
            model_version="mock_v1",
            preproc_id="mock_preproc_v1",
            training_start="2020-01-01",
            training_end="2023-01-01"
        )
        
    def infer(self, df_window: pl.DataFrame) -> ModelSignal:
        # Generate signal based on seed + last price
        last_price = df_window.select("close").tail(1).item()
        ts_int = int(df_window.select("timestamp").tail(1).item().timestamp())
        
        # Condition on recent trend (simple drift)
        first_price = df_window.select("close").head(1).item()
        ret = (last_price - first_price) / first_price
        
        rng = random.Random(self.seed + ts_int + int(last_price))
        
        # Bullish or Bearish Bias
        bias = 0.05 if ret > 0 else -0.05
        
        horizons = ["1D", "3D"]
        quantiles = {}
        direction_probs = {}
        
        for h in horizons:
            mu = bias + rng.gauss(0, 0.01)
            sigma = 0.02
            
            # 5%, 50%, 95%
            q05 = mu - 1.645 * sigma
            q50 = mu
            q95 = mu + 1.645 * sigma
            
            quantiles[h] = {
                "0.05": q05,
                "0.50": q50,
                "0.95": q95
            }
            
            prob_up = 0.5 + (mu * 5) # Scale bias
            direction_probs[h] = max(0.1, min(0.9, prob_up))
            
        return ModelSignal(
            horizons=horizons,
            quantiles=quantiles,
            direction_probs=direction_probs,
            metadata=self.metadata
        )
