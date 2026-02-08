"""
Live Selector Logic for Rank Selector V2.
Handles point-in-time feature generation, ranking, and risk control.
"""
import logging
import pandas as pd
import polars as pl
import torch
import numpy as np
from typing import List, Dict, Optional
from datetime import date

from backend.app.data.universe_api import get_universe_mask  # Assuming this exists or similar
from backend.app.features.selector_features_v2 import SelectorFeatureBuilder, SelectorFeatureConfig
from backend.app.models.selector_v2 import TwoHeadRankSelector

logger = logging.getLogger(__name__)

class LiveSelector:
    def __init__(self, 
                 model_path: str,
                 feature_config: SelectorFeatureConfig = SelectorFeatureConfig(),
                 min_breadth_live: int = 150,
                 p_trade_threshold: float = 0.55,
                 max_candidates: int = 10,
                 device: str = "cpu"):
        
        self.config = feature_config
        self.min_breadth_live = min_breadth_live
        self.p_trade_threshold = p_trade_threshold
        self.max_candidates = max_candidates
        self.device = device
        
        # Load Model
        self.model = TwoHeadRankSelector()
        state = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.to(device)
        self.model.eval()
        
        # Re-use builder logic for normalization
        # But we need a specialized "live" builder that takes DataFrame input?
        # Or we use the Builder class but mock the IO?
        self.builder = SelectorFeatureBuilder(feature_config)

    def select_candidates(self, 
                          today: date,
                          ohlcv_history: pd.DataFrame, # Pandas DataFrame of recent history (enough for rolling)
                          universe_today: pd.DataFrame # Today's universe (ticker, is_tradable)
                          ) -> List[Dict]:
        """
        Generate selection for 'today'.
        """
        logger.info(f"Running Live Selection for {today}")
        
        # 1. Feature Generation (Live)
        # Convert to Polars
        ohlcv_pl = pl.from_pandas(ohlcv_history).lazy()
        
        # We need to ensure we have enough history for the rolling windows ending on 'today'
        # The Builder computes rolling on full history.
        # We assume ohlcv_history contains [T-Lookback, T]
        
        # Compute Raw Features
        # This returns LazyFrame
        raw_features = self.builder._compute_raw_features(ohlcv_pl)
        
        # Filter for TODAY only (after rolling)
        # We need raw features for today
        # Note: Raw features computation (shifting) might need sorting.
        
        # Collect
        df_feats = raw_features.filter(pl.col("date") == today).collect()
        
        if len(df_feats) == 0:
            logger.warning(f"No features generated for {today}. Check data history.")
            return []

        # 2. Join Universe & Filter Tradable
        # universe_today: [ticker, is_tradable, tier...]
        u_pl = pl.from_pandas(universe_today)
        
        # Join
        # Ensure 'ticker' col
        df = df_feats.join(u_pl, on="ticker", how="inner").filter(pl.col("is_tradable") == True)
        
        # 3. Breadth Check
        N_t = len(df)
        logger.info(f"Tradable Universe Size: {N_t}")
        
        if N_t < self.min_breadth_live:
            logger.warning(f"Risk-Off: Breadth {N_t} < {self.min_breadth_live}. Reducing exposure.")
            # Logic: Cap K to 3? Or return empty?
            # User spec: "trigger risk-off: either no trades, or cap K to 3 and reduce exposure."
            # We will cap K=3.
            current_k = min(self.max_candidates, 3)
        else:
            current_k = self.max_candidates

        # 4. Rank Normalization
        # Apply normalization on this single day
        df_norm = self.builder._apply_rank_normalization(df)
        
        # 5. Inference
        feature_cols = ["x_lr1", "x_lr5", "x_lr20", "x_vol", "x_relvol"]
        X = df_norm.select(feature_cols).to_numpy().astype(np.float32)
        X_tensor = torch.tensor(X).unsqueeze(0).to(self.device) # [1, N, F]
        
        with torch.no_grad():
            scores, p_trades = self.model(X_tensor)
            
        scores = scores.squeeze(0).cpu().numpy() # [N]
        p_trades = p_trades.squeeze(0).cpu().numpy() # [N]
        
        # 6. Selection Logic
        results = []
        tickers = df_norm["ticker"].to_list()
        tiers = df_norm["tier"].to_list() if "tier" in df_norm.columns else ["C"]*N_t
        
        for i in range(N_t):
            res = {
                "ticker": tickers[i],
                "score": float(scores[i]),
                "p_trade": float(p_trades[i]),
                "tier": tiers[i]
            }
            
            # Filter p_trade
            if res["p_trade"] >= self.p_trade_threshold:
                results.append(res)
                
        # Rank by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Top-K
        selected = results[:current_k]
        
        logger.info(f"Selected {len(selected)} candidates (from {len(results)} qualifying).")
        return selected

if __name__ == "__main__":
    pass
