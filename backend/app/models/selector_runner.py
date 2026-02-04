import torch
import numpy as np
import os
import polars as pl
from typing import Dict, List, Optional
from datetime import date

from backend.app.models.rank_transformer import RankTransformer
from backend.app.models.selector_scaler import SelectorFeatureScaler
from backend.app.models.calibration import ScoreCalibrator
from backend.app.models.schema import FeatureContract
from backend.app.core import artifacts

class SelectorRunner:
    """
    Inference wrapper for Rank-Transformer.
    Loads checkpoint, scaler, calibrator, and runs inference on a batch of tickers.
    """
    def __init__(self, version: str = "v1", device: str = "cpu"):
        self.version = version
        self.device = torch.device(device)
        self.feature_cols = FeatureContract.CORE_FEATURES + FeatureContract.MARKET_FEATURES + FeatureContract.PRIOR_FEATURES

        # Resolve Artifacts
        checkpoint_path = artifacts.resolve_selector_checkpoint(version)
        scaler_path = artifacts.resolve_scaler_path(version)
        calib_path = artifacts.resolve_calibration_path(version)

        if not all([checkpoint_path, scaler_path, calib_path]):
            raise FileNotFoundError(f"Missing artifacts for Selector {version}")

        # Load Model
        self.model = RankTransformer(d_input=len(self.feature_cols)).to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Load Scaler
        self.scaler = SelectorFeatureScaler.load(scaler_path)

        # Load Calibrator
        self.calibrator = ScoreCalibrator.load(calib_path)

    def infer(self, features_dict: Dict[str, pl.DataFrame], lookback: int = 60) -> pl.DataFrame:
        """
        Runs inference for a batch of tickers.
        features_dict: {ticker: DataFrame with features}
        Returns: Leaderboard DataFrame
        """
        results = []

        # Prepare Batch
        # We process one ticker at a time for simplicity in inference loop,
        # or stack if all same length.
        # But features might vary in length?
        # Usually we take last N rows.

        batch_X = []
        tickers = []

        for ticker, df in features_dict.items():
            # Check length
            if len(df) < lookback:
                continue

            # Take last lookback
            # Ensure columns order
            try:
                # Convert to numpy
                # Check for missing cols
                missing = [c for c in self.feature_cols if c not in df.columns]
                if missing:
                    # Log warning?
                    continue

                x_vals = df.tail(lookback).select(self.feature_cols).to_numpy()
                batch_X.append(x_vals)
                tickers.append(ticker)
            except Exception as e:
                # print(f"Error prep {ticker}: {e}")
                continue

        if not batch_X:
            return pl.DataFrame()

        # Stack: [B, T, F]
        X_tensor = torch.tensor(np.stack(batch_X), dtype=torch.float32).to(self.device)

        # Scale
        # Note: Scaler handles tensor on device
        X_scaled = self.scaler.transform(X_tensor)

        # Forward
        with torch.no_grad():
            out = self.model(X_scaled)
            scores = out["score"].cpu().numpy().ravel() # [B]
            # Optional aux heads
            # p_up = out["p_up"].cpu().numpy().ravel()

        # Calibrate
        evs = self.calibrator.predict(scores)

        # Build Results
        for i, ticker in enumerate(tickers):
            results.append({
                "ticker": ticker,
                "score": float(scores[i]),
                "ev_10d": float(evs[i])
            })

        # Sort by Score Descending
        df = pl.DataFrame(results).sort("score", descending=True)

        # Add Rank
        df = df.with_columns([
            pl.col("score").rank(method="ordinal", descending=True).alias("rank"),
            (pl.col("score").rank(method="ordinal", descending=True) / len(df)).alias("rank_pct")
        ])

        return df
