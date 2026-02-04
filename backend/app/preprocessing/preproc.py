import polars as pl
import numpy as np
import json
import os
import hashlib
from typing import Dict, List, Optional
from backend.app.preprocessing import versioning

def compute_hash(data: Dict) -> str:
    """Stable hash of dictionary"""
    s = json.dumps(data, sort_keys=True)
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

class Preprocessor:
    def __init__(self):
        self.params: Dict[str, Dict[str, float]] = {}
        # Keep minimal transforms for now, as SelectorFeatureScaler handles robust scaling downstream.
        # This Preprocessor is mainly for raw feature engineering (log returns, ratios, etc.)
        self.config = {
            "transforms": [
                {"col": "close", "type": "log_ret"},
                {"col": "volume", "type": "log1p_zscore"},
                {"col": "ad_line", "type": "zscore"},
                {"col": "bpi", "type": "zscore"}
            ]
        }
        self.fitted = False
        self.version_hash = None

    def fit(self, df: pl.DataFrame):
        """
        Fits parameters (mean, std) on the provided DataFrame.
        """
        params = {}
        
        # 1. Close -> Log Ret (No params needed for fit)
        
        # 2. Volume -> Log1p + Zscore
        vol_log = df.select(pl.col("volume").log1p())
        params["volume"] = {
            "mean": vol_log["volume"].mean(),
            "std": vol_log["volume"].std()
        }
        
        # 3. AD Line -> Zscore
        params["ad_line"] = {
            "mean": df["ad_line"].mean(),
            "std": df["ad_line"].std()
        }
        
        # 4. BPI -> Zscore
        params["bpi"] = {
            "mean": df["bpi"].mean(),
            "std": df["bpi"].std()
        }
        
        self.params = params
        self.fitted = True
        self.version_hash = compute_hash({"config": self.config, "params": self.params})

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Applies feature engineering.
        Returns DataFrame with engineering features + original timestamps.
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted")
        
        # Calculate Log Returns
        # We need to handle the shift properly.
        # Polars: col("close").log().diff() is log(today/yesterday)
        
        df = df.with_columns([
            (pl.col("close").log().diff().fill_null(0.0)).alias("log_return_1d"),
            (pl.col("close").log().diff(5).fill_null(0.0)).alias("log_return_5d"),
            (pl.col("close").log().diff(20).fill_null(0.0)).alias("log_return_20d"),

            # Volatility (20d rolling std of 1d log returns)
            (pl.col("close").log().diff().rolling_std(20).fill_null(0.0)).alias("volatility_20d"),

            # Volume Change
            (pl.col("volume").log1p().diff(5).fill_null(0.0)).alias("volume_log_change_5d"),

            # AD Line Trend
            (pl.col("ad_line").diff(5).fill_null(0.0)).alias("ad_line_trend_5d"),

            # BPI Level (passed through, maybe normalized)
            pl.col("bpi").alias("bpi_level")
        ])
        
        # Select only what we need for downstream + timestamp + ticker
        # We assume 'ticker' column exists or is handled by caller.
        keep_cols = ["timestamp", "log_return_1d", "log_return_5d", "log_return_20d",
                     "volatility_20d", "volume_log_change_5d", "ad_line_trend_5d", "bpi_level"]

        if "ticker" in df.columns:
            keep_cols.insert(0, "ticker")

        return df.select(keep_cols)

    def attach_teacher_priors(self, feature_df: pl.DataFrame, priors_df: pl.DataFrame) -> pl.DataFrame:
        """
        Joins precomputed teacher priors to the feature dataframe.
        priors_df must have: [date, ticker, drift_20d, vol_20d, downside_q10_20d, trend_conf_20d]
        feature_df must have: [timestamp, ticker, ...]
        """
        # Ensure timestamp alignment
        # priors_df 'date' should match feature_df 'timestamp' (date part)
        # We assume priors are computed EOD for that date.
        
        # Cast timestamp to date if needed for join
        # Or just join on timestamp if priors have full datetime?
        # Usually priors are daily.
        
        # Let's assume strict join on (ticker, timestamp).
        # We might need to cast feature_df timestamp to date.
        
        # Check columns
        required_priors = ["teacher_drift_20d", "teacher_vol_20d", "teacher_downside_q10_20d", "teacher_trend_conf_20d"]
        # If priors_df uses short names, rename them.

        # Rename mapping if needed
        rename_map = {}
        for col in ["drift_20d", "vol_20d", "downside_q10_20d", "trend_conf_20d"]:
            if col in priors_df.columns and f"teacher_{col}" not in priors_df.columns:
                rename_map[col] = f"teacher_{col}"

        if rename_map:
            priors_df = priors_df.rename(rename_map)

        # Join
        # Left join to keep features, fill nulls if missing priors?
        # Or inner join to enforce priors existence?
        # Plan says "attach".

        # We join on 'ticker' and 'timestamp'.
        # Ensure types match.

        # Helper to normalize date col
        if "date" in priors_df.columns and "timestamp" not in priors_df.columns:
             priors_df = priors_df.rename({"date": "timestamp"})

        # Cast to Date
        feature_df = feature_df.with_columns(pl.col("timestamp").cast(pl.Date).alias("join_date"))
        priors_df = priors_df.with_columns(pl.col("timestamp").cast(pl.Date).alias("join_date"))

        # Perform join
        joined = feature_df.join(priors_df, on=["ticker", "join_date"], how="left")

        # Fill nulls in priors?
        # For training, maybe we drop? For inference, we might forward fill or error?
        # Let's fill with 0 (neutral) but warn?
        # Better to let NaNs propagate and handle in scaler/preflight.

        # Drop temp join col
        joined = joined.drop("join_date")

        return joined

    def save(self, path: str):
        if not self.fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        data = {
            "config": self.config,
            "params": self.params,
            "version_hash": self.version_hash
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Preprocessor':
        with open(path, 'r') as f:
            data = json.load(f)
        
        obj = cls()
        obj.config = data["config"]
        obj.params = data["params"]
        obj.version_hash = data.get("version_hash", compute_hash({"config": obj.config, "params": obj.params}))
        obj.fitted = True
        return obj
