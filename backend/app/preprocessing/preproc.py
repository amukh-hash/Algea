import polars as pl
import numpy as np
import json
import os
from typing import Dict, List, Optional
from backend.app.preprocessing import versioning

class Preprocessor:
    def __init__(self):
        self.params: Dict[str, Dict[str, float]] = {}
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

        # 1. Close -> Log Ret (No params needed for fit, but we need to know we did it?)
        # Actually log ret is stateless.

        # 2. Volume -> Log1p + Zscore
        # We need mean/std of log1p(volume)
        vol_log = df.select(pl.col("volume").log1p())
        params["volume"] = {
            "mean": vol_log.select(pl.mean("volume")).item(),
            "std": vol_log.select(pl.std("volume")).item()
        }

        # 3. AD Line -> Zscore
        params["ad_line"] = {
            "mean": df.select(pl.mean("ad_line")).item(),
            "std": df.select(pl.std("ad_line")).item()
        }

        # 4. BPI -> Zscore
        params["bpi"] = {
            "mean": df.select(pl.mean("bpi")).item(),
            "std": df.select(pl.std("bpi")).item()
        }

        self.params = params
        self.fitted = True
        self._update_hash()

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.fitted:
            raise ValueError("Preprocessor not fitted")

        # Ensure input columns exist
        # We will build a list of expressions
        exprs = [pl.col("timestamp")] # Keep timestamp

        # Close -> Log Returns
        # log(close) - log(close.shift(1))
        # Note: This introduces a null at the first position.
        # We should handle it (fill 0 or drop).
        # We'll fill 0 for now to keep alignment.
        exprs.append(
            (pl.col("close").log() - pl.col("close").log().shift(1)).fill_null(0.0).alias("log_ret").cast(pl.Float32)
        )

        # Volume -> Log1p -> Zscore
        mu_vol = self.params["volume"]["mean"]
        sigma_vol = self.params["volume"]["std"] if self.params["volume"]["std"] > 1e-9 else 1.0
        exprs.append(
            ((pl.col("volume").log1p() - mu_vol) / sigma_vol).alias("volume_norm").cast(pl.Float32)
        )

        # AD Line -> Zscore
        mu_ad = self.params["ad_line"]["mean"]
        sigma_ad = self.params["ad_line"]["std"] if self.params["ad_line"]["std"] > 1e-9 else 1.0
        exprs.append(
            ((pl.col("ad_line") - mu_ad) / sigma_ad).alias("ad_line_norm").cast(pl.Float32)
        )

        # BPI -> Zscore
        mu_bpi = self.params["bpi"]["mean"]
        sigma_bpi = self.params["bpi"]["std"] if self.params["bpi"]["std"] > 1e-9 else 1.0
        exprs.append(
            ((pl.col("bpi") - mu_bpi) / sigma_bpi).alias("bpi_norm").cast(pl.Float32)
        )

        return df.select(exprs)

    def _update_hash(self):
        # Hash config + params
        data = {
            "config": self.config,
            "params": self.params
        }
        self.version_hash = versioning.compute_hash(data)

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
        obj.version_hash = data["version_hash"]
        obj.fitted = True

        # Verify hash
        current_hash = versioning.compute_hash({"config": obj.config, "params": obj.params})
        if current_hash != obj.version_hash:
            raise ValueError(f"Preprocessor hash mismatch. File corrupted or manually edited. {current_hash} != {obj.version_hash}")

        return obj
