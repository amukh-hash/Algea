"""
Selector feature scaler — wraps scikit-learn scalers for the ranking model features.

Ported from deprecated/backend_app_snapshot/models/scalers.py.
"""
from __future__ import annotations

from typing import List

import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


class SelectorScaler:
    """Wraps a scikit-learn scaler for a fixed set of ranking-model feature columns."""

    DEFAULT_FEATURE_COLS: List[str] = [
        "log_return_1d",
        "log_return_5d",
        "log_return_20d",
        "volatility_20d",
        "relative_volume_20d",
    ]

    def __init__(self, method: str = "robust", feature_cols: List[str] | None = None) -> None:
        self.method = method
        self.scaler = RobustScaler() if method == "robust" else StandardScaler()
        self.feature_cols = feature_cols or list(self.DEFAULT_FEATURE_COLS)

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "SelectorScaler":
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for scaling: {missing}")
        self.scaler.fit(df[self.feature_cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_scaled = df.copy()
        df_scaled[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        return df_scaled

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "SelectorScaler":
        return joblib.load(path)
