import joblib
import pandas as pd
from typing import List
from sklearn.preprocessing import RobustScaler, StandardScaler

class SelectorScaler:
    """
    Wraps Scikit-Learn scalers for the Ranking Model features.
    """
    def __init__(self, method: str = 'robust'):
        self.method = method
        if method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
            
        self.feature_cols = [
            'log_return_1d',
            'log_return_5d',
            'log_return_20d',
            'volatility_20d',
            'relative_volume_20d'
        ]

    def fit(self, df: pd.DataFrame):
        """Fits on training data."""
        # Check cols
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for scaling: {missing}")
            
        self.scaler.fit(df[self.feature_cols])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns dataframe with scaled columns (inplace or copy)."""
        df_scaled = df.copy()
        vals = self.scaler.transform(df[self.feature_cols])
        
        # Replace columns
        for i, col in enumerate(self.feature_cols):
            df_scaled[col] = vals[:, i]
            
        return df_scaled
    
    def save(self, path: str):
        joblib.dump(self, path)
        
    @staticmethod
    def load(path: str):
        return joblib.load(path)
