from sklearn.preprocessing import RobustScaler, QuantileTransformer
import joblib
import numpy as np
import os
from typing import Optional, List, Union
import torch

class SelectorFeatureScaler:
    """
    Versioned, numeric-only scaler for the Selector Rank-Transformer.
    Uses RobustScaler (median/IQR) to handle outliers without clipping.
    Also supports serialization for reproducibility.
    """
    def __init__(self, version: str = "v1", feature_names: List[str] = None):
        self.version = version
        self.feature_names = feature_names
        self.scaler = RobustScaler(quantile_range=(5.0, 95.0))
        self.fitted = False

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fit scaler to training data X [N, Features] (flattened time if needed).
        """
        if feature_names:
            self.feature_names = feature_names

        # Ensure 2D
        if X.ndim == 3:
            B, T, F = X.shape
            X_flat = X.reshape(-1, F)
        else:
            X_flat = X

        self.scaler.fit(X_flat)
        self.fitted = True
        return self

    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Transform X using fitted scaler.
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted.")

        is_tensor = torch.is_tensor(X)
        if is_tensor:
            device = X.device
            X_np = X.cpu().numpy()
        else:
            X_np = X

        ndim = X_np.ndim
        if ndim == 3:
            B, T, F = X_np.shape
            X_flat = X_np.reshape(-1, F)
            X_scaled = self.scaler.transform(X_flat)
            X_out = X_scaled.reshape(B, T, F)
        else:
            X_out = self.scaler.transform(X_np)

        if is_tensor:
            return torch.tensor(X_out, dtype=torch.float32, device=device)
        return X_out

    def save(self, path: str):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "version": self.version,
            "fitted": self.fitted
        }, path)

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        instance = cls(version=data.get("version", "unknown"), feature_names=data.get("feature_names"))
        instance.scaler = data["scaler"]
        instance.fitted = data.get("fitted", True)
        return instance
