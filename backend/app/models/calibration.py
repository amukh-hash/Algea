import numpy as np
import joblib
import os
from sklearn.isotonic import IsotonicRegression
from typing import Optional, Union

class ScoreCalibrator:
    """
    Maps raw model scores (ranking logits) to Expected Value (EV).
    Uses Isotonic Regression to ensure monotonicity.
    """
    def __init__(self, version: str = "v1"):
        self.version = version
        self.calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self.fitted = False

    def fit(self, scores: np.ndarray, targets: np.ndarray):
        """
        Fit mapping from Score -> Target (EV).
        scores: [N]
        targets: [N] (e.g., 10-day forward return)
        """
        # Ensure flat
        scores = np.asarray(scores).ravel()
        targets = np.asarray(targets).ravel()

        self.calibrator.fit(scores, targets)
        self.fitted = True
        return self

    def predict(self, scores: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        if not self.fitted:
            raise RuntimeError("Calibrator not fitted.")

        scalar = np.ndim(scores) == 0
        if scalar:
            scores = np.array([scores])
        else:
            scores = np.asarray(scores).ravel()

        ev = self.calibrator.predict(scores)

        if scalar:
            return float(ev[0])
        return ev

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "calibrator": self.calibrator,
            "version": self.version,
            "fitted": self.fitted
        }, path)

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        instance = cls(version=data.get("version", "unknown"))
        instance.calibrator = data["calibrator"]
        instance.fitted = data.get("fitted", True)
        return instance
