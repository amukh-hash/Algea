"""
Score calibrator — maps raw ranking scores to Expected Value via Isotonic Regression.

Ported from deprecated/backend_app_snapshot/models/calibration.py.
"""
from __future__ import annotations

import os
from typing import Union

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression


class ScoreCalibrator:
    """Maps raw model scores to EV while ensuring monotonicity."""

    def __init__(self, version: str = "v1") -> None:
        self.version = version
        self.calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self.fitted = False

    # ------------------------------------------------------------------
    def fit(self, scores: np.ndarray, targets: np.ndarray) -> "ScoreCalibrator":
        """Fit mapping Score → Target (EV).  Both arrays shape ``[N]``."""
        scores = np.asarray(scores).ravel()
        targets = np.asarray(targets).ravel()
        self.calibrator.fit(scores, targets)
        self.fitted = True
        return self

    def predict(self, scores: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        if not self.fitted:
            raise RuntimeError("Calibrator not fitted.")
        scalar = np.ndim(scores) == 0
        scores_arr = np.atleast_1d(np.asarray(scores)).ravel()
        ev = self.calibrator.predict(scores_arr)
        return float(ev[0]) if scalar else ev

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {"calibrator": self.calibrator, "version": self.version, "fitted": self.fitted},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "ScoreCalibrator":
        data = joblib.load(path)
        instance = cls(version=data.get("version", "unknown"))
        instance.calibrator = data["calibrator"]
        instance.fitted = data.get("fitted", True)
        return instance
