"""Kronos Foundation-Model adapter for futures sleeve.

Uses a foundation time-series model via API for zero-shot
Close-to-Open (r_oc) forecasting.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional — only needed for live API calls
try:
    import requests  # noqa: F401
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]


class KronosFoundationAdapter:
    """Adapter for a foundation time-series model API.

    Parameters
    ----------
    api_url : str
        Endpoint of the foundation model service.
    api_key : str
        Bearer token for authentication.
    context_length : int
        Maximum number of time-steps sent to the model.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        context_length: int = 512,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.context_length = context_length



    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_roc_distribution(
        self, raw_k_lines: np.ndarray
    ) -> Tuple[float, float]:
        """Get continuous zero-shot Close-to-Open (r_oc) forecast.

        Parameters
        ----------
        raw_k_lines : np.ndarray
            Shape ``(T, n_features)`` — raw OHLCV (or similar) data.
            The last ``context_length`` rows are used.

        Returns
        -------
        mu : float
            Predicted continuous forward return expected value.
        sigma : float
            Variance (sigma) of the prediction array.
        """
        context = raw_k_lines[-self.context_length :]

        payload = {
            "series": context.tolist(),
            "horizon": 1,
            "continuous_outputs": True,
        }

        # ----- Live API path (uncomment for production) ---------------
        # if requests is not None:
        #     resp = requests.post(
        #         self.api_url,
        #         json=payload,
        #         headers={"Authorization": f"Bearer {self.api_key}"},
        #         timeout=10,
        #     )
        #     resp.raise_for_status()
        #     data = resp.json()
        #     return (
        #         float(data["quantiles"]["0.5"]),
        #         float(data["quantiles"]["0.1"]),
        #         float(data["quantiles"]["0.9"]),
        #     )

        # ----- Mocked response for regression scaffolding ---------------
        logger.debug(
            "Kronos mock regression response — shape=%s, payload keys=%s",
            context.shape,
            list(payload.keys()),
        )
        # Simple trailing-mean and std as dummy placeholder for continuous output
        close_col = min(3, context.shape[1] - 1)
        expected_roc = float(np.mean(context[-5:, close_col]))
        variance_roc = float(np.std(context[-5:, close_col])) or 1e-6
        return expected_roc, variance_roc

    # ------------------------------------------------------------------
    # Position Sizing: Inverse-Volatility (Blind Spot 1 Mitigation)
    # ------------------------------------------------------------------

    @staticmethod
    def get_position_weight(
        prediction: float,
        trailing_5d_std: float,
        max_weight: float = 2.0,
    ) -> float:
        """Compute position weight decoupled from PatchTST magnitude.

        The model dictates trade *direction* only (sign of prediction).
        Historical 5-day trailing volatility dictates trade *size*.
        This prevents over-leveraging from incorrect magnitude estimates
        when DA > 0.50 but R² < 0.

        Parameters
        ----------
        prediction : float
            PatchTST continuous log-return forecast (ŷ_{t+1}).
        trailing_5d_std : float
            Standard deviation of the last 5 daily log-returns.
        max_weight : float
            Cap on absolute position weight to prevent extreme sizing.

        Returns
        -------
        float — Signed position weight: sign(pred) / σ_5d, clamped.
        """
        if trailing_5d_std < 1e-8:
            return 0.0
        raw_weight = np.sign(prediction) * (1.0 / trailing_5d_std)
        return float(np.clip(raw_weight, -max_weight, max_weight))

    @staticmethod
    def compute_trailing_volatility(
        log_returns: np.ndarray,
        window: int = 5,
    ) -> float:
        """Compute trailing volatility from recent log-returns.

        Parameters
        ----------
        log_returns : np.ndarray
            Array of recent daily log-returns.
        window : int
            Number of trailing days for the volatility window.

        Returns
        -------
        float — Standard deviation of the last `window` returns.
        """
        if len(log_returns) < window:
            return float(np.std(log_returns)) if len(log_returns) > 1 else 1e-6
        return float(np.std(log_returns[-window:]))
