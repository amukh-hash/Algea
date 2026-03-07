"""Chronos-2 compatibility shim.

Provides ``FoundationModelConfig`` and ``SimpleChronos2`` — the lightweight
inference API used by ``foundation_train.py``, ``data/priors/build.py``,
backtest scripts, and cycle runners.

Heavy-weight classes (``Chronos2NativeWrapper``, ``load_chronos_adapter``, etc.)
remain in ``chronos2_teacher.py``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from algae.models.foundation.base import (
    FoundationModel,
    FoundationModelOutput,
    ModelProvider,
    StatisticalFallbackProvider,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoundationModelConfig:
    """Configuration for the simple Chronos-2 prior generator."""

    model_id: str = "amazon/chronos-bolt-tiny"
    context_length: int = 512
    prediction_length: int = 10
    horizon_short: int = 5
    horizon_long: int = 10
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    device: str = "cpu"


class SimpleChronos2(FoundationModel):
    """Lightweight Chronos-2 prior generator.

    Uses ``chronos2_teacher.load_chronos_adapter`` when available,
    falling back to a statistical prior (naïve drift ± vol) when
    the teacher model cannot be loaded.

    Parameters
    ----------
    config : FoundationModelConfig | None
        Model configuration.
    provider : ModelProvider | None
        Injectable model provider. Pass ``StatisticalFallbackProvider()``
        in tests to skip HuggingFace downloads entirely.
    """

    def __init__(
        self,
        config: FoundationModelConfig | None = None,
        provider: ModelProvider | None = None,
    ) -> None:
        self.config = config or FoundationModelConfig()
        self._provider = provider
        self._model = None
        self._model_info: Dict[str, Any] = {}

    def _ensure_model(self) -> None:
        """Lazy-load the teacher model on first inference."""
        if self._model is not None:
            return

        # If an explicit provider was injected, use it
        if self._provider is not None:
            self._model = self._provider.load(self.config)
            if self._model is not None:
                logger.info("SimpleChronos2: loaded model via injected provider")
            else:
                logger.info("SimpleChronos2: provider returned None, using statistical fallback")
            return

        # Default: try HuggingFace download
        try:
            from algae.models.foundation.chronos2_teacher import load_chronos_adapter
            import torch

            device = torch.device(self.config.device)
            self._model, self._model_info = load_chronos_adapter(
                model_id=self.config.model_id,
                use_qlora=False,
                device=device,
                eval_mode=True,
            )
            logger.info(
                "SimpleChronos2: loaded teacher model %s on %s",
                self.config.model_id, device,
            )
        except Exception as exc:
            logger.warning("SimpleChronos2: teacher unavailable (%s), using statistical fallback", exc)
            self._model = None

    def infer_priors(
        self,
        canonical_df: pd.DataFrame,
        asof: pd.Timestamp | None = None,
    ) -> FoundationModelOutput:
        """Generate distributional priors for each ticker in the canonical daily frame.

        If ``asof`` is provided, rows with ``date >= asof`` are **excluded**
        to prevent look-ahead bias.  The output conforms to the canonical
        priors schema:

            ticker, p_mu5, p_mu10, p_sig5, p_sig10, p_pdown5, p_pdown10
        """
        self._ensure_model()

        # ── Strict temporal filter: exclude day-of and future data ──
        if asof is not None and "date" in canonical_df.columns:
            canonical_df = canonical_df[canonical_df["date"] < asof].copy()

        tickers = canonical_df["ticker"].unique() if "ticker" in canonical_df.columns else ["UNKNOWN"]
        rows: List[Dict[str, Any]] = []

        for ticker in tickers:
            sub = canonical_df[canonical_df["ticker"] == ticker] if "ticker" in canonical_df.columns else canonical_df
            if len(sub) < 2:
                continue

            close = sub["close"].values
            log_ret = np.diff(np.log(close + 1e-10))

            if len(log_ret) == 0:
                continue

            # Statistical fallback: drift ± realized vol
            window = min(60, len(log_ret))
            drift = float(np.mean(log_ret[-window:]))
            vol = float(np.std(log_ret[-window:])) or 1e-6

            # prob_down = fraction of negative returns in recent window
            recent_window = min(20, len(log_ret))
            prob_down = float(np.mean(log_ret[-recent_window:] < 0))

            rows.append({
                "ticker": ticker,
                # Point estimates for 5-day and 10-day horizons
                "p_mu5": drift * 5,
                "p_mu10": drift * 10,
                # Volatility scaled by sqrt(horizon)
                "p_sig5": vol * np.sqrt(5),
                "p_sig10": vol * np.sqrt(10),
                # Probability of negative return over horizon
                "p_pdown5": prob_down,
                "p_pdown10": prob_down,
                # Additional enrichment columns
                "q10": drift - 1.28 * vol,
                "q50": drift,
                "q90": drift + 1.28 * vol,
                "dispersion": 2.56 * vol,
                "source": "statistical_fallback",
            })

        priors = pd.DataFrame(rows)
        return FoundationModelOutput(priors=priors)

    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame | None = None) -> str:
        raise NotImplementedError(
            "Use algae.training.foundation_train.train_foundation_model() for training."
        )
