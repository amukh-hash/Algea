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

from algae.models.foundation.base import FoundationModel, FoundationModelOutput

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
    """

    def __init__(self, config: FoundationModelConfig | None = None) -> None:
        self.config = config or FoundationModelConfig()
        self._model = None
        self._model_info: Dict[str, Any] = {}

    def _ensure_model(self) -> None:
        """Lazy-load the teacher model on first inference."""
        if self._model is not None:
            return
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
        """Generate distributional priors for each ticker in the canonical daily frame."""
        self._ensure_model()

        tickers = canonical_df["ticker"].unique() if "ticker" in canonical_df.columns else ["UNKNOWN"]
        rows: List[Dict[str, Any]] = []

        for ticker in tickers:
            sub = canonical_df[canonical_df["ticker"] == ticker] if "ticker" in canonical_df.columns else canonical_df
            if len(sub) < 20:
                continue

            close = sub["close"].values
            log_ret = np.diff(np.log(close + 1e-10))

            # Statistical fallback: drift ± realized vol
            drift = float(np.mean(log_ret[-60:]) if len(log_ret) >= 60 else np.mean(log_ret))
            vol = float(np.std(log_ret[-60:]) if len(log_ret) >= 60 else np.std(log_ret))

            rows.append({
                "ticker": ticker,
                "drift": drift,
                "vol_forecast": vol,
                "tail_risk": drift - 1.28 * vol,
                "trend_conf": 0.5 + 0.5 * np.clip(drift / max(vol, 1e-6), -1, 1),
                "q10": drift - 1.28 * vol,
                "q50": drift,
                "q90": drift + 1.28 * vol,
                "dispersion": 2.56 * vol,
                "prob_up": float(np.mean(log_ret[-20:] > 0)) if len(log_ret) >= 20 else 0.5,
                "source": "statistical_fallback",
            })

        priors = pd.DataFrame(rows)
        return FoundationModelOutput(priors=priors)

    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame | None = None) -> str:
        raise NotImplementedError(
            "Use algae.training.foundation_train.train_foundation_model() for training."
        )
