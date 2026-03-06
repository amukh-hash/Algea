"""
Lag-Llama zero-shot inference — deterministic quantile forecasts.

Loads pre-trained Lag-Llama from HuggingFace, runs seeded inference
on derived risk series, and converts per-step quantiles to aggregated
10-day RV quantiles.

Depends on: lag-llama, gluonts, torch.
If deps unavailable, raises ImportError at load time with clear message.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from algae.models.tsfm.lag_llama.config import LagLlamaConfig


# Dependency guard
_DEPS_AVAILABLE = True
_IMPORT_ERROR = ""
try:
    import torch
except ImportError as e:
    _DEPS_AVAILABLE = False
    _IMPORT_ERROR = f"torch not installed: {e}"


def _check_deps() -> None:
    if not _DEPS_AVAILABLE:
        raise ImportError(
            f"Lag-Llama dependencies not available: {_IMPORT_ERROR}. "
            "Install with: pip install lag-llama gluonts"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Forecast result
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ForecastResult:
    """Output of a single Lag-Llama inference run."""
    as_of_date: str
    underlying: str
    series_type: str
    quantiles: Dict[float, float]          # {0.50: rv_val, 0.90: ..., ...}
    model_id: str
    inference_seed: int
    health_score: float = 1.0              # set by hardening
    is_fallback: bool = False              # True if baseline-only
    raw_sqret_quantiles: Dict[float, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of_date": self.as_of_date,
            "underlying": self.underlying,
            "series_type": self.series_type,
            "quantiles": {str(k): v for k, v in self.quantiles.items()},
            "model_id": self.model_id,
            "inference_seed": self.inference_seed,
            "health_score": self.health_score,
            "is_fallback": self.is_fallback,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Forecaster
# ═══════════════════════════════════════════════════════════════════════════

class LagLlamaForecaster:
    """Zero-shot probabilistic risk forecaster using Lag-Llama.

    In zero-shot mode, loads the pre-trained model from HuggingFace
    and runs deterministic (seeded) inference on derived risk series.
    """

    def __init__(self, config: LagLlamaConfig) -> None:
        self.config = config
        self._model = None
        self._pipeline = None

    def _load_model(self) -> None:
        """Lazy-load model from HuggingFace."""
        _check_deps()
        try:
            from huggingface_hub import hf_hub_download
            from lag_llama.gluon.estimator import LagLlamaEstimator
        except ImportError:
            # Graceful: mark as unavailable
            self._model = None
            return

        ckpt_path = hf_hub_download(
            repo_id=self.config.model_id,
            filename="lag-llama.ckpt",
        )

        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=self.config.prediction_length,
            context_length=self.config.context_length,
            input_size=1,
            augmentations=0,
            nonnegative_pred_samples=True,
        )
        predictor = estimator.create_lightning_predictor(
            device=torch.device(self.config.device if torch.cuda.is_available() else "cpu"),
        )
        self._pipeline = predictor

    def forecast(
        self,
        context: np.ndarray,
        as_of_date: _date,
        underlying: str,
        series_type: str = "sqret",
    ) -> ForecastResult:
        """Run zero-shot inference and return RV quantiles.

        Parameters
        ----------
        context : 1-D array of len >= context_length
        as_of_date : date for this forecast
        underlying : e.g. "SPY"
        series_type : target series name

        Returns
        -------
        ForecastResult with rv10 quantiles derived from sqret forecast.
        """
        _check_deps()

        if self._pipeline is None:
            self._load_model()

        if self._pipeline is None:
            # Model unavailable — return fallback
            return ForecastResult(
                as_of_date=as_of_date.isoformat(),
                underlying=underlying,
                series_type=series_type,
                quantiles={q: 0.15 for q in self.config.quantiles},
                model_id=self.config.model_id,
                inference_seed=self.config.seed,
                is_fallback=True,
            )

        # Set seed for determinism
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # Prepare GluonTS-style dataset
        from gluonts.dataset.common import ListDataset
        import pandas as pd

        ds = ListDataset(
            [{"start": pd.Timestamp("2000-01-01"), "target": context}],
            freq="B",
        )

        # Run inference
        forecasts = list(self._pipeline.predict(ds, num_samples=self.config.num_samples))
        if not forecasts:
            return ForecastResult(
                as_of_date=as_of_date.isoformat(),
                underlying=underlying,
                series_type=series_type,
                quantiles={q: 0.15 for q in self.config.quantiles},
                model_id=self.config.model_id,
                inference_seed=self.config.seed,
                is_fallback=True,
            )

        samples = forecasts[0].samples  # shape: (num_samples, prediction_length)

        # Aggregate: sum sqret over prediction horizon → 10-day variance proxy
        # Then convert to annualised rv: rv = sqrt(252 * mean(sqret_daily))
        sum_sqret_per_sample = samples.sum(axis=1)  # (num_samples,)
        mean_sqret_per_sample = samples.mean(axis=1)

        # Compute quantiles of annualised RV
        rv_samples = np.sqrt(252.0 * np.maximum(mean_sqret_per_sample, 0.0))

        raw_sqret_quantiles = {}
        rv_quantiles = {}
        for q in self.config.quantiles:
            sqret_q = float(np.percentile(mean_sqret_per_sample, q * 100))
            raw_sqret_quantiles[q] = sqret_q
            rv_q = float(np.sqrt(252.0 * max(sqret_q, 0.0)))
            rv_quantiles[q] = rv_q

        return ForecastResult(
            as_of_date=as_of_date.isoformat(),
            underlying=underlying,
            series_type=series_type,
            quantiles=rv_quantiles,
            model_id=self.config.model_id,
            inference_seed=self.config.seed,
            raw_sqret_quantiles=raw_sqret_quantiles,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Baseline-only forecaster (always available)
# ═══════════════════════════════════════════════════════════════════════════

class BaselineForecaster:
    """EWMA-based deterministic RV forecaster (no ML dependencies)."""

    def __init__(self, config: LagLlamaConfig) -> None:
        self.config = config

    def forecast(
        self,
        close: np.ndarray,
        as_of_date: _date,
        underlying: str,
    ) -> ForecastResult:
        """Compute EWMA RV quantiles from trailing distribution."""
        if len(close) < 30:
            return ForecastResult(
                as_of_date=as_of_date.isoformat(),
                underlying=underlying,
                series_type="ewma_baseline",
                quantiles={q: 0.15 for q in self.config.quantiles},
                model_id="ewma_baseline",
                inference_seed=0,
                is_fallback=True,
            )

        # Log returns
        lr = np.diff(np.log(close))
        sqret = lr ** 2

        # EWMA variance
        alpha = 2.0 / (self.config.ewma_span + 1)
        ewma_var = np.zeros_like(sqret)
        ewma_var[0] = sqret[0]
        for i in range(1, len(sqret)):
            ewma_var[i] = alpha * sqret[i] + (1 - alpha) * ewma_var[i - 1]

        # Annualised RV from trailing EWMA
        rv_trail = np.sqrt(252.0 * ewma_var)
        # Use empirical quantiles from trailing 252 days
        lookback = min(252, len(rv_trail))
        tail = rv_trail[-lookback:]

        rv_quantiles = {}
        for q in self.config.quantiles:
            rv_quantiles[q] = float(np.percentile(tail, q * 100))

        return ForecastResult(
            as_of_date=as_of_date.isoformat(),
            underlying=underlying,
            series_type="ewma_baseline",
            quantiles=rv_quantiles,
            model_id="ewma_baseline",
            inference_seed=0,
            is_fallback=True,
        )
