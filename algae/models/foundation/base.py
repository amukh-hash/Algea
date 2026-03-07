from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import pandas as pd


# ── Canonical priors schema ──────────────────────────────────────────
PRIORS_REQUIRED_COLUMNS = (
    "ticker", "p_mu5", "p_mu10", "p_sig5", "p_sig10",
    "p_pdown5", "p_pdown10",
)


@dataclass(frozen=True)
class FoundationModelOutput:
    priors: pd.DataFrame


@runtime_checkable
class ModelProvider(Protocol):
    """Injectable provider for foundation model loading.

    Production uses HuggingFaceProvider (downloads from HF Hub).
    Tests use StatisticalFallbackProvider (no network, deterministic).
    """

    def load(self, config: Any) -> Any:
        """Return a loaded model object, or None for statistical fallback."""
        ...


class StatisticalFallbackProvider:
    """Always returns None — forces the statistical fallback path.

    Use this in tests to avoid HuggingFace downloads while still
    exercising the full priors pipeline.
    """

    def load(self, config: Any) -> None:
        return None


class FoundationModel(ABC):
    @abstractmethod
    def infer_priors(self, canonical_df: pd.DataFrame, asof: pd.Timestamp | None = None) -> FoundationModelOutput:
        raise NotImplementedError

    @abstractmethod
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame | None = None) -> Any:
        raise NotImplementedError
