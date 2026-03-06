"""Strict Pydantic data contracts for the Inference Gateway boundary.

These DTOs enforce bounded fields, regex validation, and type-safe
serialisation across the ML platform ↔ orchestrator ↔ execution boundary.
Prevents mathematically impossible values (e.g. probability > 1.0, NaN
confidence) from entering the ECE database or order pipeline.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════════════════════
# Confidence Bin Utilities
# ═══════════════════════════════════════════════════════════════════════

# Canonical confidence bin edges
_BIN_EDGES = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
CONFIDENCE_BINS: list[str] = [
    f"{_BIN_EDGES[i]:.2f}-{_BIN_EDGES[i + 1]:.2f}"
    for i in range(len(_BIN_EDGES) - 1)
]
# e.g. ['0.50-0.60', '0.60-0.70', '0.70-0.80', '0.80-0.90', '0.90-1.00']


def assign_confidence_bin(probability: float) -> str:
    """Map a probability in [0, 1] to its canonical confidence bin string.

    Probabilities below 0.50 are assigned to the lowest bin.  Probabilities
    at the upper boundary (1.0) land in the highest bin.
    """
    if probability < _BIN_EDGES[0]:
        return CONFIDENCE_BINS[0]
    for i in range(len(_BIN_EDGES) - 1):
        if _BIN_EDGES[i] <= probability < _BIN_EDGES[i + 1]:
            return CONFIDENCE_BINS[i]
    return CONFIDENCE_BINS[-1]


def tensor_hex_hash(data: bytes) -> str:
    """Compute a short SHA-256 hex digest of raw tensor bytes for auditability."""
    return hashlib.sha256(data).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════
# Inference Gateway DTOs
# ═══════════════════════════════════════════════════════════════════════

class InferenceResponse(BaseModel):
    """Response contract from the Inference Gateway to the Orchestrator.

    Attributes
    ----------
    sleeve_id : str
        Identifier for the originating sleeve (``kronos``, ``mera``, ``vrp``).
    predicted_value : float
        Continuous point forecast (e.g. expected return, IV surface delta).
    predicted_probability : float
        Model self-reported confidence, strictly bounded [0.0, 1.0].
    confidence_bin : str
        Canonical bin string (e.g. ``'0.80-0.90'``).
    tensor_hash : str
        Hex digest of input state for auditability / replay.
    model_version : str
        Artifact version string of the model that produced this prediction.
    latency_ms : float
        Wall-clock inference latency in milliseconds.
    """

    sleeve_id: str = Field(..., min_length=1, max_length=50)
    predicted_value: float
    predicted_probability: float = Field(..., ge=0.0, le=1.0)
    confidence_bin: str = Field(..., pattern=r"^\d\.\d{2}-\d\.\d{2}$")
    tensor_hash: str = Field(..., min_length=8, max_length=64)
    model_version: str = Field(default="unknown")
    latency_ms: float = Field(default=0.0, ge=0.0)

    @field_validator("confidence_bin")
    @classmethod
    def _validate_bin_in_canonical(cls, v: str) -> str:
        if v not in CONFIDENCE_BINS:
            raise ValueError(
                f"confidence_bin '{v}' is not a canonical bin. "
                f"Allowed: {CONFIDENCE_BINS}"
            )
        return v


class OrderIntent(BaseModel):
    """Intent DTO passed from the Orchestrator to the Execution layer.

    Attributes
    ----------
    instrument : str
        Trading symbol/instrument identifier.
    target_weight : float
        Portfolio weight target, bounded [-1.0, 1.0].
    sleeve_id : str
        Originating sleeve for audit trail.
    risk_metadata : InferenceResponse
        Full inference response that produced this intent.
    urgency : int
        Priority level (1 = urgent live signal, 5 = background rebalance).
    """

    instrument: str = Field(..., min_length=1)
    target_weight: float = Field(..., ge=-1.0, le=1.0)
    sleeve_id: str = Field(..., min_length=1)
    risk_metadata: InferenceResponse
    urgency: int = Field(default=1, ge=1, le=10)


class FeatureVector(BaseModel):
    """Input features contract for model inference.

    Dimension assertions prevent silent shape mismatches from reaching
    the Transformer attention layers.
    """

    sleeve_id: str
    timestamp: str
    expected_dim: int = Field(..., gt=0)
    values: list[float]

    @field_validator("values")
    @classmethod
    def _validate_dim(cls, v: list[float], info) -> list[float]:
        expected = info.data.get("expected_dim")
        if expected is not None and len(v) != expected:
            raise ValueError(
                f"Feature vector has {len(v)} dimensions, expected {expected}"
            )
        return v



class DriftCheckResult(BaseModel):
    """Result of a LiveGuard concept drift check."""

    sleeve_id: str
    mmd_score: float = Field(..., ge=0.0)
    threshold: float = Field(..., ge=0.0)
    is_drifted: bool
    checked_at: str


class ECECheckResult(BaseModel):
    """Result of an ECE calibration check."""

    sleeve_id: str
    ece_score: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    n_samples: int = Field(..., ge=0)
    is_breached: bool
    high_confidence_ece: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class DAGStateUpdate(BaseModel):
    """Finite state machine transition for the DAG orchestrator."""

    run_id: str
    from_state: str
    to_state: str
    reason: Optional[str] = None

    @field_validator("to_state")
    @classmethod
    def _validate_state(cls, v: str) -> str:
        valid = {
            "PENDING",
            "INGESTING",
            "VALIDATING_DRIFT",
            "INFERRING",
            "VALIDATING_ECE",
            "EXECUTING",
            "COMPLETED",
            "HALTED_DRIFT",
            "HALTED_ECE_BREACH",
            "HALTED_OOM",
            "CRASHED",
        }
        if v not in valid:
            raise ValueError(f"Invalid DAG state: '{v}'. Valid: {valid}")
        return v
