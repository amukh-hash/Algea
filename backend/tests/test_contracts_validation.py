"""Test Pydantic contract validation.

Validates that DTOs reject out-of-bounds probabilities, invalid
confidence bins, and malformed DAG state transitions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.schemas.contracts import (
    CONFIDENCE_BINS,
    DAGStateUpdate,
    ECECheckResult,
    FeatureVector,
    InferenceResponse,
    OrderIntent,
    assign_confidence_bin,
)


class TestInferenceResponse:
    def test_valid_response(self):
        resp = InferenceResponse(
            sleeve_id="kronos",
            predicted_value=0.023,
            predicted_probability=0.85,
            confidence_bin="0.80-0.90",
            tensor_hash="abcdef1234567890",
        )
        assert resp.predicted_probability == 0.85

    def test_probability_out_of_bounds_high(self):
        with pytest.raises(Exception):
            InferenceResponse(
                sleeve_id="kronos",
                predicted_value=0.0,
                predicted_probability=1.5,  # > 1.0
                confidence_bin="0.90-1.00",
                tensor_hash="abcdef12",
            )

    def test_probability_out_of_bounds_low(self):
        with pytest.raises(Exception):
            InferenceResponse(
                sleeve_id="kronos",
                predicted_value=0.0,
                predicted_probability=-0.1,  # < 0.0
                confidence_bin="0.50-0.60",
                tensor_hash="abcdef12",
            )

    def test_invalid_confidence_bin_format(self):
        with pytest.raises(Exception):
            InferenceResponse(
                sleeve_id="kronos",
                predicted_value=0.0,
                predicted_probability=0.7,
                confidence_bin="invalid",  # bad format
                tensor_hash="abcdef12",
            )

    def test_noncanonical_bin_rejected(self):
        with pytest.raises(Exception):
            InferenceResponse(
                sleeve_id="kronos",
                predicted_value=0.0,
                predicted_probability=0.7,
                confidence_bin="0.55-0.65",  # valid format but not canonical
                tensor_hash="abcdef12",
            )


class TestOrderIntent:
    def test_valid_intent(self):
        resp = InferenceResponse(
            sleeve_id="mera", predicted_value=0.01,
            predicted_probability=0.75, confidence_bin="0.70-0.80",
            tensor_hash="hash12345678",
        )
        intent = OrderIntent(
            instrument="AAPL", target_weight=0.05,
            sleeve_id="mera", risk_metadata=resp,
        )
        assert intent.target_weight == 0.05

    def test_weight_out_of_bounds(self):
        resp = InferenceResponse(
            sleeve_id="mera", predicted_value=0.01,
            predicted_probability=0.75, confidence_bin="0.70-0.80",
            tensor_hash="hash12345678",
        )
        with pytest.raises(Exception):
            OrderIntent(
                instrument="AAPL", target_weight=1.5,  # > 1.0
                sleeve_id="mera", risk_metadata=resp,
            )


class TestConfidenceBinAssignment:
    def test_low_probability(self):
        assert assign_confidence_bin(0.55) == "0.50-0.60"

    def test_high_probability(self):
        assert assign_confidence_bin(0.95) == "0.90-1.00"

    def test_boundary_probability(self):
        assert assign_confidence_bin(0.80) == "0.80-0.90"

    def test_below_range(self):
        assert assign_confidence_bin(0.3) == "0.50-0.60"

    def test_at_one(self):
        assert assign_confidence_bin(1.0) == "0.90-1.00"


class TestFeatureVector:
    def test_dimension_mismatch(self):
        with pytest.raises(Exception):
            FeatureVector(
                sleeve_id="kronos",
                timestamp="2025-01-01",
                values=[1.0, 2.0, 3.0],
                expected_dim=5,  # mismatch
            )

    def test_dimension_match(self):
        fv = FeatureVector(
            sleeve_id="kronos",
            timestamp="2025-01-01",
            values=[1.0, 2.0, 3.0],
            expected_dim=3,
        )
        assert len(fv.values) == 3


class TestDAGStateUpdate:
    def test_valid_transition(self):
        update = DAGStateUpdate(
            run_id="test", from_state="PENDING", to_state="INGESTING"
        )
        assert update.to_state == "INGESTING"

    def test_invalid_state(self):
        with pytest.raises(Exception, match="Invalid DAG state"):
            DAGStateUpdate(
                run_id="test", from_state="PENDING", to_state="INVALID_STATE"
            )


class TestECECheckResult:
    def test_valid_result(self):
        result = ECECheckResult(
            sleeve_id="kronos", ece_score=0.05,
            n_samples=100, is_breached=False,
        )
        assert result.is_breached is False

    def test_ece_out_of_bounds(self):
        with pytest.raises(Exception):
            ECECheckResult(
                sleeve_id="kronos", ece_score=1.5,  # > 1.0
                n_samples=100, is_breached=False,
            )
