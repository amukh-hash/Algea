"""Tests for R3: Promotion summary assembler."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from sleeves.cooc_reversal_futures.pipeline.types import (
    GateResult,
    PromotionSummary,
    ProviderInvarianceReport,
    SessionSemanticsReport,
    Tier2CalibrationReport,
    ValidationReport,
)
from sleeves.cooc_reversal_futures.pipeline.promotion_summary import (
    assemble_promotion_summary,
    persist_promotion_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_validation(gate_specs: list[tuple[str, bool]]) -> ValidationReport:
    """Create a ValidationReport from (name, passed) pairs."""
    gates = tuple(
        GateResult(name=name, passed=passed, detail=f"gate={name}")
        for name, passed in gate_specs
    )
    all_passed = all(p for _, p in gate_specs)
    return ValidationReport(
        all_passed=all_passed,
        gates=gates,
        baseline_ic=0.05,
        model_ic=0.10,
    )


def _all_pass_gates() -> list[tuple[str, bool]]:
    return [
        ("data_completeness", True),
        ("leakage_oracle", True),
        ("leakage_permutation", True),
        ("leakage_timestamp", True),
        ("model_sanity", True),
        ("strategy_polarity", True),
        ("ic_distribution", True),
        ("multi_seed_stability", True),
        ("stress_window", True),
        ("contiguous_oos", True),
    ]


# ---------------------------------------------------------------------------
# Tests: decision logic
# ---------------------------------------------------------------------------

class TestPromoteDecision:
    def test_all_pass_promotes(self):
        vr = _make_validation(_all_pass_gates())
        summary = assemble_promotion_summary(
            vr, run_id="test-run-001",
        )
        assert summary.decision == "PROMOTE"

    def test_hard_gate_fail_is_fail(self):
        specs = _all_pass_gates()
        # Fail a hard gate
        specs[0] = ("data_completeness", False)
        vr = _make_validation(specs)
        summary = assemble_promotion_summary(vr)
        assert summary.decision == "FAIL"

    def test_soft_gate_fail_is_hold(self):
        specs = _all_pass_gates()
        # Fail a soft gate
        specs[-1] = ("contiguous_oos", False)
        vr = _make_validation(specs)
        summary = assemble_promotion_summary(vr)
        assert summary.decision == "HOLD"

    def test_provider_invariance_downgrades(self):
        vr = _make_validation(_all_pass_gates())
        # Provider invariance fails
        pi = ProviderInvarianceReport(
            overall_consistent=False,
            flags=("ES: corr too low",),
        )
        summary = assemble_promotion_summary(vr, provider_invariance=pi)
        assert summary.decision == "HOLD"

    def test_provider_invariance_consistent_no_downgrade(self):
        vr = _make_validation(_all_pass_gates())
        pi = ProviderInvarianceReport(overall_consistent=True)
        summary = assemble_promotion_summary(vr, provider_invariance=pi)
        assert summary.decision == "PROMOTE"


# ---------------------------------------------------------------------------
# Tests: metadata propagation
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_feature_lists_propagated(self):
        vr = _make_validation(_all_pass_gates())
        summary = assemble_promotion_summary(
            vr,
            feature_list=["r_co", "r_oc", "vol_5d"],
            features_dropped=["bad_feat"],
        )
        assert summary.feature_list_used == ("r_co", "r_oc", "vol_5d")
        assert summary.features_dropped == ("bad_feat",)

    def test_hashes_and_seeds(self):
        vr = _make_validation(_all_pass_gates())
        summary = assemble_promotion_summary(
            vr,
            config_hash="abc123",
            data_version_hash="def456",
            seed_list=[42, 123, 7],
        )
        assert summary.config_hash == "abc123"
        assert summary.data_version_hash == "def456"
        assert summary.seed_list == (42, 123, 7)

    def test_tier2_gates_extracted(self):
        vr = _make_validation(_all_pass_gates())
        t2_cal = Tier2CalibrationReport(
            ladder={
                "summary": {
                    "tier0_sharpe": 2.0,
                    "tier1_sharpe": 1.8,
                    "tier2_sharpe": 1.5,
                    "cost_erosion_t0_t1": 0.2,
                    "cost_erosion_t1_t2": 0.3,
                },
            },
        )
        summary = assemble_promotion_summary(vr, tier2_calibration=t2_cal)
        assert summary.tier2_gates["tier2_sharpe"] == 1.5


# ---------------------------------------------------------------------------
# Tests: persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_writes_json_and_md(self):
        vr = _make_validation(_all_pass_gates())
        summary = assemble_promotion_summary(vr, run_id="persist-test")
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = persist_promotion_summary(summary, tmpdir)
            assert json_path.exists()
            md_path = Path(tmpdir) / "promotion_summary.md"
            assert md_path.exists()

    def test_json_round_trip(self):
        vr = _make_validation(_all_pass_gates())
        summary = assemble_promotion_summary(
            vr, run_id="rt-test", config_hash="xyz",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = persist_promotion_summary(summary, tmpdir)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            assert data["decision"] == "PROMOTE"
            assert data["run_id"] == "rt-test"
            assert data["config_hash"] == "xyz"

    def test_to_dict_serializable(self):
        vr = _make_validation(_all_pass_gates())
        pi = ProviderInvarianceReport(overall_consistent=True)
        summary = assemble_promotion_summary(vr, provider_invariance=pi)
        d = summary.to_dict()
        # Should be JSON-serializable
        json.dumps(d, default=str)  # no TypeError
        assert "provider_invariance" in d
        assert d["decision"] == "PROMOTE"
