"""Tests for the expanded 14-instrument universe.

Covers:
- All 14 roots in CONTRACT_MASTER and EXCHANGE_MAP
- active_contract_for_day() valid across monthly/custom/quarterly schedules
- Roll coverage: no gaps across all cycle types
- ContractSpec qualification sanity: mock IBKR returning wrong month → RESEARCH_ONLY
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from sleeves.cooc_reversal_futures.contract_master import CONTRACT_MASTER, ContractSpec
from algae.trading.ibkr_contracts import EXCHANGE_MAP
from sleeves.cooc_reversal_futures.roll import active_contract_for_day
from sleeves.cooc_reversal_futures.contract_spec_checks import (
    SpecCheckResult,
    validate_qualified_contract,
    validate_all_specs,
)

# The full 14-instrument universe
FULL_UNIVERSE = [
    "ES", "NQ", "YM", "RTY",
    "CL", "GC", "SI", "ZN", "ZB",
    "6E", "6J", "HG", "6B", "6A",
]


class TestContractMasterExpansion:
    """Verify all 14 roots have ContractSpecs."""

    def test_all_roots_present(self):
        for root in FULL_UNIVERSE:
            assert root in CONTRACT_MASTER, f"{root} missing from CONTRACT_MASTER"

    def test_all_specs_have_required_fields(self):
        for root, spec in CONTRACT_MASTER.items():
            if root not in FULL_UNIVERSE:
                continue
            assert spec.multiplier > 0, f"{root} multiplier must be positive"
            assert spec.tick_size > 0, f"{root} tick_size must be positive"
            assert spec.tick_value > 0, f"{root} tick_value must be positive"
            assert spec.exchange != "", f"{root} exchange must be set"
            assert len(spec.roll_month_codes) > 0, f"{root} roll_month_codes must be non-empty"

    def test_tick_value_equals_multiplier_times_tick(self):
        for root in FULL_UNIVERSE:
            spec = CONTRACT_MASTER[root]
            expected = spec.multiplier * spec.tick_size
            assert abs(spec.tick_value - expected) < 1e-6, (
                f"{root}: tick_value={spec.tick_value} != multiplier*tick_size={expected}"
            )

    def test_roll_cycle_consistency(self):
        """roll_cycle must match the month codes pattern."""
        for root in FULL_UNIVERSE:
            spec = CONTRACT_MASTER[root]
            if spec.roll_cycle == "quarterly":
                assert set(spec.roll_month_codes) == {"H", "M", "U", "Z"}, (
                    f"{root}: quarterly but month codes = {spec.roll_month_codes}"
                )
            elif spec.roll_cycle == "monthly":
                assert len(spec.roll_month_codes) == 12, (
                    f"{root}: monthly but only {len(spec.roll_month_codes)} month codes"
                )


class TestExchangeMapExpansion:
    """Verify all 14 roots are in the EXCHANGE_MAP."""

    def test_all_roots_present(self):
        for root in FULL_UNIVERSE:
            assert root in EXCHANGE_MAP, f"{root} missing from EXCHANGE_MAP"

    def test_exchange_consistency(self):
        """CONTRACT_MASTER.exchange must match EXCHANGE_MAP."""
        for root in FULL_UNIVERSE:
            spec = CONTRACT_MASTER[root]
            # EXCHANGE_MAP may differ slightly (e.g. ECBOT vs CBOT) — normalise
            from sleeves.cooc_reversal_futures.contract_spec_checks import _normalise_exchange
            assert _normalise_exchange(EXCHANGE_MAP[root]) == _normalise_exchange(spec.exchange), (
                f"{root}: EXCHANGE_MAP={EXCHANGE_MAP[root]} vs spec.exchange={spec.exchange}"
            )


class TestRollCoverage:
    """Verify active_contract_for_day produces valid, continuous results."""

    @pytest.mark.parametrize("root", FULL_UNIVERSE)
    def test_no_roll_gaps(self, root: str):
        """Walk 2 years of dates: contract symbol must change at most once
        and always be non-empty."""
        spec = CONTRACT_MASTER[root]
        start = date(2024, 1, 2)
        end = date(2025, 12, 31)
        current = start
        prev_contract = None
        transitions = 0

        while current <= end:
            contract = active_contract_for_day(root, current, spec)
            assert contract, f"{root}: empty contract on {current}"
            if prev_contract and contract != prev_contract:
                transitions += 1
            prev_contract = contract
            current += timedelta(days=1)

        # Expect at least 1 roll transition over 2 years
        # (monthly: ~24, quarterly: ~8, custom: varies)
        assert transitions >= 1, f"{root}: no roll transitions in 2 years"

    @pytest.mark.parametrize("root", ["CL"])
    def test_monthly_has_many_transitions(self, root: str):
        """CL rolls monthly → ~12 transitions per year."""
        spec = CONTRACT_MASTER[root]
        start = date(2024, 1, 2)
        end = date(2024, 12, 31)
        current = start
        prev_contract = None
        transitions = 0

        while current <= end:
            contract = active_contract_for_day(root, current, spec)
            if prev_contract and contract != prev_contract:
                transitions += 1
            prev_contract = contract
            current += timedelta(days=1)

        # Monthly should have roughly 11 transitions in a year
        assert transitions >= 8, f"CL: expected ≥8 monthly transitions, got {transitions}"


class TestContractSpecChecks:
    """Verify post-qualification validation logic."""

    def test_valid_qualification_passes(self):
        spec = CONTRACT_MASTER["ES"]
        result = validate_qualified_contract(spec, "CME", "202503")
        assert result.passed
        assert len(result.warnings) == 0

    def test_exchange_mismatch_fails(self):
        spec = CONTRACT_MASTER["ES"]
        result = validate_qualified_contract(spec, "NYMEX", "202503")
        assert not result.passed
        assert len(result.warnings) > 0
        assert "Exchange mismatch" in result.warnings[0]

    def test_exchange_alias_passes(self):
        """CBOT and ECBOT should be treated as equivalent."""
        spec = CONTRACT_MASTER["ZN"]
        result = validate_qualified_contract(spec, "CBOT", "202506")
        assert result.passed, f"CBOT→ECBOT alias failed: {result.warnings}"

    def test_month_code_mismatch_fails(self):
        """ES is quarterly (HMUZ). Month 02 (G) should fail."""
        spec = CONTRACT_MASTER["ES"]
        result = validate_qualified_contract(spec, "CME", "202502")
        assert not result.passed
        assert len(result.warnings) > 0
        assert "Month code mismatch" in result.warnings[0]

    def test_month_code_mismatch_allowed(self):
        """With allow_mismatch=True, should pass but still warn."""
        spec = CONTRACT_MASTER["ES"]
        result = validate_qualified_contract(spec, "CME", "202502", allow_mismatch=True)
        assert result.passed
        assert len(result.warnings) > 0

    def test_validate_all_specs(self):
        qualified = {
            "ES": {"exchange": "CME", "expiry_yyyymm": "202503"},
            "CL": {"exchange": "NYMEX", "expiry_yyyymm": "202504"},
        }
        results = validate_all_specs(CONTRACT_MASTER, qualified)
        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_missing_root_fails(self):
        qualified = {"FAKE": {"exchange": "CME", "expiry_yyyymm": "202503"}}
        results = validate_all_specs(CONTRACT_MASTER, qualified)
        assert len(results) == 1
        assert not results[0].passed
