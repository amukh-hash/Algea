"""Tests for SleeveDecision lifecycle: OK, HALTED, FAILED, DISABLED paths.

Per-sleeve unit tests validating that canonical SleeveDecision objects
enforce the authoritative 4-state invariant table:

    OK        -> intents MUST be non-empty
    HALTED    -> intents MUST be empty (valid data, intentional no-trade)
    FAILED    -> intents MUST be empty (data/system error)
    DISABLED  -> intents MUST be empty (feature off)
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from backend.app.contracts.canonical import (
    AssetClass,
    ExecutionPhase,
    SleeveDecision,
    SleeveStatus,
    TargetIntent,
    TraceRef,
)


_NOW = datetime(2026, 3, 7, 10, 0, 0, tzinfo=timezone.utc)
_DATE = date(2026, 3, 7)


def _trace(sleeve: str = "core") -> TraceRef:
    return TraceRef(
        run_id="run-test",
        sleeve=sleeve,
        sleeve_run_id="run-test",
        source_artifact="test_fixture",
        source_row_index=0,
        control_snapshot_id="ctrl-test",
        market_snapshot_id="mkt-test",
        portfolio_snapshot_id="pf-test",
    )


def _intent(sleeve: str = "core", symbol: str = "ES", weight: float = 0.02) -> TargetIntent:
    return TargetIntent(
        intent_id=f"ti-{sleeve}-{symbol}",
        run_id="run-test",
        asof_date=_DATE,
        sleeve=sleeve,
        symbol=symbol,
        asset_class=AssetClass.FUTURE,
        target_weight=weight,
        execution_phase=ExecutionPhase.FUTURES_OPEN,
        multiplier=50.0,
        trace=_trace(sleeve),
    )


def _decision(sleeve: str, status: SleeveStatus, intents=None, reason=None, **kw) -> SleeveDecision:
    defaults = dict(
        schema_version="sleeve_decision.v1",
        sleeve=sleeve,
        run_id="run-test",
        asof_date=_DATE,
        status=status,
        reason=reason,
        intents=intents if intents is not None else (),
        diagnostics=kw.get("diagnostics", {}),
        warnings=kw.get("warnings", ()),
        artifact_refs={},
        control_snapshot_id="ctrl-test",
        market_snapshot_id="mkt-test",
        portfolio_snapshot_id="pf-test",
        started_at=_NOW,
        completed_at=_NOW,
        generated_by="test",
    )
    return SleeveDecision(**defaults)


class TestCoreSleeveDecision:
    def test_ok_with_intent(self):
        d = _decision("core", SleeveStatus.OK, intents=(_intent("core"),))
        assert d.status == SleeveStatus.OK
        assert len(d.intents) == 1
        assert d.intents[0].sleeve == "core"

    def test_failed_on_missing_config(self):
        d = _decision("core", SleeveStatus.FAILED, reason="missing config")
        assert d.status == SleeveStatus.FAILED
        assert d.reason == "missing config"

    def test_halted_on_no_edge(self):
        d = _decision("core", SleeveStatus.HALTED, reason="no edge detected")
        assert d.status == SleeveStatus.HALTED
        assert d.intents == ()


class TestVRPSleeveDecision:
    def test_ok_with_equity_intent(self):
        intent = TargetIntent(
            intent_id="ti-vrp-spy",
            run_id="run-test",
            asof_date=_DATE,
            sleeve="vrp",
            symbol="SPY",
            asset_class=AssetClass.EQUITY,
            target_weight=0.05,
            execution_phase=ExecutionPhase.INTRADAY,
            multiplier=1.0,
            trace=_trace("vrp"),
            metadata={"tenor": 30, "edge": 0.015},
        )
        d = _decision("vrp", SleeveStatus.OK, intents=(intent,))
        assert d.intents[0].metadata["tenor"] == 30

    def test_halted_on_drift_threshold(self):
        d = _decision("vrp", SleeveStatus.HALTED, reason="drift threshold exceeded")
        assert d.status == SleeveStatus.HALTED

    def test_failed_on_stale_surface(self):
        d = _decision("vrp", SleeveStatus.FAILED, reason="IV surface stale >2h")
        assert d.status == SleeveStatus.FAILED


class TestSelectorSleeveDecision:
    def test_ok_with_multiple_intents(self):
        intents = tuple(
            TargetIntent(
                intent_id=f"ti-sel-{sym}",
                run_id="run-test",
                asof_date=_DATE,
                sleeve="selector",
                symbol=sym,
                asset_class=AssetClass.EQUITY,
                target_weight=w,
                execution_phase=ExecutionPhase.INTRADAY,
                multiplier=1.0,
                trace=_trace("selector"),
            )
            for sym, w in [("AAPL", 0.05), ("MSFT", -0.03), ("NVDA", 0.02)]
        )
        d = _decision("selector", SleeveStatus.OK, intents=intents)
        assert len(d.intents) == 3

    def test_halted_on_turnover_exhausted(self):
        d = _decision(
            "selector",
            SleeveStatus.HALTED,
            reason="turnover budget exhausted",
            diagnostics={"turnover_used": 0.98},
            warnings=("approaching daily limit",),
        )
        assert d.status == SleeveStatus.HALTED
        assert d.diagnostics["turnover_used"] == 0.98


class TestStatArbSleeveDecision:
    def test_ok_with_leg_level_intents(self):
        """StatArb must emit leg-level intents, not pair diagnostics."""
        intents = (
            TargetIntent(
                intent_id="ti-sa-kre",
                run_id="run-test",
                asof_date=_DATE,
                sleeve="statarb",
                symbol="KRE",
                asset_class=AssetClass.EQUITY,
                target_weight=0.03,
                execution_phase=ExecutionPhase.INTRADAY,
                multiplier=1.0,
                trace=_trace("statarb"),
            ),
            TargetIntent(
                intent_id="ti-sa-iwm",
                run_id="run-test",
                asof_date=_DATE,
                sleeve="statarb",
                symbol="IWM",
                asset_class=AssetClass.EQUITY,
                target_weight=-0.025,
                execution_phase=ExecutionPhase.INTRADAY,
                multiplier=1.0,
                trace=_trace("statarb"),
            ),
        )
        d = _decision("statarb", SleeveStatus.OK, intents=intents)
        symbols = {i.symbol for i in d.intents}
        assert "KRE" in symbols
        assert "IWM" in symbols

    def test_halted_on_no_confirmed_pairs(self):
        d = _decision("statarb", SleeveStatus.HALTED, reason="no confirmed pairs")
        assert d.status == SleeveStatus.HALTED

    def test_failed_on_builder_failure(self):
        d = _decision("statarb", SleeveStatus.FAILED, reason="feature builder exception")
        assert d.status == SleeveStatus.FAILED


class TestFuturesOvernightSleeveDecision:
    def test_ok_with_sized_intent(self):
        intent = _intent("futures_overnight", "ES", 0.015)
        d = _decision("futures_overnight", SleeveStatus.OK, intents=(intent,))
        assert d.intents[0].target_weight == 0.015

    def test_failed_on_insufficient_history(self):
        d = _decision("futures_overnight", SleeveStatus.FAILED, reason="insufficient bars (<30)")
        assert d.status == SleeveStatus.FAILED

    def test_halted_on_low_confidence(self):
        d = _decision("futures_overnight", SleeveStatus.HALTED, reason="low confidence forecast")
        assert d.status == SleeveStatus.HALTED


class TestDisabledSleeve:
    def test_disabled_returns_no_intents(self):
        d = _decision("vrp", SleeveStatus.DISABLED, reason="feature off")
        assert d.intents == ()
        assert d.status == SleeveStatus.DISABLED

    def test_disabled_forbids_intents(self):
        with pytest.raises(ValueError, match="status=DISABLED but intents is non-empty"):
            _decision("vrp", SleeveStatus.DISABLED, intents=(_intent("vrp"),))
