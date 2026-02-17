"""Tests for family run ingestion logic.

Uses fixture three_sleeve JSONs to verify:
- exactly 1 family run is created
- equity metric has >= 2 points
- daily artifacts registered
- events include DECISION_MADE and RISK_LIMIT warnings
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ── Fixtures ────────────────────────────────────────────────────────────────

DAY1 = {
    "asof": "2026-02-16",
    "mode": "ibkr",
    "vrp_mode": "noop",
    "timestamp": "2026-02-16T10:45:35.136702",
    "account": {"equity": 1000000.0, "cash": 1000000.0, "buying_power": 4000000.0},
    "sleeve_capital": {"core": 500000.0, "vrp": 300000.0, "selector": 200000.0},
    "sleeves": {
        "core": {
            "sleeve": "core",
            "capital": 500000.0,
            "mode": "ibkr",
            "status": "submitted",
            "orders": [],
            "message": "Orders submitted via paper cycle runner",
        },
        "vrp": {
            "sleeve": "vrp",
            "capital": 300000.0,
            "mode": "noop",
            "status": "noop",
            "orders": [],
            "message": "VRP noop",
            "sizing_warning": {
                "spx_notional_per_contract": "~$500,000",
                "recommendation": "Use SPY options",
            },
        },
        "selector": {
            "sleeve": "selector",
            "capital": 200000.0,
            "mode": "ibkr",
            "status": "submitted",
            "orders": [],
            "intents": [
                {"symbol": "ASTS", "side": "buy", "quantity": 200, "score": 0.709, "weight": 0.1, "estimated_notional": 20000.0},
                {"symbol": "LAKE", "side": "sell", "quantity": 200, "score": -2.73, "weight": 0.1, "estimated_notional": 20000.0},
            ],
            "num_longs": 1,
            "num_shorts": 1,
            "message": "2 orders computed",
        },
    },
}

DAY2 = {
    "asof": "2026-02-17",
    "mode": "noop",
    "vrp_mode": "noop",
    "timestamp": "2026-02-17T02:48:13.757941",
    "account": {"equity": 1005000.0, "cash": 995000.0, "buying_power": 3980000.0},
    "sleeve_capital": {"core": 502500.0, "vrp": 301500.0, "selector": 201000.0},
    "sleeves": {
        "core": {
            "sleeve": "core",
            "capital": 502500.0,
            "mode": "noop",
            "status": "inputs_missing",
            "orders": [],
        },
        "vrp": {
            "sleeve": "vrp",
            "capital": 301500.0,
            "mode": "noop",
            "status": "noop",
            "orders": [],
            "message": "VRP noop",
        },
        "selector": {
            "sleeve": "selector",
            "capital": 201000.0,
            "mode": "noop",
            "status": "noop",
            "orders": [],
            "intents": [
                {"symbol": "DAVA", "side": "buy", "quantity": 200, "score": 0.65, "weight": 0.1, "estimated_notional": 20100.0},
            ],
            "num_longs": 1,
            "num_shorts": 0,
            "message": "1 orders computed, none submitted",
        },
    },
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_fixture_dir(tmp_path: Path) -> Path:
    """Write fixture JSONs to a temp directory mimicking data_lake structure."""
    for day_data, date_str in [(DAY1, "2026-02-16"), (DAY2, "2026-02-17")]:
        report_dir = tmp_path / "data_lake" / "three_sleeves" / "paper" / date_str
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "three_sleeve_report.json").write_text(json.dumps(day_data), encoding="utf-8")
    return tmp_path


# ── Tests ───────────────────────────────────────────────────────────────────

def test_family_run_creates_single_run(tmp_path):
    """Ingesting 2 three_sleeve reports should create exactly 1 family run."""
    import sys
    root = _make_fixture_dir(tmp_path)

    # Patch ROOT and reimport
    sys.path.insert(0, str(root))
    from backend.app.telemetry.storage import TelemetryStorage
    from backend.app.telemetry.emitter import TelemetryEmitter

    test_storage = TelemetryStorage(
        db_url=f"sqlite:///{tmp_path}/test_telemetry.db",
        artifacts_root=f"{tmp_path}/artifacts",
    )
    emitter = TelemetryEmitter(test_storage)

    # Import the ingestion functions
    from backend.scripts.ingest_real_reports import (
        _extract_three_sleeve_metrics,
        _extract_three_sleeve_events,
        _parse_report_timestamp,
        family_run_id,
        ingest_family,
    )

    # Build family
    reports = []
    for day_data, date_str in [(DAY1, "2026-02-16"), (DAY2, "2026-02-17")]:
        report_path = root / "data_lake" / "three_sleeves" / "paper" / date_str / "three_sleeve_report.json"
        reports.append((report_path, day_data))

    # Monkey-patch the module-level storage reference
    import backend.scripts.ingest_real_reports as ingest_mod
    orig_storage = ingest_mod.storage
    ingest_mod.storage = test_storage
    try:
        from backend.app.telemetry.schemas import RunType
        result = ingest_family(
            emitter, family_key="three_sleeve",
            kind="three_sleeve_report", run_type=RunType.sleeve_paper,
            reports=reports,
        )
    finally:
        ingest_mod.storage = orig_storage

    assert result is not None

    # Verify exactly 1 run
    runs, total = test_storage.list_runs({}, limit=100, offset=0)
    assert total == 1
    run = runs[0]
    assert "family" in run.tags
    assert run.meta.get("family_members") == 2

    # Verify equity has >= 2 points
    series = test_storage.query_metrics(run.run_id, ["equity"], None, None, None)
    assert len(series["equity"]) >= 2
    assert series["equity"][0].value == 1000000.0
    assert series["equity"][1].value == 1005000.0

    # Verify artifacts
    artifacts = test_storage.list_artifacts(run.run_id)
    assert len(artifacts) == 2

    # Verify events include DECISION_MADE and RISK_LIMIT
    events = test_storage.query_events(run.run_id, None, None, None, None, 100)
    event_types = [e.type for e in events]
    assert "DECISION_MADE" in event_types
    assert "RISK_LIMIT" in event_types  # inputs_missing and sizing_warning


def test_family_idempotency(tmp_path):
    """Re-ingesting the same reports should not create duplicate metric points."""
    root = _make_fixture_dir(tmp_path)

    from backend.app.telemetry.storage import TelemetryStorage
    from backend.app.telemetry.emitter import TelemetryEmitter

    test_storage = TelemetryStorage(
        db_url=f"sqlite:///{tmp_path}/test_telemetry.db",
        artifacts_root=f"{tmp_path}/artifacts",
    )
    emitter = TelemetryEmitter(test_storage)

    reports = []
    for day_data, date_str in [(DAY1, "2026-02-16"), (DAY2, "2026-02-17")]:
        report_path = root / "data_lake" / "three_sleeves" / "paper" / date_str / "three_sleeve_report.json"
        reports.append((report_path, day_data))

    import backend.scripts.ingest_real_reports as ingest_mod
    from backend.scripts.ingest_real_reports import ingest_family
    orig_storage = ingest_mod.storage
    ingest_mod.storage = test_storage
    try:
        from backend.app.telemetry.schemas import RunType
        # Ingest twice
        ingest_family(emitter, "three_sleeve", "three_sleeve_report", RunType.sleeve_paper, reports)
        ingest_family(emitter, "three_sleeve", "three_sleeve_report", RunType.sleeve_paper, reports)
    finally:
        ingest_mod.storage = orig_storage

    # Should still have exactly 1 run
    runs, total = test_storage.list_runs({}, limit=100, offset=0)
    assert total == 1

    # Equity should still have exactly 2 points (not 4)
    series = test_storage.query_metrics(runs[0].run_id, ["equity"], None, None, None)
    assert len(series["equity"]) == 2


def test_three_sleeve_metrics_completeness(tmp_path):
    """Verify all specified C1 metrics are emitted."""
    from backend.app.telemetry.storage import TelemetryStorage
    from backend.app.telemetry.emitter import TelemetryEmitter
    from backend.scripts.ingest_real_reports import _extract_three_sleeve_metrics, family_run_id

    run_id = family_run_id("three_sleeve")
    ts = datetime(2026, 2, 16, 12, 0, 0, tzinfo=timezone.utc)

    points = _extract_three_sleeve_metrics(DAY1, ts, run_id)
    keys = {p.key for p in points}

    expected_keys = {
        "equity", "cash", "buying_power",
        "sleeve_capital.core", "sleeve_capital.vrp", "sleeve_capital.selector", "sleeve_capital.total",
        "sleeve.core.orders_count", "sleeve.vrp.orders_count", "sleeve.selector.orders_count",
        "sleeve.selector.intents_count", "sleeve.selector.intent_notional_sum",
        "sleeve.selector.intent_abs_weight_sum",
        "sleeve.selector.num_longs", "sleeve.selector.num_shorts",
    }
    assert expected_keys.issubset(keys), f"Missing keys: {expected_keys - keys}"


def test_three_sleeve_events_completeness(tmp_path):
    """Verify all specified C2 events are emitted, including warnings."""
    from backend.scripts.ingest_real_reports import _extract_three_sleeve_events, family_run_id

    run_id = family_run_id("three_sleeve")
    ts = datetime(2026, 2, 16, 12, 0, 0, tzinfo=timezone.utc)

    # Day 1: ibkr mode, submitted + empty orders, sizing_warning
    events1 = _extract_three_sleeve_events(DAY1, ts, run_id)
    types1 = [(e[0].value, e[1].value) for e in events1]
    assert ("info", "DECISION_MADE") in types1
    assert ("warn", "RISK_LIMIT") in types1  # sizing_warning
    assert ("info", "ORDER_SUBMITTED") in types1  # submitted + empty orders

    # Day 2: inputs_missing
    events2 = _extract_three_sleeve_events(DAY2, ts, run_id)
    types2 = [(e[0].value, e[1].value) for e in events2]
    assert ("info", "DECISION_MADE") in types2
    assert ("warn", "RISK_LIMIT") in types2  # inputs_missing
