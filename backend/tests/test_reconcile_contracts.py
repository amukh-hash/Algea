from datetime import date

import pytest

from backend.app.execution.reconcile_futures import reconcile_day
from backend.app.schemas.fill_position import FILLS_SCHEMA_VERSION, POSITIONS_SCHEMA_VERSION


def test_reconcile_accepts_canonical_and_alt_keys():
    report = reconcile_day(
        asof=date(2026, 2, 17),
        open_intents=[{"symbol": "ESH26", "quantity": 2, "side": "buy"}],
        close_intents=[],
        open_orders=None,
        close_orders=None,
        fills=[{"schema_version": FILLS_SCHEMA_VERSION, "symbol": "ESH26", "quantity": 2, "price": 5000, "side": "buy"}],
        positions=[{"schema_version": POSITIONS_SCHEMA_VERSION, "symbol": "ESH26", "quantity": 2, "avg_cost": 5000}],
    )
    assert report["status"] in {"CLEAN", "ISSUES_FOUND"}


def test_reconcile_rejects_unknown_schema_version():
    with pytest.raises(ValueError):
        reconcile_day(
            asof=date(2026, 2, 17),
            open_intents=[],
            close_intents=[],
            open_orders=None,
            close_orders=None,
            fills=[{"schema_version": "fills.v999", "symbol": "ESH26", "quantity": 1, "price": 1, "side": "buy"}],
            positions=[],
        )
