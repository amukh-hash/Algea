from datetime import date, datetime, time

from backend.app.orchestrator.calendar import MarketCalendar
from backend.app.orchestrator.config import OrchestratorConfig


def test_orchestrator_calendar_holiday_golden():
    cal = MarketCalendar(OrchestratorConfig())
    # New Year's Day (holiday)
    assert not cal.is_trading_day(datetime(2026, 1, 1, 12, 0, tzinfo=cal.tz))
    # First trading day after holiday window
    assert cal.is_trading_day(datetime(2026, 1, 2, 12, 0, tzinfo=cal.tz))


def test_session_bounds_are_ordered():
    cal = MarketCalendar(OrchestratorConfig())
    b = cal.session_bounds(date(2026, 2, 17))
    assert b["preopen_start"] < b["open_start"] < b["intraday_start"] < b["preclose_start"] < b["close_start"]
