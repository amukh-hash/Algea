from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def et_anchor(ts_date: date, hhmmss: str) -> datetime:
    h, m, s = map(int, hhmmss.split(":"))
    return datetime.combine(ts_date, time(h, m, s), tzinfo=ET)


def is_trading_day(ts_date: date) -> bool:
    return ts_date.weekday() < 5
