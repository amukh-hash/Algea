from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def now_et() -> datetime:
    return datetime.now(tz=ET)


def normalize_asof_date(asof: date | datetime | None) -> date:
    if asof is None:
        return now_et().date()
    if isinstance(asof, datetime):
        return asof.astimezone(ET).date()
    return asof
