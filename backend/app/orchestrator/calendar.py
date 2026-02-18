from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from zoneinfo import ZoneInfo

from .config import OrchestratorConfig

try:
    import exchange_calendars as xcals
except Exception:  # pragma: no cover
    xcals = None


class Session(str, Enum):
    PREMARKET = "premarket"
    OPEN = "open"
    INTRADAY = "intraday"
    PRECLOSE = "preclose"
    CLOSE = "close"
    OVERNIGHT = "overnight"


@dataclass(frozen=True)
class SessionBounds:
    premarket_start: datetime
    open_start: datetime
    intraday_start: datetime
    preclose_start: datetime
    close_start: datetime
    afterhours_end: datetime


def _parse_hhmm(value: str) -> time:
    hh, mm = value.split(":")
    return time(int(hh), int(mm))


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d + timedelta(days=7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1) - timedelta(days=1)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def _easter(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _fallback_holidays(year: int) -> set[date]:
    return {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed(date(year, 12, 25)),
    }


def _fallback_early_closes(year: int) -> set[date]:
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    christmas_eve = date(year, 12, 24)
    if christmas_eve.weekday() >= 5:
        christmas_eve = date(year, 12, 23)
    return {thanksgiving + timedelta(days=1), christmas_eve}


class MarketCalendar:
    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self.tz = ZoneInfo(config.timezone)
        self._calendar = xcals.get_calendar(config.exchange) if xcals is not None else None

    def is_trading_day(self, dt: datetime) -> bool:
        day = dt.astimezone(self.tz).date()
        if self._calendar is not None:
            return bool(self._calendar.is_session(day))
        return day.weekday() < 5 and day not in _fallback_holidays(day.year)

    def session_bounds(self, asof_date: date) -> dict[str, datetime]:
        windows = self.config.session_windows
        open_ts = datetime.combine(asof_date, _parse_hhmm(windows["OPEN"].start), tzinfo=self.tz)
        close_ts = datetime.combine(asof_date, _parse_hhmm(windows["CLOSE"].end), tzinfo=self.tz)
        if self._calendar is not None and self._calendar.is_session(asof_date):
            o = self._calendar.session_open(asof_date).tz_convert(self.tz.key).to_pydatetime()
            c = self._calendar.session_close(asof_date).tz_convert(self.tz.key).to_pydatetime()
            open_ts = o
            close_ts = c
        elif asof_date in _fallback_early_closes(asof_date.year):
            close_ts = datetime.combine(asof_date, time(13, 0), tzinfo=self.tz)

        return {
            "preopen_start": datetime.combine(asof_date, _parse_hhmm(windows["PREMARKET"].start), tzinfo=self.tz),
            "open_start": datetime.combine(asof_date, _parse_hhmm(windows["OPEN"].start), tzinfo=self.tz),
            "intraday_start": datetime.combine(asof_date, _parse_hhmm(windows["INTRADAY"].start), tzinfo=self.tz),
            "preclose_start": datetime.combine(asof_date, _parse_hhmm(windows["PRECLOSE"].start), tzinfo=self.tz),
            "close_start": datetime.combine(asof_date, _parse_hhmm(windows["CLOSE"].start), tzinfo=self.tz),
            "open": open_ts,
            "close": close_ts,
            "afterhours_end": datetime.combine(asof_date, _parse_hhmm(windows["AFTERHOURS"].end), tzinfo=self.tz),
        }

    def current_session(self, now: datetime) -> Session:
        now_local = now.astimezone(self.tz)
        if not self.is_trading_day(now_local):
            return Session.OVERNIGHT
        b = self.session_bounds(now_local.date())
        if now_local < b["preopen_start"]:
            return Session.OVERNIGHT
        if now_local < b["open_start"]:
            return Session.PREMARKET
        if now_local < b["intraday_start"]:
            return Session.OPEN
        if now_local < b["preclose_start"]:
            return Session.INTRADAY
        if now_local < b["close_start"]:
            return Session.PRECLOSE
        if now_local < b["afterhours_end"]:
            return Session.CLOSE
        return Session.OVERNIGHT


def is_trading_day(dt: datetime, config: OrchestratorConfig | None = None) -> bool:
    return MarketCalendar(config or OrchestratorConfig()).is_trading_day(dt)


def session_bounds(asof_date: date, config: OrchestratorConfig | None = None) -> dict[str, datetime]:
    return MarketCalendar(config or OrchestratorConfig()).session_bounds(asof_date)


def current_session(now: datetime, config: OrchestratorConfig | None = None) -> Session:
    return MarketCalendar(config or OrchestratorConfig()).current_session(now)
