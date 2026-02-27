"""
Macro event calendar — stub for CPI, FOMC, NFP blackout windows.

Provides a simple lookup for known recurring macro events and
determines whether a given date is within a blackout window
where new spread entries should be blocked.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional, Tuple

from algea.execution.options.config import VRPConfig


# ═══════════════════════════════════════════════════════════════════════════
# Event types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MacroEvent:
    """A single known macro event."""
    name: str          # e.g. "CPI", "FOMC", "NFP"
    event_date: date   # the actual event date


# ═══════════════════════════════════════════════════════════════════════════
# Calendar
# ═══════════════════════════════════════════════════════════════════════════

class EventCalendar:
    """Maintains a list of macro events and checks blackout windows.

    Usage::

        cal = EventCalendar(events=[
            MacroEvent("FOMC", date(2024, 6, 12)),
            MacroEvent("CPI",  date(2024, 6, 14)),
        ])
        cal.is_blackout(date(2024, 6, 11), config)  # True if within window
    """

    def __init__(self, events: Optional[List[MacroEvent]] = None) -> None:
        self._events = events or []

    @property
    def events(self) -> List[MacroEvent]:
        return list(self._events)

    def add_event(self, event: MacroEvent) -> None:
        self._events.append(event)

    def add_events(self, events: List[MacroEvent]) -> None:
        self._events.extend(events)

    def is_blackout(
        self,
        as_of_date: date,
        config: VRPConfig,
    ) -> bool:
        """Return True if ``as_of_date`` is within any event's blackout window.

        Blackout = [event_date - avoid_before, event_date + avoid_after].
        """
        before = config.avoid_open_days_before_event
        after = config.avoid_open_days_after_event
        for ev in self._events:
            window_start = ev.event_date - timedelta(days=before)
            window_end = ev.event_date + timedelta(days=after)
            if window_start <= as_of_date <= window_end:
                return True
        return False

    def get_blackout_events(
        self,
        as_of_date: date,
        config: VRPConfig,
    ) -> List[MacroEvent]:
        """Return list of events whose blackout window covers ``as_of_date``."""
        before = config.avoid_open_days_before_event
        after = config.avoid_open_days_after_event
        result = []
        for ev in self._events:
            window_start = ev.event_date - timedelta(days=before)
            window_end = ev.event_date + timedelta(days=after)
            if window_start <= as_of_date <= window_end:
                result.append(ev)
        return result

    def next_event(self, as_of_date: date) -> Optional[MacroEvent]:
        """Return the next upcoming event after ``as_of_date``, or None."""
        future = [e for e in self._events if e.event_date > as_of_date]
        return min(future, key=lambda e: e.event_date) if future else None
