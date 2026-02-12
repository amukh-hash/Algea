"""Tests for event calendar and blackout windows (Phase 4)."""
from __future__ import annotations

from datetime import date

import pytest

from algaie.execution.options.config import VRPConfig
from backend.app.risk.event_calendar import EventCalendar, MacroEvent


class TestBlackoutWindow:
    def test_date_before_fomc_is_blackout(self):
        """Day before FOMC should be blacked out."""
        cal = EventCalendar(events=[
            MacroEvent("FOMC", date(2024, 6, 12)),
        ])
        cfg = VRPConfig(avoid_open_days_before_event=1, avoid_open_days_after_event=0)
        assert cal.is_blackout(date(2024, 6, 11), cfg)

    def test_event_day_is_blackout(self):
        """Event day itself should be blacked out."""
        cal = EventCalendar(events=[
            MacroEvent("CPI", date(2024, 6, 14)),
        ])
        cfg = VRPConfig(avoid_open_days_before_event=1, avoid_open_days_after_event=0)
        assert cal.is_blackout(date(2024, 6, 14), cfg)

    def test_day_after_event_not_blacked_out_by_default(self):
        """With avoid_after=0, day after event is not blackout."""
        cal = EventCalendar(events=[
            MacroEvent("NFP", date(2024, 6, 7)),
        ])
        cfg = VRPConfig(avoid_open_days_before_event=1, avoid_open_days_after_event=0)
        assert not cal.is_blackout(date(2024, 6, 8), cfg)

    def test_outside_blackout(self):
        """Dates far from any event are not blackout."""
        cal = EventCalendar(events=[
            MacroEvent("FOMC", date(2024, 6, 12)),
        ])
        cfg = VRPConfig(avoid_open_days_before_event=1, avoid_open_days_after_event=0)
        assert not cal.is_blackout(date(2024, 6, 5), cfg)

    def test_after_window_with_positive_after(self):
        """With avoid_after=1, day after event is blacked out."""
        cal = EventCalendar(events=[
            MacroEvent("CPI", date(2024, 6, 14)),
        ])
        cfg = VRPConfig(avoid_open_days_before_event=0, avoid_open_days_after_event=1)
        assert cal.is_blackout(date(2024, 6, 15), cfg)
        assert not cal.is_blackout(date(2024, 6, 16), cfg)


class TestMultipleEvents:
    def test_overlapping_blackouts(self):
        """Multiple events can create overlapping blackouts."""
        cal = EventCalendar(events=[
            MacroEvent("CPI", date(2024, 6, 14)),
            MacroEvent("FOMC", date(2024, 6, 14)),
        ])
        cfg = VRPConfig(avoid_open_days_before_event=1, avoid_open_days_after_event=0)
        evts = cal.get_blackout_events(date(2024, 6, 13), cfg)
        assert len(evts) == 2


class TestNextEvent:
    def test_next_event_found(self):
        cal = EventCalendar(events=[
            MacroEvent("CPI", date(2024, 5, 10)),
            MacroEvent("FOMC", date(2024, 6, 12)),
        ])
        nxt = cal.next_event(date(2024, 5, 15))
        assert nxt is not None
        assert nxt.name == "FOMC"

    def test_no_future_event(self):
        cal = EventCalendar(events=[
            MacroEvent("CPI", date(2024, 5, 10)),
        ])
        assert cal.next_event(date(2024, 6, 1)) is None
