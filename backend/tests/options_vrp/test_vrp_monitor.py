"""Tests for VRP telemetry and monitoring (Phase 5)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from algae.execution.options.config import VRPConfig
from backend.app.monitoring.vrp_monitor import (
    DailySnapshot,
    MonitorAlert,
    VRPMonitor,
)


class TestThresholdAlerts:
    def test_scenario_loss_warning(self):
        cfg = VRPConfig(max_worst_case_scenario_loss_pct_nav=0.06,
                        monitor_scenario_warning_pct=0.80)
        monitor = VRPMonitor(cfg)
        snap = DailySnapshot(
            as_of_date="2024-06-01",
            scenario_worst_loss_pct=0.05,  # 83% of 0.06 budget
        )
        snap = monitor.record(snap)
        assert any(a.metric == "scenario_worst_loss_pct" for a in snap.alerts)

    def test_no_alert_when_below_threshold(self):
        cfg = VRPConfig(max_worst_case_scenario_loss_pct_nav=0.06,
                        monitor_scenario_warning_pct=0.80)
        monitor = VRPMonitor(cfg)
        snap = DailySnapshot(
            as_of_date="2024-06-01",
            scenario_worst_loss_pct=0.02,  # 33% of budget — fine
        )
        snap = monitor.record(snap)
        assert len(snap.alerts) == 0

    def test_health_warning(self):
        cfg = VRPConfig(monitor_health_warning=0.70)
        monitor = VRPMonitor(cfg)
        snap = DailySnapshot(as_of_date="2024-06-01", forecast_health=0.50)
        snap = monitor.record(snap)
        assert any(a.metric == "forecast_health" for a in snap.alerts)

    def test_convexity_warning(self):
        cfg = VRPConfig(max_short_convexity_score=80.0,
                        monitor_convexity_warning_pct=0.80)
        monitor = VRPMonitor(cfg)
        snap = DailySnapshot(as_of_date="2024-06-01", short_convexity_score=70.0)
        snap = monitor.record(snap)
        assert any(a.metric == "short_convexity_score" for a in snap.alerts)

    def test_allocation_overrun(self):
        cfg = VRPConfig(w_max_vrp=0.25)
        monitor = VRPMonitor(cfg)
        snap = DailySnapshot(as_of_date="2024-06-01", w_vrp=0.30)
        snap = monitor.record(snap)
        assert any(a.severity == "CRITICAL" for a in snap.alerts)


class TestMonitorPersistence:
    def test_save_creates_files(self):
        monitor = VRPMonitor()
        monitor.record(DailySnapshot(as_of_date="2024-06-01"))
        with tempfile.TemporaryDirectory() as tmp:
            path = monitor.save(Path(tmp))
            assert path.exists()
            csv_path = path.parent / "daily_telemetry.csv"
            assert csv_path.exists()
