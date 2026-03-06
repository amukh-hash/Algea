"""Tests for BrokerWatchdog and _is_port_open / _try_connect_broker helpers.

Uses mocked socket and broker adapter to avoid real IBKR dependencies.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


# ── _is_port_open tests ─────────────────────────────────────────────

class TestIsPortOpen:
    """Test the TCP port probe utility."""

    def test_port_open_returns_true(self):
        from backend.app.api.control_routes import _is_port_open

        with patch("backend.app.api.control_routes._socket") as mock_socket:
            mock_sock = MagicMock()
            mock_sock.connect_ex.return_value = 0
            mock_socket.socket.return_value = mock_sock
            mock_socket.AF_INET = 2
            mock_socket.SOCK_STREAM = 1

            assert _is_port_open("127.0.0.1", 4002) is True
            mock_sock.connect_ex.assert_called_once_with(("127.0.0.1", 4002))
            mock_sock.close.assert_called_once()

    def test_port_closed_returns_false(self):
        from backend.app.api.control_routes import _is_port_open

        with patch("backend.app.api.control_routes._socket") as mock_socket:
            mock_sock = MagicMock()
            mock_sock.connect_ex.return_value = 111  # Connection refused
            mock_socket.socket.return_value = mock_sock
            mock_socket.AF_INET = 2
            mock_socket.SOCK_STREAM = 1

            assert _is_port_open("127.0.0.1", 4002) is False


# ── BrokerWatchdog tests ────────────────────────────────────────────

class TestBrokerWatchdog:
    """Test the BrokerWatchdog daemon behavior."""

    def test_outside_trading_window_is_noop(self):
        """Watchdog should not attempt connections on weekends."""
        from backend.app.api.control_routes import BrokerWatchdog

        wd = BrokerWatchdog(interval=1)
        # Simulate a Saturday
        with patch("backend.app.api.control_routes.datetime") as mock_dt:
            saturday = datetime(2026, 3, 7, 10, 0, 0)  # Saturday
            mock_dt.now.return_value = saturday
            assert wd._in_trading_window() is False

    def test_inside_trading_window(self):
        """Watchdog should recognize valid weekday trading hours."""
        from backend.app.api.control_routes import BrokerWatchdog

        wd = BrokerWatchdog(interval=1)
        with patch("backend.app.api.control_routes.datetime") as mock_dt:
            wednesday_930am = datetime(2026, 3, 4, 9, 30, 0)  # Wednesday
            mock_dt.now.return_value = wednesday_930am
            assert wd._in_trading_window() is True

    def test_before_trading_window(self):
        """Watchdog should not connect before 06:25."""
        from backend.app.api.control_routes import BrokerWatchdog

        wd = BrokerWatchdog(interval=1)
        with patch("backend.app.api.control_routes.datetime") as mock_dt:
            early = datetime(2026, 3, 4, 5, 0, 0)  # 5 AM Wednesday
            mock_dt.now.return_value = early
            assert wd._in_trading_window() is False

    def test_tick_connects_when_port_open_no_adapter(self):
        """Watchdog should auto-connect when Gateway is reachable but no adapter exists."""
        from backend.app.api import control_routes
        from backend.app.api.control_routes import BrokerWatchdog

        wd = BrokerWatchdog(interval=1)

        original_adapter = control_routes._broker_adapter
        try:
            control_routes._broker_adapter = None

            with (
                patch.object(wd, "_in_trading_window", return_value=True),
                patch("backend.app.api.control_routes._is_port_open", return_value=True),
                patch("backend.app.api.control_routes._try_connect_broker") as mock_connect,
                patch("backend.app.api.control_routes.bridge_broker_status"),
                patch.dict("os.environ", {"IBKR_GATEWAY_URL": "127.0.0.1:4002"}),
            ):
                mock_connect.return_value = {"connected": True, "status": "connected"}
                wd._tick()
                mock_connect.assert_called_once_with(source="watchdog")
        finally:
            control_routes._broker_adapter = original_adapter

    def test_tick_clears_adapter_when_port_closed(self):
        """Watchdog should clear the adapter when the Gateway goes away."""
        from backend.app.api import control_routes
        from backend.app.api.control_routes import BrokerWatchdog

        wd = BrokerWatchdog(interval=1)
        mock_adapter = MagicMock()

        original_adapter = control_routes._broker_adapter
        try:
            control_routes._broker_adapter = mock_adapter

            with (
                patch.object(wd, "_in_trading_window", return_value=True),
                patch("backend.app.api.control_routes._is_port_open", return_value=False),
                patch("backend.app.api.control_routes.bridge_broker_status"),
                patch.dict("os.environ", {"IBKR_GATEWAY_URL": "127.0.0.1:4002"}),
            ):
                wd._tick()
                mock_adapter.disconnect.assert_called_once()
                assert control_routes._broker_adapter is None
        finally:
            control_routes._broker_adapter = original_adapter

    def test_start_stop_lifecycle(self):
        """Watchdog thread starts and stops cleanly."""
        from backend.app.api.control_routes import BrokerWatchdog

        wd = BrokerWatchdog(interval=0.1)

        # Patch _loop to just sleep briefly
        def fake_loop():
            while wd._running:
                time.sleep(0.05)

        with patch.object(wd, "_loop", side_effect=fake_loop):
            wd.start()
            assert wd._running is True
            assert wd._thread is not None
            assert wd._thread.is_alive()

            wd.stop()
            assert wd._running is False
            time.sleep(0.2)
            assert not wd._thread.is_alive()
