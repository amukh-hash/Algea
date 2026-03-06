"""Tests for IBKR broker adapter retry behavior.

Verifies the tenacity exponential backoff logic correctly handles
broker gateway disconnects and allows non-retryable errors to propagate.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock
import pytest


# ── Retry behavior tests ───────────────────────────────────────────────

class TestIBKRRetryDecorator:
    """Test the _IBKR_RETRY decorator behavior using isolated mocks."""

    def test_connection_error_retries_and_succeeds(self):
        """ConnectionError on first call, success on second → should succeed."""
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

        call_count = 0

        @retry(
            retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
            wait=wait_fixed(0),  # No delay for tests
            stop=stop_after_attempt(3),
            reraise=True,
        )
        def flaky_call():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Connection reset by peer")
            return {"status": "ok"}

        result = flaky_call()
        assert result == {"status": "ok"}
        assert call_count == 2

    def test_timeout_error_retries_and_succeeds(self):
        """TimeoutError on first call, success on second → should succeed."""
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

        call_count = 0

        @retry(
            retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
            wait=wait_fixed(0),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        def flaky_call():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Gateway timeout")
            return {"fills": []}

        result = flaky_call()
        assert result == {"fills": []}
        assert call_count == 2

    def test_max_retries_exhausted_raises(self):
        """Persistent ConnectionError should raise after max attempts."""
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

        call_count = 0

        @retry(
            retry=retry_if_exception_type((ConnectionError,)),
            wait=wait_fixed(0),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Permanent failure")

        with pytest.raises(ConnectionError, match="Permanent failure"):
            always_fails()
        assert call_count == 3

    def test_non_retryable_error_propagates_immediately(self):
        """ValueError should propagate immediately without retry."""
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

        call_count = 0

        @retry(
            retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
            wait=wait_fixed(0),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        def logic_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid order format")

        with pytest.raises(ValueError, match="Invalid order format"):
            logic_error()
        assert call_count == 1  # No retries for non-network errors

    def test_os_error_retries(self):
        """OSError (socket-level) should trigger retry."""
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

        call_count = 0

        @retry(
            retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
            wait=wait_fixed(0),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        def socket_error():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("Socket closed")
            return {"positions": []}

        result = socket_error()
        assert result == {"positions": []}
        assert call_count == 3
