"""ZMQ Dark Traffic Bridge — mirrors REST/SSE events to ZeroMQ PUB channels.

This module provides helper functions that backend route handlers call
after processing REST requests or SSE events. Each function serializes
the payload to a simple JSON-envelope format and publishes it on the
ZMQ event channel (port 5556) alongside the existing SSE/REST response.

This enables "dark traffic" mode: the native C++ frontend passively
consumes the ZMQ stream in parallel with the browser-based Next.js
frontend, without any code changes to the frontend.

Usage:
    from backend.app.api.zmq_bridge import bridge_control_snapshot, bridge_portfolio_summary

    # After serving GET /api/control/state:
    bridge_control_snapshot(state_dict)

    # After computing portfolio summary:
    bridge_portfolio_summary(summary_dict)
"""
from __future__ import annotations

import json
import logging
import struct
import time

logger = logging.getLogger("algae.zmq_bridge")


def _get_publisher():
    """Lazy import to avoid circular dependency."""
    from backend.app.api.zmq_streamer import zmq_publisher
    return zmq_publisher


def _pack_json_envelope(topic: str, payload: dict) -> tuple[bytes, bytes]:
    """Pack a JSON payload into a ZMQ multipart frame.

    Returns:
        (topic_bytes, json_payload_bytes) ready for zmq_publisher.publish_event_sync
    """
    # Add metadata for the native frontend
    envelope = {
        "ts": time.time(),
        "topic": topic,
        "data": payload,
    }
    return topic.encode("utf-8"), json.dumps(envelope, default=str).encode("utf-8")


# ── Control State ────────────────────────────────────────────────────

def bridge_control_snapshot(state: dict) -> None:
    """Mirror a control state snapshot to ZMQ.

    Called after GET /api/control/state or any control mutation.
    """
    pub = _get_publisher()
    if not pub.is_running:
        return

    topic_bytes, payload_bytes = _pack_json_envelope("control.snapshot", state)
    try:
        pub.publish_event_sync(topic_bytes, payload_bytes)
    except Exception:
        logger.debug("ZMQ bridge: control snapshot publish failed", exc_info=True)


def bridge_control_mutation(action: str, details: dict) -> None:
    """Mirror a control mutation (pause, freeze sleeve, etc.) to ZMQ."""
    pub = _get_publisher()
    if not pub.is_running:
        return

    payload = {"action": action, **details}
    topic_bytes, payload_bytes = _pack_json_envelope("control.mutation", payload)
    try:
        pub.publish_event_sync(topic_bytes, payload_bytes)
    except Exception:
        logger.debug("ZMQ bridge: control mutation publish failed", exc_info=True)


# ── Portfolio ────────────────────────────────────────────────────────

def bridge_portfolio_summary(summary: dict) -> None:
    """Mirror portfolio summary to ZMQ.

    Called after GET /api/control/portfolio-summary.
    """
    pub = _get_publisher()
    if not pub.is_running:
        return

    topic_bytes, payload_bytes = _pack_json_envelope("portfolio.summary", summary)
    try:
        pub.publish_event_sync(topic_bytes, payload_bytes)
    except Exception:
        logger.debug("ZMQ bridge: portfolio summary publish failed", exc_info=True)


# ── Live Prices ──────────────────────────────────────────────────────

def bridge_live_prices(prices: dict) -> None:
    """Mirror live price quotes to ZMQ.

    Called after GET /api/quotes.
    """
    pub = _get_publisher()
    if not pub.is_running:
        return

    topic_bytes, payload_bytes = _pack_json_envelope("telemetry.prices", prices)
    try:
        pub.publish_event_sync(topic_bytes, payload_bytes)
    except Exception:
        logger.debug("ZMQ bridge: live prices publish failed", exc_info=True)


# ── Broker Status ────────────────────────────────────────────────────

def bridge_broker_status(status: dict) -> None:
    """Mirror broker connectivity check to ZMQ."""
    pub = _get_publisher()
    if not pub.is_running:
        return

    topic_bytes, payload_bytes = _pack_json_envelope("telemetry.broker", status)
    try:
        pub.publish_event_sync(topic_bytes, payload_bytes)
    except Exception:
        logger.debug("ZMQ bridge: broker status publish failed", exc_info=True)


# ── Calendar ─────────────────────────────────────────────────────────

def bridge_calendar(calendar: dict) -> None:
    """Mirror calendar/session data to ZMQ."""
    pub = _get_publisher()
    if not pub.is_running:
        return

    topic_bytes, payload_bytes = _pack_json_envelope("telemetry.calendar", calendar)
    try:
        pub.publish_event_sync(topic_bytes, payload_bytes)
    except Exception:
        logger.debug("ZMQ bridge: calendar publish failed", exc_info=True)
