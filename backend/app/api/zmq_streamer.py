"""ZMQ Streaming Publisher — dual-channel telemetry for native frontend.

Provides two ZeroMQ PUB sockets:
  - Event channel (port 5556): Protobuf-encoded discrete events (ticks, alerts, metrics)
  - Grid  channel (port 5557): Apache Arrow IPC-encoded dense tabular data

Designed for tight integration with the FastAPI ASGI lifespan (non-blocking).
Uses zmq.asyncio to avoid blocking uvicorn worker threads.

Usage:
    from backend.app.api.zmq_streamer import ZmqPublisher, zmq_publisher

    # In lifespan:
    zmq_publisher.start()
    yield
    zmq_publisher.shutdown()

    # Publishing events:
    await zmq_publisher.publish_event(b"telemetry.metric", envelope_bytes)
    await zmq_publisher.publish_grid(b"grid.executions", arrow_batch)
"""
from __future__ import annotations

import asyncio
import logging
import struct
import threading
import time
from typing import TYPE_CHECKING

import pyarrow as pa

logger = logging.getLogger("algae.zmq_streamer")

# Lazy import to avoid hard crash if zmq/protobuf not installed yet
_zmq = None
_zmq_asyncio = None


def _ensure_zmq():
    global _zmq, _zmq_asyncio
    if _zmq is None:
        import zmq
        _zmq = zmq


# ── Sequence Counter ────────────────────────────────────────────────

class _SequenceCounter:
    """Thread-safe monotonically increasing sequence counter."""
    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

    @property
    def value(self) -> int:
        return self._value


# ── ZMQ Publisher ───────────────────────────────────────────────────

class ZmqPublisher:
    """Dual-channel ZeroMQ publisher for the native frontend.

    Event channel (Protobuf): tcp://*:{event_port}
    Grid channel  (Arrow IPC): tcp://*:{grid_port}

    THREAD SAFETY: All socket sends are serialized via _send_lock to prevent
    concurrent access from async uvicorn workers and sync background threads,
    which would corrupt libzmq's internal reference counters (refs_ >= 0 crash).
    """

    def __init__(
        self,
        event_port: int = 5556,
        grid_port: int = 5557,
        *,
        enabled: bool = True,
    ) -> None:
        self.event_port = event_port
        self.grid_port = grid_port
        self.enabled = enabled
        self.sequence = _SequenceCounter()
        self._send_lock = threading.Lock()

        self._ctx = None
        self._event_sock = None
        self._grid_sock = None
        self._started = False

    def start(self) -> None:
        """Initialize ZMQ context and bind sockets. Call during ASGI lifespan startup."""
        if not self.enabled:
            logger.info("ZMQ publisher disabled — skipping initialization")
            return

        _ensure_zmq()

        self._ctx = _zmq.Context()

        # Event channel — Protobuf binary frames
        self._event_sock = self._ctx.socket(_zmq.PUB)
        self._event_sock.setsockopt(_zmq.SNDHWM, 50_000)  # High water mark
        self._event_sock.setsockopt(_zmq.LINGER, 0)        # Don't block on shutdown
        self._event_sock.bind(f"tcp://*:{self.event_port}")

        # Grid channel — Arrow IPC stream format
        self._grid_sock = self._ctx.socket(_zmq.PUB)
        self._grid_sock.setsockopt(_zmq.SNDHWM, 1_000)
        self._grid_sock.setsockopt(_zmq.LINGER, 0)
        self._grid_sock.bind(f"tcp://*:{self.grid_port}")

        self._started = True
        logger.info(
            "ZMQ publisher started — event=tcp://*:%d  grid=tcp://*:%d",
            self.event_port, self.grid_port,
        )

    def shutdown(self) -> None:
        """Close sockets and terminate context. Call during ASGI lifespan shutdown."""
        if not self._started:
            return

        for sock in (self._event_sock, self._grid_sock):
            if sock is not None:
                try:
                    sock.close(linger=0)
                except Exception:
                    pass

        if self._ctx is not None:
            try:
                self._ctx.term()
            except Exception:
                pass

        self._started = False
        logger.info("ZMQ publisher shut down")

    # ── Event Channel (Protobuf) ────────────────────────────────────

    async def publish_event(self, topic: bytes, payload: bytes) -> None:
        """Publish a Protobuf-serialized message on the event channel.

        Args:
            topic:   Routing topic (e.g. b"telemetry.metric", b"control.snapshot")
            payload: Pre-serialized Protobuf bytes (StreamEnvelope or ControlSnapshot)
        """
        if not self._started:
            return

        seq = self.sequence.next()
        # Multipart: [topic] [8-byte sequence BE] [protobuf payload]
        seq_bytes = struct.pack(">Q", seq)
        with self._send_lock:
            try:
                self._event_sock.send_multipart(
                    [topic, seq_bytes, payload],
                    flags=_zmq.NOBLOCK,
                )
            except _zmq.Again:
                logger.warning("ZMQ event channel HWM reached -- dropping frame seq=%d", seq)
            except Exception:
                logger.exception("ZMQ event publish failed seq=%d", seq)

    def publish_event_sync(self, topic: bytes, payload: bytes) -> None:
        """Synchronous (non-async) variant for use from background threads."""
        if not self._started:
            return

        seq = self.sequence.next()
        seq_bytes = struct.pack(">Q", seq)
        with self._send_lock:
            try:
                self._event_sock.send_multipart(
                    [topic, seq_bytes, payload],
                    flags=_zmq.NOBLOCK,
                )
            except _zmq.Again:
                logger.warning("ZMQ event channel HWM reached (sync) — dropping frame seq=%d", seq)
            except Exception:
                logger.exception("ZMQ event publish failed (sync) seq=%d", seq)

    # ── Grid Channel (Apache Arrow IPC) ─────────────────────────────

    async def publish_grid(self, topic: bytes, batch: pa.RecordBatch) -> None:
        """Publish an Apache Arrow RecordBatch on the grid channel.

        Args:
            topic: Routing topic (e.g. b"grid.executions", b"grid.positions")
            batch: Arrow RecordBatch to serialize and send
        """
        if not self._started:
            return

        seq = self.sequence.next()
        seq_bytes = struct.pack(">Q", seq)

        # Serialize RecordBatch to Arrow IPC stream format
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, batch.schema) as writer:
            writer.write_batch(batch)
        arrow_bytes = sink.getvalue().to_pybytes()

        with self._send_lock:
            try:
                self._grid_sock.send_multipart(
                    [topic, seq_bytes, arrow_bytes],
                    flags=_zmq.NOBLOCK,
                )
            except _zmq.Again:
                logger.warning("ZMQ grid channel HWM reached -- dropping batch seq=%d", seq)
            except Exception:
                logger.exception("ZMQ grid publish failed seq=%d", seq)

    # ── Properties ──────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._started

    @property
    def current_sequence(self) -> int:
        return self.sequence.value


# ── Module-level singleton ──────────────────────────────────────────
# Import this from anywhere in the backend to publish events.

zmq_publisher = ZmqPublisher(
    event_port=int(__import__("os").environ.get("ALGAIE_ZMQ_EVENT_PORT", "5556")),
    grid_port=int(__import__("os").environ.get("ALGAIE_ZMQ_GRID_PORT", "5557")),
    enabled=__import__("os").environ.get("ALGAIE_ZMQ_ENABLED", "1") == "1",
)
