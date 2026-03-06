"""
zmq_grid_publisher.py — PyArrow IPC Grid Publisher (XPUB State-Aware)

Publishes dense geometric data (forecast quantiles, allocation weights)
to the native frontend via Apache Arrow IPC over ZeroMQ port 5557.

Architecture:
  - XPUB socket (not PUB) — surfaces subscription state changes
  - _monitor_subscriptions() reaper task drains XPUB receive buffer
  - Prevents ephemeral port exhaustion from dead peer accumulation
  - Skips serialization when zero peers are subscribed to a topic

CPU-bound serialization is offloaded to asyncio.to_thread() to prevent
Python GIL starvation during live market order routing.

Memory management:
  - Explicit del batch/sink after serialization
  - Uses system memory pool for immediate OS return
  - SNDHWM=1000 implements drop-head when C++ disconnects
"""
from __future__ import annotations

import asyncio
import gc
import logging
import os
from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.ipc as ipc
import zmq
import zmq.asyncio

logger = logging.getLogger("algae.zmq.grid_publisher")

# Force Arrow to use the system allocator for immediate memory return
pa.set_memory_pool(pa.system_memory_pool())


class ArrowGridPublisher:
    """Async ZMQ XPUB publisher for Arrow IPC payloads with subscription reaping."""

    def __init__(self, ctx: zmq.asyncio.Context | None = None, port: int = 5557):
        self._ctx = ctx or zmq.asyncio.Context.instance()

        # Upgrade naive PUB → State-Aware XPUB
        # XPUB surfaces subscription/unsubscription events as recv() messages
        self.sock = self._ctx.socket(zmq.XPUB)
        self.sock.setsockopt(zmq.SNDHWM, 1000)
        self.sock.setsockopt(zmq.RCVHWM, 1000)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.SNDTIMEO, 0)  # Non-blocking send (Blind Spot 1)
        self.sock.bind(f"tcp://0.0.0.0:{port}")

        self._seq = 0
        self._active_topics: set[str] = set()
        self._peer_count = 0
        self._peer_events: list[dict] = []  # Audit trail for SRE
        self._market_tz = ZoneInfo("America/New_York")
        self._market_open = time(9, 30)
        self._market_close = time(16, 0)
        self._pagerduty_routing_key = os.environ.get("ALGAE_PAGERDUTY_ROUTING_KEY", "")

        # Spawn the non-blocking subscription reaper
        self._reaper_task = asyncio.create_task(self._monitor_subscriptions())

        logger.info("ArrowGridPublisher (XPUB) bound on tcp://0.0.0.0:%d", port)

    async def _monitor_subscriptions(self) -> None:
        """Actively drains the XPUB receive buffer to prevent pipeline deadlocks.

        XPUB receives 1-byte prefix: 0x01=subscribe, 0x00=unsubscribe.
        Blind Spot 1 mitigation: broad except ensures the reaper never dies.
        """
        try:
            while True:
                try:
                    event = await self.sock.recv()
                    if len(event) < 1:
                        continue

                    state = event[0]
                    topic = event[1:].decode("utf-8", errors="replace")

                    if state == 1:
                        self._active_topics.add(topic)
                        self._peer_count += 1
                        self._peer_events.append({
                            "action": "subscribe",
                            "topic": topic,
                            "timestamp": datetime.now(self._market_tz).isoformat(),
                            "peer_count": self._peer_count,
                        })
                        logger.info(
                            "ZMQ XPUB: peer subscribed to '%s' (active_topics=%d, peers=%d)",
                            topic, len(self._active_topics), self._peer_count,
                        )
                    elif state == 0:
                        self._active_topics.discard(topic)
                        self._peer_count = max(0, self._peer_count - 1)
                        now_est = datetime.now(self._market_tz)
                        self._peer_events.append({
                            "action": "unsubscribe",
                            "topic": topic,
                            "timestamp": now_est.isoformat(),
                            "peer_count": self._peer_count,
                        })
                        logger.info(
                            "ZMQ XPUB: peer evicted from '%s' — file descriptors reclaimed "
                            "(active_topics=%d, peers=%d)",
                            topic, len(self._active_topics), self._peer_count,
                        )
                        # §2.1: PagerDuty alert if peer drops during market hours
                        if self._market_open <= now_est.time() <= self._market_close:
                            if now_est.weekday() < 5:  # Mon-Fri
                                await self._fire_pagerduty_alert(topic, now_est)
                except asyncio.CancelledError:
                    raise  # Re-raise CancelledError to allow clean shutdown
                except Exception:
                    # Blind Spot 1: never let a parse error kill the reaper
                    logger.warning("XPUB reaper: error processing event", exc_info=True)
        except asyncio.CancelledError:
            logger.info("XPUB subscription reaper shutting down")

    async def _fire_pagerduty_alert(self, topic: str, timestamp: datetime) -> None:
        """Fire a CRITICAL PagerDuty Events API v2 alert for peer disconnect."""
        if not self._pagerduty_routing_key:
            logger.warning(
                "XPUB: peer dropped during market hours on '%s' but "
                "ALGAE_PAGERDUTY_ROUTING_KEY not set — alert suppressed",
                topic,
            )
            return

        import aiohttp

        payload = {
            "routing_key": self._pagerduty_routing_key,
            "event_action": "trigger",
            "payload": {
                "summary": f"Trading terminal disconnected from ZMQ topic '{topic}' during market hours",
                "severity": "critical",
                "source": "algae-backend-xpub-reaper",
                "component": "zmq_grid_publisher",
                "group": "trading-floor",
                "timestamp": timestamp.isoformat(),
                "custom_details": {
                    "topic": topic,
                    "remaining_peers": self._peer_count,
                    "active_topics": list(self._active_topics),
                },
            },
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 202:
                        logger.critical(
                            "PagerDuty CRITICAL alert fired: peer dropped from '%s' at %s",
                            topic, timestamp.isoformat(),
                        )
                    else:
                        body = await resp.text()
                        logger.error("PagerDuty API returned %d: %s", resp.status, body)
        except Exception:
            logger.error("Failed to fire PagerDuty alert", exc_info=True)

    def _serialize_to_ipc(self, df: Any) -> bytes:
        """CPU-bound: convert DataFrame → Arrow IPC bytes with 64-byte alignment."""
        batch = pa.RecordBatch.from_pandas(df)
        sink = pa.BufferOutputStream()
        # Explicit 64-byte alignment for C++ AVX-512 SIMD vectorization safety
        # Blind Spot 3: disable dictionary encoding — C++ expects contiguous STRING
        options = ipc.IpcWriteOptions(alignment=64, emit_dictionary_deltas=False)
        with ipc.new_stream(sink, batch.schema, options=options) as writer:
            writer.write_batch(batch)
        result = sink.getvalue().to_pybytes()

        # Aggressive memory cleanup — Arrow's C++ allocator bypasses pymalloc
        del batch
        del sink
        gc.collect()

        return result

    async def broadcast_grid(self, topic: bytes, df: Any) -> None:
        """Serialize and publish a DataFrame as Arrow IPC over ZMQ.

        Skips serialization entirely if zero peers are subscribed to this topic.
        Serialization is offloaded to a thread to prevent GIL starvation.
        """
        # CPU optimization: bypass PyArrow if no peers are listening
        topic_str = topic.decode("utf-8", errors="replace")
        if topic_str not in self._active_topics and self._peer_count == 0:
            return

        try:
            ipc_bytes = await asyncio.to_thread(self._serialize_to_ipc, df)
            self._seq += 1
            await self.sock.send_multipart([topic, ipc_bytes], flags=zmq.NOBLOCK)
            logger.debug(
                "Published %s: %d bytes (seq=%d)",
                topic_str, len(ipc_bytes), self._seq,
            )
        except zmq.Again:
            # SNDTIMEO=0 + NOBLOCK: drop frame rather than block event loop
            logger.warning("Grid publish dropped for topic=%s (SNDHWM full)", topic_str)
        except Exception:
            logger.warning("Grid publish failed for topic=%s", topic_str, exc_info=True)

    async def broadcast_fanchart(self, df: Any) -> None:
        """Publish PatchTST forecast quantiles (P10/P25/P50/P75/P90)."""
        await self.broadcast_grid(b"chart.kronos_fan", df)

    async def broadcast_sankey(self, df: Any) -> None:
        """Publish Meta-Allocator sleeve weight flows."""
        await self.broadcast_grid(b"chart.sankey_alloc", df)

    async def broadcast_positions(self, df: Any) -> None:
        """Publish positions grid for Tab 5 Arrow IPC hydration."""
        await self.broadcast_grid(b"grid.positions", df)

    def close(self) -> None:
        """Shutdown: cancel reaper, close socket."""
        if self._reaper_task and not self._reaper_task.done():
            self._reaper_task.cancel()
        self.sock.close()
        logger.info(
            "ArrowGridPublisher closed (final seq=%d, topics=%d)",
            self._seq, len(self._active_topics),
        )
