"""Operational Deployment: Dark Traffic Validator

Connects to the ZMQ event channel (tcp://127.0.0.1:5556) and verifies
that the backend is publishing dark-traffic events.

Usage:
    python backend/scripts/verify_zmq_dark_traffic.py
"""
from __future__ import annotations

import sys
import time
import urllib.request

try:
    import zmq
except ImportError:
    print("[ERROR] pyzmq not installed. Run: pip install pyzmq")
    sys.exit(1)


def trigger_rest_traffic():
    """Fire REST calls to make the backend publish ZMQ events."""
    endpoints = [
        "http://127.0.0.1:8000/api/control/state",
        "http://127.0.0.1:8000/api/control/calendar",
        "http://127.0.0.1:8000/api/control/broker-status",
    ]
    for url in endpoints:
        try:
            urllib.request.urlopen(url, timeout=3)
        except Exception:
            pass


def main():
    event_port = 5556
    grid_port = 5557
    timeout_s = 30

    print(f"[INFO] Connecting to ZMQ event channel tcp://127.0.0.1:{event_port} ...")

    ctx = zmq.Context()

    # Event channel
    event_sock = ctx.socket(zmq.SUB)
    event_sock.connect(f"tcp://127.0.0.1:{event_port}")
    event_sock.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all topics
    event_sock.setsockopt(zmq.RCVTIMEO, 2000)  # 2s recv timeout

    # Grid channel
    grid_sock = ctx.socket(zmq.SUB)
    grid_sock.connect(f"tcp://127.0.0.1:{grid_port}")
    grid_sock.setsockopt(zmq.SUBSCRIBE, b"")
    grid_sock.setsockopt(zmq.RCVTIMEO, 2000)

    print(f"[OK] ZMQ sockets connected (event:{event_port}, grid:{grid_port})")

    # ZMQ slow-joiner mitigation: wait for SUB to fully handshake
    # before triggering traffic. Without this, PUB sends arrive before
    # SUB subscription propagates and they are silently dropped.
    print("[INFO] Waiting 2s for ZMQ subscription handshake...")
    time.sleep(2)

    print(f"[INFO] Triggering REST calls to generate ZMQ traffic...")
    trigger_rest_traffic()
    print("[INFO] Waiting for messages...")
    print()

    received = 0
    topics_seen = set()
    start = time.time()
    last_trigger = start

    while time.time() - start < timeout_s:
        # Re-trigger REST every 5 seconds to keep traffic flowing
        now = time.time()
        if now - last_trigger > 5:
            trigger_rest_traffic()
            last_trigger = now

        # Check event channel
        try:
            parts = event_sock.recv_multipart(flags=zmq.NOBLOCK)
            topic = parts[0].decode("utf-8", errors="replace")
            topics_seen.add(topic)
            received += 1
            payload_size = sum(len(p) for p in parts)
            print(f"  [EVENT] topic={topic:30s}  parts={len(parts)}  bytes={payload_size}")
        except zmq.Again:
            pass

        # Check grid channel
        try:
            parts = grid_sock.recv_multipart(flags=zmq.NOBLOCK)
            topic = parts[0].decode("utf-8", errors="replace")
            topics_seen.add(topic)
            received += 1
            payload_size = sum(len(p) for p in parts)
            print(f"  [GRID]  topic={topic:30s}  parts={len(parts)}  bytes={payload_size}")
        except zmq.Again:
            pass

        if received == 0:
            time.sleep(0.5)
        else:
            time.sleep(0.1)

        # Stop after getting enough samples
        if received >= 10:
            break

    print()
    print("=" * 60)

    if received > 0:
        elapsed = time.time() - start
        print(f"[OK] Received {received} messages in {elapsed:.1f}s")
        print(f"[OK] Topics seen: {', '.join(sorted(topics_seen))}")
        print()
        print("[PASS] Dark traffic is flowing.")
        print("       The native frontend can passively consume this data")
        print("       via tcp://127.0.0.1:5556 (events) and :5557 (grids).")
    else:
        print(f"[FAIL] No messages received in {timeout_s}s")
        print()
        print("Troubleshooting:")
        print("  1. Is the backend running? uvicorn backend.app.api.main:app")
        print("  2. Is ALGAIE_ZMQ_ENABLED=1? (default: yes)")
        print("  3. Check backend logs for 'ZMQ event publish failed'")
        sys.exit(1)

    # Cleanup
    event_sock.close()
    grid_sock.close()
    ctx.term()


if __name__ == "__main__":
    main()
