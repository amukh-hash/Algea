"""Shared-memory kill switch listener for the backend daemon.

Implements the Out-Of-Band (OOB) safety mechanism described in Phase 5
of the native frontend architecture. Uses boost::interprocess-compatible
shared memory layout (via mmap) so the C++ native frontend can perform
wait-free atomic writes to halt individual sleeves.

The backend polls this shared memory block and also registers a SIGUSR1
handler for immediate notification from the native UI.

Memory Layout (must match C++ SharedControlBlock exactly):
    Offset  0: uint64  sequence_id     (monotonic counter)
    Offset  8: uint32  halt_mask       (bitmask, bit i = sleeve i halted)
    Offset 12: char[52] reason         (human-readable halt reason)
    Total: 64 bytes, aligned to 64-byte cache line

Usage:
    listener = KillSwitchListener("AlgaieControlPlane")
    listener.start()
    ...
    if listener.is_sleeve_halted(4):
        # Sleeve 4 is halted
    ...
    listener.stop()
"""
from __future__ import annotations

import ctypes
import logging
import mmap
import os
import signal
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger("algae.kill_switch")

# ── Shared Memory Layout ────────────────────────────────────────────

CONTROL_BLOCK_SIZE = 64  # 8 (seq) + 4 (mask) + 52 (reason)
SHM_NAME = "AlgaieControlPlane"

# Sleeve ID → name mapping for logging
SLEEVE_NAMES = {
    0: "core",
    1: "vrp",
    2: "selector",
    3: "statarb",
    4: "chronos",
    5: "tft",
}


class SharedControlBlock(ctypes.Structure):
    """Mirror of the C++ SharedControlBlock (64-byte aligned)."""
    _pack_ = 1
    _fields_ = [
        ("sequence_id", ctypes.c_uint64),
        ("halt_mask", ctypes.c_uint32),
        ("reason", ctypes.c_char * 52),
    ]


# ── Kill Switch Listener ────────────────────────────────────────────

class KillSwitchListener:
    """Monitors shared memory for sleeve halt signals from the native UI.

    Supports two notification mechanisms:
      1. Polling: 100 Hz check of the shared memory sequence counter
      2. SIGUSR1: Immediate interrupt from the native frontend process
    """

    def __init__(
        self,
        shm_name: str = SHM_NAME,
        on_halt: Callable[[int, str], None] | None = None,
    ) -> None:
        self.shm_name = shm_name
        self.on_halt = on_halt  # Callback: on_halt(sleeve_id, reason)
        self._shm_fd = None
        self._mmap: mmap.mmap | None = None
        self._last_seq = 0
        self._halt_mask = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False

    def start(self) -> None:
        """Open or create the shared memory segment and start polling."""
        try:
            self._open_shm()
            self._register_signal()
            self._thread = threading.Thread(
                target=self._poll_loop,
                name="kill-switch-listener",
                daemon=True,
            )
            self._thread.start()
            self._started = True
            logger.info("Kill switch listener started — shm=%s", self.shm_name)
        except Exception:
            logger.exception("Failed to start kill switch listener")
            # Non-fatal: system operates without OOB kill switch
            self._started = False

    def stop(self) -> None:
        """Stop polling and release shared memory."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        self._started = False
        logger.info("Kill switch listener stopped")

    def is_sleeve_halted(self, sleeve_id: int) -> bool:
        """Check if a specific sleeve is halted."""
        return bool(self._halt_mask & (1 << sleeve_id))

    @property
    def halt_mask(self) -> int:
        return self._halt_mask

    @property
    def halt_reason(self) -> str:
        """Read the current halt reason string from shared memory."""
        if self._mmap is None:
            return ""
        try:
            self._mmap.seek(12)
            raw = self._mmap.read(64)
            return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        except Exception:
            return ""

    # ── Internal ────────────────────────────────────────────────────

    def _open_shm(self) -> None:
        """Open or create the shared memory segment."""
        if sys.platform == "win32":
            # Windows: use named shared memory via mmap
            self._mmap = mmap.mmap(
                -1,
                CONTROL_BLOCK_SIZE,
                tagname=self.shm_name,
                access=mmap.ACCESS_WRITE,
            )
            # Zero-initialize
            self._mmap.seek(0)
            self._mmap.write(b"\x00" * CONTROL_BLOCK_SIZE)
        else:
            # POSIX: use /dev/shm
            shm_path = Path(f"/dev/shm/{self.shm_name}")
            if not shm_path.exists():
                with open(shm_path, "wb") as f:
                    f.write(b"\x00" * CONTROL_BLOCK_SIZE)
            fd = os.open(str(shm_path), os.O_RDWR)
            self._mmap = mmap.mmap(fd, CONTROL_BLOCK_SIZE)
            os.close(fd)

    def _register_signal(self) -> None:
        """Register SIGUSR1 handler for immediate interrupt (POSIX only)."""
        if sys.platform == "win32":
            # Windows does not support SIGUSR1
            logger.debug("SIGUSR1 not available on Windows — relying on polling only")
            return

        def _sigusr1_handler(signum, frame):
            self._check_shm()

        signal.signal(signal.SIGUSR1, _sigusr1_handler)
        logger.debug("SIGUSR1 handler registered")

    def _poll_loop(self) -> None:
        """Background polling loop — checks shared memory at 100 Hz."""
        while not self._stop_event.is_set():
            self._check_shm()
            self._stop_event.wait(0.01)  # 100 Hz

    def _check_shm(self) -> None:
        """Read the shared memory control block and dispatch halt events."""
        if self._mmap is None:
            return

        try:
            self._mmap.seek(0)
            raw = self._mmap.read(CONTROL_BLOCK_SIZE)
            seq, mask = struct.unpack_from(">QI", raw, 0)

            if seq <= self._last_seq:
                return  # No change

            self._last_seq = seq
            old_mask = self._halt_mask
            self._halt_mask = mask

            # Detect newly halted sleeves
            newly_halted = mask & ~old_mask
            if newly_halted and self.on_halt:
                reason_bytes = raw[12:64]
                reason = reason_bytes.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
                for sleeve_id in range(32):
                    if newly_halted & (1 << sleeve_id):
                        name = SLEEVE_NAMES.get(sleeve_id, f"sleeve_{sleeve_id}")
                        logger.critical(
                            "OOB KILL SWITCH: Sleeve %s (id=%d) HALTED — reason: %s",
                            name, sleeve_id, reason,
                        )
                        try:
                            self.on_halt(sleeve_id, reason)
                        except Exception:
                            logger.exception("Kill switch callback error for sleeve %d", sleeve_id)

            # Detect newly unhalted sleeves
            newly_unhalted = old_mask & ~mask
            if newly_unhalted:
                for sleeve_id in range(32):
                    if newly_unhalted & (1 << sleeve_id):
                        name = SLEEVE_NAMES.get(sleeve_id, f"sleeve_{sleeve_id}")
                        logger.info("OOB KILL SWITCH: Sleeve %s (id=%d) RESUMED", name, sleeve_id)

        except Exception:
            logger.exception("Error reading kill switch shared memory")
