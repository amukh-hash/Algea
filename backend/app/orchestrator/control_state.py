"""Thread-safe in-memory control state for runtime overrides.

This singleton holds operator overrides (pause, vol regime, blocked symbols, etc.)
that the orchestrator and API read in real time. All mutations are thread-safe and
logged to an internal audit trail.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class _ControlState:
    paused: bool = False
    vol_regime_override: str | None = None  # "CRASH_RISK" | "CAUTION" | None
    blocked_symbols: set[str] = field(default_factory=set)
    frozen_sleeves: set[str] = field(default_factory=set)
    gross_exposure_cap: float | None = None
    execution_mode: str = "paper"  # "noop" | "paper" | "ibkr"
    audit_log: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _audit(self, action: str, detail: dict[str, Any] | None = None) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "detail": detail or {},
        }
        self.audit_log.append(entry)
        # Keep only last 500 entries
        if len(self.audit_log) > 500:
            self.audit_log = self.audit_log[-500:]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "paused": self.paused,
                "vol_regime_override": self.vol_regime_override,
                "blocked_symbols": sorted(self.blocked_symbols),
                "frozen_sleeves": sorted(self.frozen_sleeves),
                "gross_exposure_cap": self.gross_exposure_cap,
                "execution_mode": self.execution_mode,
            }

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self.paused = paused
            self._audit("pause" if paused else "resume")

    def set_vol_regime(self, regime: str | None) -> None:
        with self._lock:
            self.vol_regime_override = regime
            self._audit("vol_regime_override", {"regime": regime})

    def set_blocked_symbols(self, symbols: list[str]) -> None:
        with self._lock:
            self.blocked_symbols = set(s.upper() for s in symbols)
            self._audit("blocked_symbols", {"symbols": sorted(self.blocked_symbols)})

    def set_frozen_sleeves(self, sleeves: list[str]) -> None:
        with self._lock:
            self.frozen_sleeves = set(sleeves)
            self._audit("frozen_sleeves", {"sleeves": sorted(self.frozen_sleeves)})

    def set_exposure_cap(self, cap: float | None) -> None:
        with self._lock:
            self.gross_exposure_cap = cap
            self._audit("exposure_cap", {"cap": cap})

    def set_execution_mode(self, mode: str) -> None:
        if mode not in ("noop", "paper", "ibkr"):
            raise ValueError(f"Invalid mode: {mode}")
        with self._lock:
            self.execution_mode = mode
            self._audit("execution_mode", {"mode": mode})

    def get_audit(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            return list(reversed(self.audit_log[-limit:]))


# Module-level singleton
control_state = _ControlState()
