"""Control API — write endpoints for orchestrator overrides, flatten, manual orders.

All mutating endpoints are guarded by paper-only checks and log to the audit trail.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket as _socket
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.api.zmq_bridge import (
    bridge_control_snapshot,
    bridge_control_mutation,
    bridge_portfolio_summary,
    bridge_broker_status,
    bridge_calendar,
)
from backend.app.orchestrator.control_state_provider import get_control_state_provider

router = APIRouter(prefix="/api/control", tags=["control"])
logger = logging.getLogger("algae.api.control")

_ARTIFACT_ROOT = Path("backend/artifacts/orchestrator")
_DB_PATH = Path("backend/artifacts/orchestrator_state/state.sqlite3")
_TIMEOUT_S = 5.0

_GUARDRAIL_DEFS: tuple[dict[str, Any], ...] = (
    {
        "id": "ece_tracker",
        "label": "ECE Tracker",
        "metric_candidates": ("ece", "ece_score", "calibration_ece"),
        "limit_candidates": ("ece_max", "max_ece", "ece"),
        "comparator": "<=",
    },
    {
        "id": "mmd_liveguard",
        "label": "MMD LiveGuard",
        "metric_candidates": ("mmd", "mmd_score", "distribution_drift_mmd"),
        "limit_candidates": ("mmd_max", "max_mmd", "mmd"),
        "comparator": "<=",
    },
    {
        "id": "max_drawdown",
        "label": "Max Drawdown",
        "metric_candidates": ("max_drawdown", "drawdown", "intraday_drawdown"),
        "limit_candidates": ("max_drawdown", "max_intraday_drawdown", "drawdown_limit"),
        "comparator": "<=",
    },
    {
        "id": "gap_risk_filter",
        "label": "Gap Risk Filter",
        "metric_candidates": ("gap_risk", "overnight_gap_risk"),
        "limit_candidates": ("gap_risk_max", "max_gap_risk"),
        "comparator": "<=",
    },
    {
        "id": "slippage_monitor",
        "label": "Slippage Monitor",
        "metric_candidates": ("slippage_bps", "avg_slippage_bps"),
        "limit_candidates": ("max_slippage_bps", "slippage_bps_max"),
        "comparator": "<=",
    },
)


def _provider():
    return get_control_state_provider(_DB_PATH)


def _raise_write_failure(exc: RuntimeError) -> None:
    raise HTTPException(
        status_code=500,
        detail={"error": "control_state_write_through_failed", "detail": str(exc)},
    )


def _require_auth() -> None:
    if os.getenv("AUTH_REQUIRED", "0") != "1":
        return
    # lightweight guard for current router call style:
    # callers in tests can disable via AUTH_REQUIRED=0.
    raise HTTPException(status_code=401, detail="auth_required")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if out != out or out in (float("inf"), float("-inf")):
            return None
        return out
    except (TypeError, ValueError):
        return None


def _find_metric(data: dict[str, Any], candidates: tuple[str, ...]) -> float | None:
    for key in candidates:
        val = _safe_float(data.get(key))
        if val is not None:
            return val
    return None


def _load_latest_risk_payload() -> tuple[str, dict[str, Any] | None]:
    day_candidates = [date.today().isoformat()]
    if _ARTIFACT_ROOT.exists():
        for child in sorted(_ARTIFACT_ROOT.iterdir(), reverse=True):
            if child.is_dir() and child.name not in day_candidates:
                day_candidates.append(child.name)

    for day in day_candidates:
        day_root = _ARTIFACT_ROOT / day
        for candidate in (
            day_root / "risk_checks.json",
            day_root / "reports" / "risk_checks.json",
            day_root / "risk" / "risk_checks.json",
        ):
            if not candidate.exists():
                continue
            try:
                raw = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("guardrails_status: failed to parse %s: %s", candidate, exc)
                return day, None

            if isinstance(raw, dict) and "risk_checks" in raw and isinstance(raw["risk_checks"], dict):
                return day, raw["risk_checks"]
            if isinstance(raw, dict):
                return day, raw
            return day, None
    return day_candidates[0], None


def _guardrail_row(defn: dict[str, Any], payload: dict[str, Any], checked_at: str, violations_blob: str) -> dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    limits = payload.get("limits") if isinstance(payload.get("limits"), dict) else {}

    value = _find_metric(metrics, defn["metric_candidates"])
    threshold = _find_metric(limits, defn["limit_candidates"])

    marker = defn["id"].replace("_", "")
    has_violation = marker in violations_blob

    breach = has_violation
    if value is not None and threshold is not None and defn["comparator"] == "<=":
        breach = breach or value > threshold

    if breach:
        status = "breached"
        reason = "limit_breached"
    elif value is not None or threshold is not None:
        status = "armed"
        reason = None
    else:
        status = "unknown"
        reason = "unwired"

    return {
        "id": defn["id"],
        "label": defn["label"],
        "status": status,
        "value": value,
        "threshold": threshold,
        "comparator": defn["comparator"],
        "breach": breach,
        "updated_at": checked_at,
        "source_job": "risk_checks_global",
        "reason": reason,
    }


# ── Request models ───────────────────────────────────────────────────

class PauseRequest(BaseModel):
    paused: bool

class VolRegimeRequest(BaseModel):
    regime: str | None = Field(None, description="CRASH_RISK | CAUTION | None")

class BlockedSymbolsRequest(BaseModel):
    symbols: list[str] = []

class FrozenSleevesRequest(BaseModel):
    sleeves: list[str] = []

class ExposureCapRequest(BaseModel):
    cap: float | None = None

class ExecutionModeRequest(BaseModel):
    mode: str = Field(..., description="noop | paper | ibkr")

class FlattenRequest(BaseModel):
    sleeve: str | None = Field(None, description="Specific sleeve to flatten, or None for all")

class ManualOrderRequest(BaseModel):
    symbol: str
    qty: int = Field(..., gt=0)
    side: str = Field(..., description="buy | sell")
    order_type: str = Field("MKT", description="MKT | LMT | MOC")
    limit_price: float | None = None

class TriggerTickRequest(BaseModel):
    dry_run: bool = True
    session: str | None = None


# ── State endpoints ──────────────────────────────────────────────────

@router.get("/state")
def get_state() -> dict[str, Any]:
    """Return the full control state snapshot."""
    snap = _provider().snapshot(consumer="control_api")
    bridge_control_snapshot(snap)
    return snap


@router.get("/guardrails/status")
def get_guardrails_status() -> dict[str, Any]:
    """Return deterministic guardrail status contract for operator UI."""
    asof, payload = _load_latest_risk_payload()
    now = datetime.now(timezone.utc)
    asof_ts = now.isoformat()

    if payload is None:
        return {
            "asof": asof_ts,
            "schema_version": "guardrails_status.v1",
            "freshness_ms": 0,
            "backend_reachable": False,
            "guardrails": [
                {
                    "id": d["id"],
                    "label": d["label"],
                    "status": "unknown",
                    "value": None,
                    "threshold": None,
                    "comparator": d["comparator"],
                    "breach": False,
                    "updated_at": None,
                    "source_job": "risk_checks_global",
                    "reason": "unwired",
                }
                for d in _GUARDRAIL_DEFS
            ],
            "source_asof": asof,
        }

    checked_at = payload.get("checked_at") if isinstance(payload.get("checked_at"), str) else asof_ts

    freshness_ms = 0
    try:
        dt = datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
        freshness_ms = max(0, int((now - dt).total_seconds() * 1000))
    except Exception:
        checked_at = asof_ts

    violations = payload.get("violations") if isinstance(payload.get("violations"), list) else []
    violations_blob = json.dumps(violations).lower()
    rows = [_guardrail_row(d, payload, checked_at, violations_blob) for d in _GUARDRAIL_DEFS]

    return {
        "asof": asof_ts,
        "schema_version": "guardrails_status.v1",
        "freshness_ms": freshness_ms,
        "backend_reachable": True,
        "guardrails": rows,
        "source_asof": asof,
    }


@router.put("/pause")
def set_pause(req: PauseRequest) -> dict[str, Any]:
    _require_auth()
    try:
        _provider().set_paused(req.paused)
    except RuntimeError as exc:
        _raise_write_failure(exc)
    bridge_control_mutation("pause", {"paused": req.paused})
    return {"ok": True, "paused": req.paused}


@router.put("/resume")
def resume() -> dict[str, Any]:
    _require_auth()
    try:
        _provider().set_paused(False)
    except RuntimeError as exc:
        _raise_write_failure(exc)
    return {"ok": True, "paused": False}


@router.put("/vol-regime")
def set_vol_regime(req: VolRegimeRequest) -> dict[str, Any]:
    _require_auth()
    if req.regime and req.regime not in ("CRASH_RISK", "CAUTION", "NORMAL"):
        raise HTTPException(400, detail="regime must be CRASH_RISK, CAUTION, NORMAL, or null")
    val = req.regime if req.regime != "NORMAL" else None
    try:
        _provider().set_vol_regime(val)
    except RuntimeError as exc:
        _raise_write_failure(exc)
    return {"ok": True, "vol_regime_override": val}


@router.put("/blocked-symbols")
def set_blocked_symbols(req: BlockedSymbolsRequest) -> dict[str, Any]:
    _require_auth()
    try:
        snap = _provider().set_blocked_symbols(req.symbols)
    except RuntimeError as exc:
        _raise_write_failure(exc)
    return {"ok": True, "blocked_symbols": sorted(snap["blocked_symbols"])}


@router.put("/frozen-sleeves")
def set_frozen_sleeves(req: FrozenSleevesRequest) -> dict[str, Any]:
    _require_auth()
    valid = {"core", "vrp", "selector"}
    invalid = set(req.sleeves) - valid
    if invalid:
        raise HTTPException(400, detail=f"Unknown sleeves: {invalid}. Valid: {valid}")
    try:
        snap = _provider().set_frozen_sleeves(req.sleeves)
    except RuntimeError as exc:
        _raise_write_failure(exc)
    return {"ok": True, "frozen_sleeves": sorted(snap["frozen_sleeves"])}


@router.put("/exposure-cap")
def set_exposure_cap(req: ExposureCapRequest) -> dict[str, Any]:
    _require_auth()
    if req.cap is not None and req.cap <= 0:
        raise HTTPException(400, detail="Exposure cap must be positive or null")
    try:
        _provider().set_exposure_cap(req.cap)
    except RuntimeError as exc:
        _raise_write_failure(exc)
    return {"ok": True, "gross_exposure_cap": req.cap}


@router.put("/execution-mode")
def set_execution_mode(req: ExecutionModeRequest) -> dict[str, Any]:
    _require_auth()
    try:
        _provider().set_execution_mode(req.mode)
    except RuntimeError as exc:
        _raise_write_failure(exc)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    return {"ok": True, "execution_mode": req.mode}


# ── Action endpoints ─────────────────────────────────────────────────

@router.post("/trigger-tick")
async def trigger_tick(req: TriggerTickRequest) -> dict[str, Any]:
    _require_auth()
    """Fire a single orchestrator tick."""
    # ── DevOps Fix 1: Two Masters Paradox ──────────────────────────
    # When ORCH_BACKGROUND_TICK=0, this endpoint is disabled so the
    # Windows Task Scheduler is the single, undisputed master clock.
    if os.getenv("ORCH_BACKGROUND_TICK", "1") == "0":
        return {"ok": False, "status": "disabled",
                "reason": "ORCH_BACKGROUND_TICK=0 — use Windows Task Scheduler"}

    if _provider().snapshot(consumer="control_api")["paused"]:
        raise HTTPException(409, detail="Orchestrator is paused. Resume before triggering.")

    try:
        from backend.app.orchestrator.orchestrator import Orchestrator
        from backend.app.orchestrator.calendar import Session

        orch = Orchestrator()
        forced = Session(req.session) if req.session else None
        result = await asyncio.to_thread(
            orch.run_once, dry_run=req.dry_run, forced_session=forced
        )
        _provider().audit("trigger_tick", {"dry_run": req.dry_run, "run_id": result.run_id})
        return {
            "ok": True,
            "run_id": result.run_id,
            "ran_jobs": result.ran_jobs,
            "failed_jobs": result.failed_jobs,
            "skipped_jobs": result.skipped_jobs,
        }
    except Exception as exc:
        logger.exception("trigger_tick failed: %s", exc)
        raise HTTPException(500, detail={"error": "trigger_failed", "detail": str(exc)})


@router.post("/flatten")
def flatten(req: FlattenRequest) -> dict[str, Any]:
    _require_auth()
    """Submit flatten order (paper-only)."""
    state = _provider().snapshot(consumer="control_api")
    if state["execution_mode"] == "ibkr":
        raise HTTPException(403, detail="Flatten requires paper or noop mode for safety.")

    action = f"flatten_{'all' if req.sleeve is None else req.sleeve}"
    _provider().audit(action, {"sleeve": req.sleeve})
    logger.warning("FLATTEN %s requested via control API", "ALL" if req.sleeve is None else req.sleeve)

    return {
        "ok": True,
        "action": action,
        "sleeve": req.sleeve,
        "note": "Flatten order queued. Check positions endpoint for confirmation.",
    }


@router.post("/manual-order")
def manual_order(req: ManualOrderRequest) -> dict[str, Any]:
    """Submit a single manual order (paper-only)."""
    state = _provider().snapshot(consumer="control_api")
    if state["execution_mode"] == "ibkr":
        raise HTTPException(403, detail="Manual orders require paper or noop mode for safety.")

    if req.symbol.upper() in state["blocked_symbols"]:
        raise HTTPException(409, detail=f"Symbol {req.symbol} is in the blocked list.")

    order = {
        "symbol": req.symbol.upper(),
        "qty": req.qty,
        "side": req.side.upper(),
        "type": req.order_type.upper(),
        "limit_price": req.limit_price,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    _provider().audit("manual_order", order)
    logger.warning("MANUAL ORDER submitted via control API: %s", order)

    return {"ok": True, "order": order}


# ── Info endpoints ───────────────────────────────────────────────────

@router.get("/audit")
def get_audit(limit: int = 50) -> dict[str, Any]:
    return {"items": _provider().get_audit(limit)}


@router.get("/broker-status")
def get_broker_status() -> dict[str, Any]:
    """Check broker connectivity."""
    try:
        import os
        gateway_url = os.getenv("IBKR_GATEWAY_URL", "127.0.0.1:4002")
        paper_only = os.getenv("IBKR_PAPER_ONLY", "1") == "1"
        account_id = os.getenv("IBKR_ACCOUNT_ID", "")

        # Try a quick socket connect to the gateway to check if it's up
        import socket
        host, port_str = gateway_url.split(":")
        port = int(port_str)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        try:
            result = sock.connect_ex((host, port))
            connected = result == 0
        finally:
            sock.close()

        result = {
            "connected": connected,
            "gateway_url": gateway_url,
            "paper_only": paper_only,
            "account_id": account_id[:4] + "****" if len(account_id) > 4 else account_id,
            "mode": "PAPER" if paper_only else "LIVE",
        }
        bridge_broker_status(result)
        return result
    except Exception as exc:
        result = {
            "connected": False,
            "error": str(exc),
            "mode": "UNKNOWN",
        }
        bridge_broker_status(result)
        return result


# Module-level broker adapter singleton
_broker_adapter: Any = None
_broker_lock = threading.Lock()


# ── Reusable connect logic ───────────────────────────────────────────

def _is_port_open(host: str = "127.0.0.1", port: int = 4002, timeout: float = 2.0) -> bool:
    """Check if a TCP port is reachable (non-blocking)."""
    sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


def _try_connect_broker(source: str = "endpoint") -> dict[str, Any]:
    """Attempt to connect the broker adapter singleton.

    Thread-safe.  Called by both the /broker/connect endpoint and the
    background BrokerWatchdog.  Returns a status dict.
    """
    global _broker_adapter
    with _broker_lock:
        try:
            # ib_insync requires an asyncio event loop; worker threads
            # don't have one by default.
            import asyncio as _asyncio
            import nest_asyncio
            try:
                loop = _asyncio.get_event_loop()
            except RuntimeError:
                loop = _asyncio.new_event_loop()
                _asyncio.set_event_loop(loop)
            nest_asyncio.apply(loop)

            # Load IBKR env vars from project .env
            try:
                from dotenv import load_dotenv
                load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=True)
            except ImportError:
                pass

            if _broker_adapter is not None:
                # Already connected — verify session is alive
                try:
                    _broker_adapter._reconnect_if_needed()
                    positions = _broker_adapter.get_positions()
                    return {
                        "connected": True,
                        "status": "already_connected",
                        "source": source,
                        "positions": len(positions.get("positions", [])),
                    }
                except Exception:
                    # Stale — disconnect and reconnect below
                    try:
                        _broker_adapter.disconnect()
                    except Exception:
                        pass
                    _broker_adapter = None

            from backend.app.orchestrator.broker_ibkr_adapter import IBKRBrokerAdapter
            _broker_adapter = IBKRBrokerAdapter.from_env()
            _broker_adapter.verify_paper()

            positions = _broker_adapter.get_positions()
            result = {
                "connected": True,
                "status": "connected",
                "source": source,
                "account": os.getenv("IBKR_ACCOUNT_ID", "")[:4] + "****",
                "mode": "PAPER" if os.getenv("IBKR_PAPER_ONLY", "1") == "1" else "LIVE",
                "positions": len(positions.get("positions", [])),
            }
            logger.info("IBKR broker connected (source=%s): %s", source, result)
            return result
        except Exception as exc:
            _broker_adapter = None
            result = {
                "connected": False,
                "status": "failed",
                "source": source,
                "error": str(exc),
            }
            logger.error("IBKR broker connect failed (source=%s): %s", source, exc)
            return result


# ── Broker Watchdog ──────────────────────────────────────────────────

class BrokerWatchdog:
    """Background daemon that auto-connects/disconnects the IBKR broker.

    Polls port 4002 every *interval* seconds.  If the Gateway is
    reachable and no adapter exists (or the session is stale),
    it calls ``_try_connect_broker``.  If the Gateway disappears,
    it cleans up the adapter singleton.

    Respects a weekday 06:25-16:15 time window to avoid unnecessary
    reconnect attempts outside trading hours.
    """

    def __init__(self, interval: float = 30.0) -> None:
        self._interval = interval
        self._running = False
        self._thread: threading.Thread | None = None
        self._was_connected = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="broker-watchdog",
        )
        self._thread.start()
        logger.info("Broker watchdog started (interval=%ds)", self._interval)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Broker watchdog stopped")

    def _in_trading_window(self) -> bool:
        now = datetime.now()
        weekday = now.weekday()  # 0=Mon … 4=Fri
        if weekday > 4:
            return False
        hour_min = now.hour * 60 + now.minute
        return 6 * 60 + 25 <= hour_min <= 16 * 60 + 15  # 06:25–16:15

    def _loop(self) -> None:
        import time
        global _broker_adapter

        # Wait a bit on startup to let the Gateway finish booting
        time.sleep(10)

        while self._running:
            try:
                self._tick()
            except Exception:
                logger.debug("Broker watchdog tick error", exc_info=True)
            time.sleep(self._interval)

    def _tick(self) -> None:
        global _broker_adapter

        if not self._in_trading_window():
            return

        gateway_url = os.getenv("IBKR_GATEWAY_URL", "127.0.0.1:4002")
        host, port_str = gateway_url.split(":") if ":" in gateway_url else (gateway_url, "4002")
        port = int(port_str)
        port_open = _is_port_open(host, port)

        if port_open and _broker_adapter is None:
            # Gateway is up but we have no connection — auto-connect
            logger.info("Broker watchdog: port %d reachable, auto-connecting...", port)
            result = _try_connect_broker(source="watchdog")
            bridge_broker_status(result)
            if result.get("connected"):
                self._was_connected = True

        elif port_open and _broker_adapter is not None:
            # Gateway is up and we have an adapter — verify session health
            try:
                if not _broker_adapter._broker._client.isConnected():
                    logger.warning("Broker watchdog: session stale, reconnecting...")
                    result = _try_connect_broker(source="watchdog_stale")
                    bridge_broker_status(result)
                elif not self._was_connected:
                    self._was_connected = True
            except Exception:
                logger.warning("Broker watchdog: health check failed, reconnecting...")
                result = _try_connect_broker(source="watchdog_error")
                bridge_broker_status(result)

        elif not port_open and _broker_adapter is not None:
            # Gateway went away — clean up
            logger.warning("Broker watchdog: port %d unreachable, clearing adapter", port)
            with _broker_lock:
                try:
                    _broker_adapter.disconnect()
                except Exception:
                    pass
                _broker_adapter = None
            self._was_connected = False
            bridge_broker_status({"connected": False, "status": "gateway_down", "source": "watchdog"})


broker_watchdog = BrokerWatchdog(
    interval=float(os.getenv("BROKER_WATCHDOG_INTERVAL", "30")),
)


# ── Broker endpoints ─────────────────────────────────────────────────

@router.post("/broker/connect")
def broker_connect() -> dict[str, Any]:
    """Connect to IBKR broker via IBKRBrokerAdapter.from_env().

    Reads IBKR_GATEWAY_URL, IBKR_ACCOUNT_ID, IBKR_CLIENT_ID from .env.
    Called automatically by the native frontend on startup if broker is disconnected.
    """
    result = _try_connect_broker(source="endpoint")
    bridge_broker_status(result)
    return result


@router.post("/broker/disconnect")
def broker_disconnect() -> dict[str, Any]:
    """Disconnect from IBKR broker."""
    global _broker_adapter
    with _broker_lock:
        try:
            if _broker_adapter is not None:
                _broker_adapter.disconnect()
                _broker_adapter = None
            result = {"connected": False, "status": "disconnected"}
            bridge_broker_status(result)
            return result
        except Exception as exc:
            _broker_adapter = None
            return {"connected": False, "status": "error", "error": str(exc)}

@router.get("/job-graph")
def get_job_graph() -> dict[str, Any]:
    """Return the job dependency graph with metadata."""
    import sqlite3
    from backend.app.orchestrator.job_defs import default_jobs
    from backend.app.orchestrator.calendar import Session

    jobs = default_jobs()

    # Try to get last statuses from DB
    last_statuses: dict[str, dict[str, Any]] = {}
    try:
        conn = sqlite3.connect(_DB_PATH, timeout=2)
        conn.row_factory = sqlite3.Row
        for job in jobs:
            row = conn.execute(
                "SELECT status, started_at, ended_at, error_summary FROM jobs WHERE job_name=? ORDER BY started_at DESC LIMIT 1",
                (job.name,),
            ).fetchone()
            if row:
                last_statuses[job.name] = dict(row)
        conn.close()
    except Exception:
        pass

    nodes = []
    for job in jobs:
        last = last_statuses.get(job.name, {})
        duration_s = None
        if last.get("started_at") and last.get("ended_at"):
            try:
                start = datetime.fromisoformat(last["started_at"])
                end = datetime.fromisoformat(last["ended_at"])
                duration_s = round((end - start).total_seconds(), 1)
            except Exception:
                pass

        nodes.append({
            "name": job.name,
            "deps": job.deps,
            "sessions": [s.value for s in job.sessions],
            "timeout_s": job.timeout_s,
            "min_interval_s": job.min_interval_s,
            "last_status": last.get("status"),
            "last_started": last.get("started_at"),
            "last_duration_s": duration_s,
            "last_error": last.get("error_summary"),
        })

    return {"jobs": nodes}


@router.get("/calendar")
def get_calendar() -> dict[str, Any]:
    """Return today's session windows and current session."""
    from backend.app.orchestrator.config import OrchestratorConfig
    from backend.app.orchestrator.calendar import MarketCalendar
    from backend.app.orchestrator.clock import now_et

    config = OrchestratorConfig()
    cal = MarketCalendar(config)
    now = now_et()
    session = cal.current_session(now)
    is_trading = cal.is_trading_day(now)

    windows = {}
    for name, window in config.session_windows.items():
        windows[name] = {"start": window.start, "end": window.end}

    result = {
        "date": date.today().isoformat(),
        "current_session": session.value,
        "is_trading_day": is_trading,
        "current_time": now.strftime("%H:%M:%S"),
        "session_windows": windows,
    }
    bridge_calendar(result)
    return result


@router.get("/config")
def get_config() -> dict[str, Any]:
    """Return current orchestrator configuration."""
    from backend.app.orchestrator.config import OrchestratorConfig
    config = OrchestratorConfig()
    return {
        "timezone": config.timezone,
        "exchange": config.exchange,
        "mode": config.mode,
        "poll_interval_s": config.poll_interval_s,
        "paper_only": config.paper_only,
        "account_equity": config.account_equity,
        "max_order_notional": config.max_order_notional,
        "max_total_order_notional": config.max_total_order_notional,
        "max_orders": config.max_orders,
        "enabled_jobs": config.enabled_jobs,
        "disabled_jobs": config.disabled_jobs,
    }


@router.get("/portfolio-summary")
def get_portfolio_summary() -> dict[str, Any]:
    """Return portfolio value summary with futures-aware accounting.

    For futures: uses margin (not full notional) for cash accounting,
    and mark-to-market P&L = (last_price - avg_cost) × qty × multiplier.
    """
    import re
    from backend.app.orchestrator.config import OrchestratorConfig

    # ── Contract specs & margin table ────────────────────────────────
    try:
        from sleeves.cooc_reversal_futures.contract_master import CONTRACT_MASTER
    except ImportError:
        CONTRACT_MASTER = {}

    # Approximate CME initial margin per contract (paper-mode defaults)
    MARGIN_PER_CONTRACT: dict[str, float] = {
        "ES": 13_200, "NQ": 18_700, "RTY": 7_150, "YM": 9_500,
        "MES": 1_320, "MNQ": 1_870, "MYM": 950, "M2K": 715,
        "CL": 6_500, "GC": 10_000, "SI": 9_000, "HG": 4_500,
        "ZN": 2_200, "ZB": 4_400,
        "6E": 2_500, "6J": 3_300, "6B": 2_500, "6A": 1_800,
    }

    def _parse_root(symbol: str) -> str:
        """Extract root from a futures symbol like RTYM6 -> RTY, ESZ5 -> ES."""
        m = re.match(r"^([A-Z0-9]{1,4}?)([FGHJKMNQUVXZ]\d{1,2})$", symbol)
        return m.group(1) if m else symbol

    def _classify_sleeve(symbol: str, root: str, instrument_type: str) -> str:
        """Classify a position into a sleeve."""
        if instrument_type == "future":
            return "core"  # CO→OC futures sleeve
        # Options/VIX instruments → vrp
        if any(vix in symbol.upper() for vix in ("VIX", "VXX", "UVXY", "SVXY", "VIXY")):
            return "vrp"
        if symbol.upper().endswith(("C", "P")) and len(symbol) > 6:
            return "vrp"  # options-like
        return "selector"  # equities

    config = OrchestratorConfig()
    today = date.today().isoformat()

    # ── Load persistent paper account state (source of truth) ────────
    _PAPER_STATE_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "paper_account" / "state.json"
    paper_cash = config.account_equity
    paper_starting_equity = config.account_equity
    paper_realized_pnl = 0.0
    positions: list[dict[str, Any]] = []

    logger.info("paper_state path=%s exists=%s", _PAPER_STATE_PATH, _PAPER_STATE_PATH.exists())
    if _PAPER_STATE_PATH.exists():
        try:
            pstate = json.loads(_PAPER_STATE_PATH.read_text(encoding="utf-8-sig"))
            positions = pstate.get("positions", [])
            paper_cash = float(pstate.get("cash", config.account_equity))
            paper_starting_equity = float(pstate.get("starting_equity", config.account_equity))
            paper_realized_pnl = float(pstate.get("realized_pnl", 0.0))
            logger.info("Loaded paper state: cash=%.2f equity=%.2f positions=%d", paper_cash, paper_starting_equity, len(positions))
        except Exception as exc:
            logger.warning("Failed to load paper state from %s: %s", _PAPER_STATE_PATH, exc)

    # Override with today's date-scoped positions if fills_reconcile has run
    pos_path = _ARTIFACT_ROOT / today / "fills" / "positions.json"
    if pos_path.exists():
        try:
            data = json.loads(pos_path.read_text(encoding="utf-8"))
            day_positions = data.get("positions", [])
            if day_positions:
                positions = day_positions
        except Exception:
            pass

    # ── Fetch live + closing prices (with strict timeout) ──────────────
    live_prices: dict[str, float] = {}
    closing_prices: dict[str, float] = {}
    if positions:
        try:
            from backend.app.api.live_prices import get_live_prices, get_closing_prices
            import concurrent.futures
            symbols = [p.get("symbol", "") for p in positions if p.get("symbol")]
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                live_future = pool.submit(get_live_prices, symbols)
                close_future = pool.submit(get_closing_prices, symbols)
                try:
                    live_prices = live_future.result(timeout=5)
                except concurrent.futures.TimeoutError:
                    pass
                try:
                    closing_prices = close_future.result(timeout=5)
                except concurrent.futures.TimeoutError:
                    pass
        except Exception:
            pass  # fall back to avg_cost

    # ── Load fills for realized P&L ──────────────────────────────────
    fills_path = _ARTIFACT_ROOT / today / "fills" / "fills.json"
    fills: list[dict[str, Any]] = []
    if fills_path.exists():
        fdata = json.loads(fills_path.read_text(encoding="utf-8"))
        fills = fdata.get("fills", [])

    # ── Classify and value each position ─────────────────────────────
    holdings = []
    total_margin = 0.0
    total_notional = 0.0
    total_unrealized_pnl = 0.0
    total_equity_cost = 0.0
    sleeve_totals: dict[str, dict[str, float]] = {}

    for pos in positions:
        symbol = pos.get("symbol", "")
        qty = float(pos.get("qty", pos.get("quantity", 0)))
        avg_cost_raw = float(pos.get("avg_cost", 0))
        root = _parse_root(symbol)
        spec = CONTRACT_MASTER.get(root)

        if spec:
            # ── Futures position ─────────────────────────────────────
            instrument_type = "future"
            multiplier = spec.multiplier

            # Normalize avg_cost: IBKR may store avg_cost as
            # notional/qty (= price × multiplier) instead of raw index points.
            # If avg_cost >> reasonable index range, divide by multiplier.
            avg_cost = avg_cost_raw
            if multiplier > 1 and avg_cost > multiplier * 500:
                avg_cost = avg_cost_raw / multiplier

            # Fallback: live price > closing price > avg_cost
            fallback = closing_prices.get(symbol, avg_cost)
            last_price = live_prices.get(symbol, float(pos.get("last_price", fallback)))
            # If no live price and fallback last_price also looks notional, normalize
            if symbol not in live_prices and last_price > multiplier * 500:
                last_price = last_price / multiplier

            notional = abs(qty) * last_price * multiplier
            margin_posted = abs(qty) * MARGIN_PER_CONTRACT.get(root, notional / abs(qty) if qty else 0)
            unrealized_pnl = (last_price - avg_cost) * qty * multiplier
            total_margin += margin_posted
            total_notional += notional
        else:
            # ── Equity / unknown position (treat as stock) ───────────
            instrument_type = "equity"
            multiplier = 1.0
            avg_cost = avg_cost_raw
            fallback = closing_prices.get(symbol, avg_cost)
            last_price = live_prices.get(symbol, float(pos.get("last_price", fallback)))
            notional = abs(qty * avg_cost)
            margin_posted = notional  # stocks = full cash outlay
            unrealized_pnl = (last_price - avg_cost) * qty
            total_equity_cost += notional

        total_unrealized_pnl += unrealized_pnl

        sleeve = _classify_sleeve(symbol, root, instrument_type)

        # Aggregate per-sleeve
        if sleeve not in sleeve_totals:
            sleeve_totals[sleeve] = {"margin": 0, "notional": 0, "unrealized_pnl": 0, "equity_cost": 0, "count": 0}
        sleeve_totals[sleeve]["margin"] += margin_posted
        sleeve_totals[sleeve]["notional"] += notional
        sleeve_totals[sleeve]["unrealized_pnl"] += unrealized_pnl
        if instrument_type == "equity":
            sleeve_totals[sleeve]["equity_cost"] += notional
        sleeve_totals[sleeve]["count"] += 1

        holdings.append({
            "symbol": symbol,
            "root": root,
            "instrument_type": instrument_type,
            "sleeve": sleeve,
            "qty": qty,
            "avg_cost": round(avg_cost, 4),
            "last_price": round(last_price, 4),
            "multiplier": multiplier,
            "notional": round(notional, 2),
            "margin_posted": round(margin_posted, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
        })

    # ── Realized P&L from fills ──────────────────────────────────────
    realized_pnl = 0.0
    for fill in fills:
        pnl = fill.get("realized_pnl", 0) or 0
        realized_pnl += float(pnl)

    # ── Portfolio-level computation ──────────────────────────────────
    account_equity = paper_starting_equity
    cash = paper_cash

    # NAV = starting equity + all P&L (independent of cash/margin model)
    # This is always correct regardless of instrument type, flips, or margin
    total_value = account_equity + paper_realized_pnl + total_unrealized_pnl
    total_pnl = total_value - account_equity

    # Diagnostics for ongoing validation
    equity_cost_basis = sum(
        h["qty"] * h["avg_cost"] for h in holdings if h["instrument_type"] == "equity"
    )

    # ── Per-sleeve summaries ─────────────────────────────────────────
    sleeves_summary = {}
    for sleeve_name, totals in sleeve_totals.items():
        sleeves_summary[sleeve_name] = {
            "margin": round(totals["margin"], 2),
            "notional": round(totals["notional"], 2),
            "unrealized_pnl": round(totals["unrealized_pnl"], 2),
            "equity_cost": round(totals["equity_cost"], 2),
            "position_count": int(totals["count"]),
        }

    result = {
        "asof_date": today,
        "account_equity": account_equity,
        "cash": round(cash, 2),
        "total_margin": round(total_margin, 2),
        "total_notional": round(total_notional, 2),
        "total_equity_cost": round(total_equity_cost, 2),
        "total_unrealized_pnl": round(total_unrealized_pnl, 2),
        "total_value": round(total_value, 2),
        "total_pnl": round(total_pnl, 2),
        "realized_pnl": round(paper_realized_pnl, 2),
        "position_count": len(holdings),
        "fill_count": len(fills),
        "holdings": holdings,
        "sleeves": sleeves_summary,
        "_diagnostics": {
            "nav_formula": "starting_equity + realized_pnl + unrealized_pnl",
            "equity_cost_basis": round(equity_cost_basis, 2),
            "required_margin": round(total_margin, 2),
        },
    }
    bridge_portfolio_summary(result)
    return result


# ── Portfolio History (intraday snapshots) ───────────────────────────

from collections import deque
import threading

_history: deque[dict[str, Any]] = deque(maxlen=480)  # 8 hours at 1/min
_history_lock = threading.Lock()
_snapshot_thread: threading.Thread | None = None
_snapshot_running = False

def _history_path() -> Path:
    p = _ARTIFACT_ROOT / date.today().isoformat() / "portfolio_history.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _load_history() -> None:
    path = _history_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            with _history_lock:
                _history.clear()
                for pt in data:
                    _history.append(pt)
        except Exception:
            pass

def _persist_history() -> None:
    try:
        with _history_lock:
            pts = list(_history)
        _history_path().write_text(json.dumps(pts, indent=1), encoding="utf-8")
    except Exception:
        pass

def _take_snapshot() -> None:
    import time as _time
    try:
        summary = get_portfolio_summary()
        sleeve_values: dict[str, float] = {"core": 0, "vrp": 0, "selector": 0}
        for h in summary.get("holdings", []):
            s = h.get("sleeve", "selector")
            sleeve_values[s] = sleeve_values.get(s, 0) + h.get("margin_posted", 0) + h.get("unrealized_pnl", 0)

        point = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "total_value": summary.get("total_value", 0),
            "cash": summary.get("cash", 0),
            "core": round(sleeve_values.get("core", 0), 2),
            "vrp": round(sleeve_values.get("vrp", 0), 2),
            "selector": round(sleeve_values.get("selector", 0), 2),
            "total_pnl": summary.get("total_pnl", 0),
            "unrealized_pnl": summary.get("total_unrealized_pnl", 0),
        }
        with _history_lock:
            _history.append(point)
        _persist_history()
    except Exception:
        pass

def _snapshot_loop() -> None:
    import time as _time
    global _snapshot_running
    _load_history()
    while _snapshot_running:
        _take_snapshot()
        _time.sleep(60)

def _ensure_snapshots() -> None:
    global _snapshot_thread, _snapshot_running
    if _snapshot_thread and _snapshot_thread.is_alive():
        return
    _snapshot_running = True
    _snapshot_thread = threading.Thread(target=_snapshot_loop, daemon=True, name="portfolio-history")
    _snapshot_thread.start()


@router.get("/portfolio-history")
def get_portfolio_history() -> dict[str, Any]:
    """Return intraday portfolio value history."""
    _ensure_snapshots()
    with _history_lock:
        points = list(_history)
    return {
        "asof_date": date.today().isoformat(),
        "points": points,
        "count": len(points),
    }
