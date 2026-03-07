import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from backend.app.version import APP_DISPLAY
from backend.app.core.logging_config import configure_logging
from fastapi.middleware.cors import CORSMiddleware

from .control_routes import router as control_router
from .healthz import router as healthz_router
from .live_prices import router as live_prices_router
from .middleware import RequestIdMiddleware
from .orch_routes import router as orch_router
from .telemetry_routes import router as telemetry_router
from .zmq_streamer import zmq_publisher
from .kill_switch_listener import KillSwitchListener
from .control_routes import broker_watchdog

configure_logging()
logger = logging.getLogger(__name__)


# ── Kill Switch Callback ────────────────────────────────────────────
# When the native UI halts a sleeve via shared memory, this callback
# freezes the sleeve in the orchestrator control state.

def _on_sleeve_halt(sleeve_id: int, reason: str) -> None:
    """Freeze a sleeve in the control DB when the OOB kill switch fires."""
    from .kill_switch_listener import SLEEVE_NAMES
    import sqlite3

    sleeve_name = SLEEVE_NAMES.get(sleeve_id, f"sleeve_{sleeve_id}")
    logger.critical("OOB KILL SWITCH → freezing sleeve %s (reason: %s)", sleeve_name, reason)

    try:
        _state_db = Path("backend/artifacts/orchestrator_state/state.sqlite3")
        if _state_db.exists():
            conn = sqlite3.connect(_state_db, timeout=5)
            # Read current frozen list, append if not already present
            row = conn.execute(
                "SELECT frozen_sleeves FROM app_control_state WHERE id=1"
            ).fetchone()
            if row:
                import json
                frozen = json.loads(row[0]) if row[0] else []
                if sleeve_name not in frozen:
                    frozen.append(sleeve_name)
                    conn.execute(
                        "UPDATE app_control_state SET frozen_sleeves=? WHERE id=1",
                        (json.dumps(frozen),),
                    )
                    conn.commit()
            conn.close()
    except Exception as e:
        logger.error("Failed to freeze sleeve via kill switch: %s", e)


# ── NTP Sync ────────────────────────────────────────────────────────

async def _assert_ntp_sync() -> None:
    """Verify system clock is within 500ms of UTC via NTP.

    Fail-closed: On drift exceeding tolerance, raises SystemExit
    to prevent the server from starting in an unsafe state.
    Override with --force-ignore-ntp if operators have verified time manually.
    """
    import os
    NTP_MAX_DRIFT_S = 0.5

    if os.getenv("ALGAE_FORCE_IGNORE_NTP", "0") == "1":
        logging.getLogger("ntp").warning(
            "NTP check SKIPPED — ALGAE_FORCE_IGNORE_NTP=1 override active"
        )
        return

    try:
        import ntplib
        client = ntplib.NTPClient()
        response = client.request("time.nist.gov", version=3)
        drift = abs(response.offset)
        if drift > NTP_MAX_DRIFT_S:
            raise SystemExit(
                f"NTP drift detected: {drift:.3f}s (limit: {NTP_MAX_DRIFT_S}s). "
                f"Refusing to start. Run 'w32tm /resync' or 'ntpdate' to fix, "
                f"or set ALGAE_FORCE_IGNORE_NTP=1 to override."
            )
        logging.getLogger("ntp").info("NTP OK — drift=%.3fs (< %.1fs)", drift, NTP_MAX_DRIFT_S)
    except ImportError:
        logging.getLogger("ntp").warning("ntplib not installed — NTP check skipped")
    except SystemExit:
        raise  # Re-raise SystemExit (don't swallow it)
    except OSError as e:
        logging.getLogger("ntp").warning("NTP server unreachable — %s (non-fatal)", e)


# ── ASGI Lifespan ───────────────────────────────────────────────────

_kill_switch = KillSwitchListener(on_halt=_on_sleeve_halt)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """ASGI lifespan — start/stop ZMQ publisher, kill switch, broker watchdog, and NTP check."""
    # ── Startup ─────────────────────────────────────────────────────
    await _assert_ntp_sync()
    zmq_publisher.start()
    _kill_switch.start()
    broker_watchdog.start()

    yield

    # ── Shutdown ────────────────────────────────────────────────────
    broker_watchdog.stop()
    _kill_switch.stop()
    zmq_publisher.shutdown()


app = FastAPI(
    title=APP_DISPLAY,
    description=f"{APP_DISPLAY} telemetry and control API",
    lifespan=lifespan,
)

app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Native frontend uses REST directly, no browser CORS needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(healthz_router)
app.include_router(telemetry_router)
app.include_router(orch_router)
app.include_router(control_router)
app.include_router(live_prices_router)

