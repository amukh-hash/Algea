from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .telemetry_routes import router as telemetry_router

APP_VERSION = "1.0"

app = FastAPI(title="ALGAIE Telemetry API")


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz() -> dict:
    return {
        "ok": True,
        "app": "algaie",
        "ts": datetime.now(timezone.utc).isoformat(),
        "orchestrator": {"ok": True, "last_heartbeat": None},
        "version": APP_VERSION,
    }


async def _get_orchestrator_status() -> dict:
    # placeholder lightweight status; real checks must remain bounded
    await asyncio.sleep(0)
    return {"ok": True, "state": "idle"}


@app.get("/api/orchestrator/status")
async def orchestrator_status(request: Request):
    try:
        status = await asyncio.wait_for(_get_orchestrator_status(), timeout=3.0)
        return {"ok": True, "orchestrator": status}
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={
                "error": "timeout",
                "detail": "orchestrator status probe exceeded 3s timeout",
                "request_id": getattr(request.state, "request_id", "unknown"),
                "hint": "orchestrator may be unhealthy",
            },
        )


app.include_router(telemetry_router)
