import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .healthz import router as healthz_router
from .middleware import RequestIdMiddleware
from .orch_routes import router as orch_router
from .telemetry_routes import router as telemetry_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="ALGAIE Telemetry API")

app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(healthz_router)
app.include_router(telemetry_router)
app.include_router(orch_router)
