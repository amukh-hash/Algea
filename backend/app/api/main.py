from fastapi import FastAPI

from .telemetry_routes import router as telemetry_router

app = FastAPI(title="ALGAIE Telemetry API")
app.include_router(telemetry_router)
