from fastapi import FastAPI

from .telemetry_routes import router as telemetry_router

app = FastAPI(title="ALGAIE Telemetry API")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(telemetry_router)
