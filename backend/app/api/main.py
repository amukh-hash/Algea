from fastapi import FastAPI

from backend.app.control_room.api import router as control_room_router

app = FastAPI(title="ALGAIE Control Room API")
app.include_router(control_room_router, prefix="/control-room", tags=["control-room"])
