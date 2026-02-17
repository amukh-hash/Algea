from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from backend.app.telemetry.schemas import (
    ArtifactListResponse,
    EventListResponse,
    MetricSeriesResponse,
    Run,
    RunListResponse,
)
from backend.app.telemetry.storage import TelemetryStorage, now_utc, queue_get

router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])
storage = TelemetryStorage()


@router.get("/runs", response_model=RunListResponse)
def list_runs(
    run_type: str | None = None,
    status: str | None = None,
    sleeve_name: str | None = None,
    q: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> RunListResponse:
    items, total = storage.list_runs(
        {"run_type": run_type, "status": status, "sleeve_name": sleeve_name, "q": q},
        limit=limit,
        offset=offset,
    )
    return RunListResponse(items=items, total=total)


@router.get("/runs/{run_id}", response_model=Run)
def get_run(run_id: str) -> Run:
    run = storage.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/runs/{run_id}/metrics")
def get_metrics(
    run_id: str,
    keys: str,
    start: datetime | None = None,
    end: datetime | None = None,
    every_ms: int | None = Query(None, ge=100),
    format: str | None = Query(None, description="Set to 'lw' for lightweight-charts format"),
) -> MetricSeriesResponse | dict:
    key_list = [key.strip() for key in keys.split(",") if key.strip()]
    series = storage.query_metrics(run_id, key_list, start, end, every_ms)
    if format == "lw":
        lw_series: dict[str, list[dict[str, float]]] = {}
        for key, points in series.items():
            lw_series[key] = [
                {"time": int(p.ts.timestamp()), "value": p.value}
                for p in points
            ]
        return {"series": lw_series}
    return MetricSeriesResponse(series=series)


@router.get("/runs/{run_id}/events", response_model=EventListResponse)
def get_events(
    run_id: str,
    start: datetime | None = None,
    end: datetime | None = None,
    level: str | None = None,
    type: str | None = None,
    limit: int = Query(500, ge=1, le=5000),
) -> EventListResponse:
    return EventListResponse(items=storage.query_events(run_id, start, end, level, type, limit))


@router.get("/runs/{run_id}/artifacts", response_model=ArtifactListResponse)
def list_artifacts(run_id: str) -> ArtifactListResponse:
    return ArtifactListResponse(items=storage.list_artifacts(run_id))


@router.get("/runs/{run_id}/artifacts/{artifact_id}")
def get_artifact(run_id: str, artifact_id: str) -> FileResponse:
    artifact = storage.get_artifact(run_id, artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    path = Path(artifact.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing")
    return FileResponse(path, media_type=artifact.mime, filename=path.name)


@router.get("/stream/runs/{run_id}")
def stream_run(run_id: str, since_ts: datetime | None = None) -> StreamingResponse:
    queue = storage.subscribe(run_id)

    async def event_gen():
        try:
            if since_ts:
                for event in storage.query_events(run_id, since_ts, None, None, None, limit=250):
                    yield _sse("event", event.model_dump_json())
            while True:
                item = queue_get(queue, timeout=1)
                if item:
                    yield _sse(item["type"], json.dumps(item["data"]))
                else:
                    yield _sse("heartbeat", json.dumps({"ts": now_utc().isoformat()}))
        finally:
            storage.unsubscribe(run_id, queue)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"
