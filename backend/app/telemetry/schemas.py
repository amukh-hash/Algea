from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RunType(str, Enum):
    sleeve_live = "sleeve_live"
    sleeve_paper = "sleeve_paper"
    backtest = "backtest"
    train = "train"


class RunStatus(str, Enum):
    starting = "starting"
    running = "running"
    paused = "paused"
    stopped = "stopped"
    completed = "completed"
    error = "error"


class EventLevel(str, Enum):
    debug = "debug"
    info = "info"
    warn = "warn"
    error = "error"


class EventType(str, Enum):
    DECISION_MADE = "DECISION_MADE"
    GATE_TRIPPED = "GATE_TRIPPED"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    RISK_LIMIT = "RISK_LIMIT"
    ERROR = "ERROR"
    CHECKPOINT_SAVED = "CHECKPOINT_SAVED"
    EVAL_COMPLETE = "EVAL_COMPLETE"
    BACKTEST_COMPLETE = "BACKTEST_COMPLETE"


class ArtifactKind(str, Enum):
    report = "report"
    plot = "plot"
    table = "table"
    checkpoint = "checkpoint"
    config = "config"
    log = "log"
    other = "other"


class Run(BaseModel):
    run_id: str
    run_type: RunType
    name: str
    sleeve_name: str | None = None
    status: RunStatus
    started_at: datetime
    ended_at: datetime | None = None
    git_sha: str = ""
    config_hash: str = ""
    data_version: str = ""
    tags: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class MetricPoint(BaseModel):
    run_id: str
    ts: datetime
    key: str
    value: float
    labels: dict[str, str] = Field(default_factory=dict)


class Event(BaseModel):
    run_id: str
    ts: datetime
    level: EventLevel
    type: EventType
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)


class Artifact(BaseModel):
    run_id: str
    artifact_id: str
    path: str
    kind: ArtifactKind
    mime: str
    bytes: int
    created_at: datetime
    meta: dict[str, Any] = Field(default_factory=dict)


class RunListResponse(BaseModel):
    items: list[Run]
    total: int


class MetricSeriesResponse(BaseModel):
    series: dict[str, list[MetricPoint]]


class EventListResponse(BaseModel):
    items: list[Event]


class ArtifactListResponse(BaseModel):
    items: list[Artifact]
