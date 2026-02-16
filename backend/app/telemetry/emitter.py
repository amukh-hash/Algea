from __future__ import annotations

import hashlib
import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import Artifact, Event, EventLevel, EventType, MetricPoint, Run, RunStatus, RunType
from .storage import TelemetryStorage, now_utc

logger = logging.getLogger(__name__)


class TelemetryEmitter:
    def __init__(self, storage: TelemetryStorage):
        self.storage = storage

    def start_run(
        self,
        run_type: RunType,
        name: str,
        sleeve_name: str | None = None,
        meta: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        git_sha: str = "dev",
        config_hash: str = "dev",
        data_version: str = "dev",
    ) -> str:
        run_id = str(uuid.uuid4())
        run = Run(
            run_id=run_id,
            run_type=run_type,
            name=name,
            sleeve_name=sleeve_name,
            status=RunStatus.starting,
            started_at=now_utc(),
            git_sha=git_sha,
            config_hash=config_hash,
            data_version=data_version,
            tags=tags or [],
            meta=meta or {},
        )
        self._safe_call(self.storage.upsert_run, run)
        self.storage.publish(run_id, {"type": "status", "data": {"run_id": run_id, "status": "starting", "ts": now_utc().isoformat()}})
        return run_id

    def set_status(self, run_id: str, status: RunStatus) -> None:
        run = self.storage.get_run(run_id)
        if not run:
            return
        run.status = status
        if status in {RunStatus.completed, RunStatus.error, RunStatus.stopped}:
            run.ended_at = now_utc()
        self._safe_call(self.storage.upsert_run, run)
        self.storage.publish(run_id, {"type": "status", "data": {"run_id": run_id, "status": status.value, "ts": now_utc().isoformat()}})

    def emit_metric(
        self,
        run_id: str,
        key: str,
        value: float,
        ts: datetime | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        point = MetricPoint(run_id=run_id, ts=ts or now_utc(), key=key, value=float(value), labels=labels or {})
        self._safe_call(self.storage.insert_metric, point)
        self.storage.publish(run_id, {"type": "metric", "data": point.model_dump(mode="json")})

    def emit_event(
        self,
        run_id: str,
        level: EventLevel,
        type: EventType,
        message: str,
        ts: datetime | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        event = Event(
            run_id=run_id,
            ts=ts or now_utc(),
            level=level,
            type=type,
            message=message,
            payload=payload or {},
        )
        self._safe_call(self.storage.insert_event, event)
        self.storage.publish(run_id, {"type": "event", "data": event.model_dump(mode="json")})

    def register_artifact(
        self,
        run_id: str,
        path: str,
        kind: str,
        mime: str,
        bytes: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        src_path = Path(path)
        artifact_id = hashlib.sha1(f"{run_id}:{path}:{datetime.now(tz=timezone.utc).isoformat()}".encode()).hexdigest()[:16]
        dst_dir = self.storage.artifacts_root / run_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / src_path.name
        if src_path.exists() and src_path.resolve() != dst_path.resolve():
            shutil.copy2(src_path, dst_path)
        artifact = Artifact(
            run_id=run_id,
            artifact_id=artifact_id,
            path=str(dst_path),
            kind=kind,
            mime=mime,
            bytes=bytes if bytes is not None else (dst_path.stat().st_size if dst_path.exists() else 0),
            created_at=now_utc(),
            meta=meta or {},
        )
        self._safe_call(self.storage.register_artifact, artifact)
        return artifact_id

    def _safe_call(self, fn, *args):
        try:
            fn(*args)
        except Exception:  # best effort by design
            logger.exception("telemetry emit failed", extra={"args": json.dumps(str(args))})
