from __future__ import annotations

import json
import sqlite3
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from queue import Full
from typing import Any, Iterable

from .schemas import Artifact, Event, MetricPoint, Run


class TelemetryStorage:
    def __init__(self, db_url: str | None = None, artifacts_root: str = "backend/artifacts") -> None:
        self.db_path = Path((db_url or "sqlite:///backend/telemetry.db").replace("sqlite:///", ""))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_root = Path(artifacts_root)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._listeners: dict[str, list[Queue[dict[str, Any]]]] = defaultdict(list)
        self._stream_seq: dict[str, int] = defaultdict(int)
        self._stream_ring: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=1000))
        self._dropped_events: dict[str, int] = defaultdict(int)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=3)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    run_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    sleeve_name TEXT,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    git_sha TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    data_version TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    meta TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    ts TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value REAL NOT NULL,
                    labels TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_metrics_run_key_ts ON metrics(run_id, key, ts);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_metrics_unique ON metrics(run_id, key, ts);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    ts TEXT NOT NULL,
                    level TEXT NOT NULL,
                    type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_run_ts ON events(run_id, ts);
                CREATE INDEX IF NOT EXISTS idx_events_run_type_ts ON events(run_id, type, ts);
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    path TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    mime TEXT NOT NULL,
                    bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    meta TEXT NOT NULL
                );
                """
            )

    def publish(self, run_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._stream_seq[run_id] += 1
            envelope = {"id": self._stream_seq[run_id], **payload}
            self._stream_ring[run_id].append(envelope)
            for queue in self._listeners[run_id]:
                try:
                    queue.put_nowait(envelope)
                except Full:
                    self._dropped_events[run_id] += 1

    def stream_snapshot(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        with self._lock:
            ring = list(self._stream_ring[run_id])
            dropped = int(self._dropped_events.get(run_id, 0))
        return {
            "status": run.status.value if run else "unknown",
            "last_event_id": ring[-1]["id"] if ring else 0,
            "recent": ring[-100:],
            "dropped_events": dropped,
        }

    def replay_since(self, run_id: str, last_event_id: int) -> list[dict[str, Any]]:
        with self._lock:
            return [item for item in list(self._stream_ring[run_id]) if int(item.get("id", 0)) > last_event_id]

    def subscribe(self, run_id: str) -> Queue[dict[str, Any]]:
        queue: Queue[dict[str, Any]] = Queue(maxsize=1000)
        with self._lock:
            self._listeners[run_id].append(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: Queue[dict[str, Any]]) -> None:
        with self._lock:
            if run_id in self._listeners and queue in self._listeners[run_id]:
                self._listeners[run_id].remove(queue)

    def upsert_run(self, run: Run) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO runs(run_id, run_type, name, sleeve_name, status, started_at, ended_at, git_sha, config_hash, data_version, tags, meta)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id) DO UPDATE SET
                    run_type=excluded.run_type,
                    name=excluded.name,
                    sleeve_name=excluded.sleeve_name,
                    status=excluded.status,
                    started_at=excluded.started_at,
                    ended_at=excluded.ended_at,
                    git_sha=excluded.git_sha,
                    config_hash=excluded.config_hash,
                    data_version=excluded.data_version,
                    tags=excluded.tags,
                    meta=excluded.meta
                """,
                (
                    run.run_id,
                    run.run_type.value,
                    run.name,
                    run.sleeve_name,
                    run.status.value,
                    run.started_at.isoformat(),
                    run.ended_at.isoformat() if run.ended_at else None,
                    run.git_sha,
                    run.config_hash,
                    run.data_version,
                    json.dumps(run.tags),
                    json.dumps(run.meta),
                ),
            )

    def list_runs(self, filters: dict[str, str | None], limit: int, offset: int) -> tuple[list[Run], int]:
        clauses, params = [], []
        for key in ("run_type", "status", "sleeve_name"):
            if filters.get(key):
                clauses.append(f"{key} = ?")
                params.append(filters[key])
        if filters.get("q"):
            clauses.append("(name LIKE ? OR run_id LIKE ?)")
            like = f"%{filters['q']}%"
            params.extend([like, like])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM runs {where} ORDER BY started_at DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
            total = conn.execute(f"SELECT COUNT(*) FROM runs {where}", params).fetchone()[0]
        return [self._row_to_run(r) for r in rows], total

    def get_run(self, run_id: str) -> Run | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._row_to_run(row) if row else None

    def insert_metric(self, point: MetricPoint) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO metrics(run_id, ts, key, value, labels) VALUES(?,?,?,?,?)",
                (point.run_id, point.ts.isoformat(), point.key, point.value, json.dumps(point.labels)),
            )

    def insert_metrics_batch(self, points: list[MetricPoint]) -> int:
        """Batch insert metrics, ignoring duplicates on (run_id, key, ts). Returns count inserted."""
        if not points:
            return 0
        with self._conn() as conn:
            cursor = conn.executemany(
                "INSERT OR IGNORE INTO metrics(run_id, ts, key, value, labels) VALUES(?,?,?,?,?)",
                [
                    (p.run_id, p.ts.isoformat(), p.key, p.value, json.dumps(p.labels))
                    for p in points
                ],
            )
            return cursor.rowcount

    def delete_run_cascade(self, run_id: str) -> None:
        """Delete a run and all its metrics, events, and artifacts."""
        with self._conn() as conn:
            conn.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM events WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM artifacts WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

    def query_metrics(
        self,
        run_id: str,
        keys: Iterable[str],
        start: datetime | None,
        end: datetime | None,
        every_ms: int | None,
    ) -> dict[str, list[MetricPoint]]:
        series: dict[str, list[MetricPoint]] = {key: [] for key in keys}
        for key in keys:
            clauses = ["run_id = ?", "key = ?"]
            params: list[Any] = [run_id, key]
            if start:
                clauses.append("ts >= ?")
                params.append(start.isoformat())
            if end:
                clauses.append("ts <= ?")
                params.append(end.isoformat())
            sql = f"SELECT * FROM metrics WHERE {' AND '.join(clauses)} ORDER BY ts ASC"
            with self._conn() as conn:
                rows = conn.execute(sql, params).fetchall()
            points = [
                MetricPoint(
                    run_id=row["run_id"],
                    ts=datetime.fromisoformat(row["ts"]),
                    key=row["key"],
                    value=row["value"],
                    labels=json.loads(row["labels"]),
                )
                for row in rows
            ]
            if every_ms:
                points = self._downsample(points, every_ms)
            series[key] = points
        return series

    def _downsample(self, points: list[MetricPoint], every_ms: int) -> list[MetricPoint]:
        buckets: dict[int, MetricPoint] = {}
        for point in points:
            bucket = int(point.ts.timestamp() * 1000) // every_ms
            buckets[bucket] = point
        return [buckets[k] for k in sorted(buckets)]

    def insert_event(self, event: Event) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO events(run_id, ts, level, type, message, payload) VALUES(?,?,?,?,?,?)",
                (
                    event.run_id,
                    event.ts.isoformat(),
                    event.level.value,
                    event.type.value,
                    event.message,
                    json.dumps(event.payload),
                ),
            )

    def query_events(
        self,
        run_id: str,
        start: datetime | None,
        end: datetime | None,
        level: str | None,
        event_type: str | None,
        limit: int,
    ) -> list[Event]:
        clauses = ["run_id = ?"]
        params: list[Any] = [run_id]
        if start:
            clauses.append("ts >= ?")
            params.append(start.isoformat())
        if end:
            clauses.append("ts <= ?")
            params.append(end.isoformat())
        if level:
            clauses.append("level = ?")
            params.append(level)
        if event_type:
            clauses.append("type = ?")
            params.append(event_type)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM events WHERE {' AND '.join(clauses)} ORDER BY ts DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [
            Event(
                run_id=row["run_id"],
                ts=datetime.fromisoformat(row["ts"]),
                level=row["level"],
                type=row["type"],
                message=row["message"],
                payload=json.loads(row["payload"]),
            )
            for row in rows
        ]

    def register_artifact(self, artifact: Artifact) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO artifacts(artifact_id, run_id, path, kind, mime, bytes, created_at, meta) VALUES(?,?,?,?,?,?,?,?)",
                (
                    artifact.artifact_id,
                    artifact.run_id,
                    artifact.path,
                    artifact.kind.value,
                    artifact.mime,
                    artifact.bytes,
                    artifact.created_at.isoformat(),
                    json.dumps(artifact.meta),
                ),
            )

    def list_artifacts(self, run_id: str) -> list[Artifact]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM artifacts WHERE run_id = ? ORDER BY created_at DESC", (run_id,)
            ).fetchall()
        return [
            Artifact(
                run_id=row["run_id"],
                artifact_id=row["artifact_id"],
                path=row["path"],
                kind=row["kind"],
                mime=row["mime"],
                bytes=row["bytes"],
                created_at=datetime.fromisoformat(row["created_at"]),
                meta=json.loads(row["meta"]),
            )
            for row in rows
        ]

    def get_artifact(self, run_id: str, artifact_id: str) -> Artifact | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE run_id = ? AND artifact_id = ?", (run_id, artifact_id)
            ).fetchone()
        if not row:
            return None
        return Artifact(
            run_id=row["run_id"],
            artifact_id=row["artifact_id"],
            path=row["path"],
            kind=row["kind"],
            mime=row["mime"],
            bytes=row["bytes"],
            created_at=datetime.fromisoformat(row["created_at"]),
            meta=json.loads(row["meta"]),
        )

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> Run:
        return Run(
            run_id=row["run_id"],
            run_type=row["run_type"],
            name=row["name"],
            sleeve_name=row["sleeve_name"],
            status=row["status"],
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
            git_sha=row["git_sha"],
            config_hash=row["config_hash"],
            data_version=row["data_version"],
            tags=json.loads(row["tags"]),
            meta=json.loads(row["meta"]),
        )


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def queue_get(queue: Queue[dict[str, Any]], timeout: float = 1.0) -> dict[str, Any] | None:
    try:
        return queue.get(timeout=timeout)
    except Empty:
        return None
