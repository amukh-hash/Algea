# SSE (Server-Sent Events) Contract

## Overview

The backend exposes SSE streams for real-time telemetry updates. The primary stream endpoint is:

```
GET /api/telemetry/stream/runs/{run_id}
```

## Event Types

| Event     | Description                          | Data Shape                              |
|-----------|--------------------------------------|-----------------------------------------|
| `metric`  | New metric data point                | `MetricPoint` JSON                      |
| `event`   | Telemetry event (decision, risk, …)  | `TelemetryEvent` JSON                   |
| `status`  | Run status change                    | `{"status": "running" \| "completed"}` |
| `snapshot` | Reconnect acknowledgment            | `{"run_id": "...", "ack_last_event_id": "N"}` |
| `heartbeat` | Keep-alive (every ~1s when idle)   | `{"ts": "ISO8601"}`                     |

## Event ID Protocol

Every SSE frame includes a monotonically increasing `id:` field:

```
id: 1
event: heartbeat
data: {"ts": "2026-02-18T14:00:00Z"}

id: 2
event: metric
data: {"run_id": "abc", "key": "pnl", "value": 42.0, ...}
```

### Guarantees
- IDs are integers, starting from 1
- IDs are strictly monotonically increasing within a stream
- IDs reset on new connections

## Reconnect Protocol

### `Last-Event-ID` Header

When the browser's `EventSource` reconnects, it sends `Last-Event-ID: N`. The server:

1. Acknowledges with a `snapshot` event containing `ack_last_event_id`
2. Resumes the live stream from the next event

### `since_ts` Query Parameter

For explicit replay of historical events:

```
GET /api/telemetry/stream/runs/{run_id}?since_ts=2026-02-18T10:00:00
```

The server replays matching events from the database, then switches to live streaming.

## Client-Side Expectations

| Feature          | Implementation                    | Location                  |
|------------------|-----------------------------------|---------------------------|
| Dedupe by ID     | Skip events where `id <= lastId`  | `useRunStream.ts:61`      |
| Gap detection    | Toast if `id > lastId + 1`        | `useRunStream.ts:62`      |
| Bounded buffers  | Metrics: 200, Events: 200, Charts: 1000 | `constants.ts`      |
| Batched flush    | 150ms interval                    | `useRunStream.ts:54`      |
| Backoff reconnect | 1s → 2s → 4s → max 30s          | `useEventSource.ts`       |
| Max retries      | 20 attempts before error state    | `useEventSource.ts`       |
