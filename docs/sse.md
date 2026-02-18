# SSE Contract

Endpoint: `GET /api/telemetry/stream/runs/{run_id}`

## Event framing
Each message includes:
- `id: <monotonic integer>`
- `event: metric|event|status|heartbeat|snapshot`
- `data: <json payload>`

## Reconnect
- Client sends `Last-Event-ID` automatically.
- Server reads `last-event-id` header.
- Server emits `snapshot` acknowledging last seen id when reconnecting.
- Heartbeats continue with IDs to preserve monotonic sequence.

## Client behavior
- Drop duplicates (`id <= lastSeenId`).
- Raise gap warning toast if id jumps unexpectedly.
- Bounded buffers: metrics=200, chart points=1000, events=200.
