# SSE Contract

## Event framing
Each SSE message is framed as:

```
id: <monotonic integer>
event: <metric|event|status|heartbeat|snapshot>
data: <json>
```

## Reconnect semantics
- Client sends `Last-Event-ID` on reconnect (browser-managed).
- Server attempts replay from ring buffer for ids greater than `Last-Event-ID`.
- If replay unavailable, server emits a `snapshot` event with `ack_last_event_id` before live flow.

## Client protections
- Dedupe: ignore `id <= lastSeenId`.
- Gap detection: if id jumps (`> lastSeenId + 1`), surface warning.
- Bounded windows:
  - metric history/key: 200
  - chart points/key: 1000
  - events: 200
- Flush cadence is throttled to avoid rerender storms.
