"use client";

import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import { MetricPoint, TelemetryEvent } from "./types";

export function useRunStream(runId: string | null) {
  const [metrics, setMetrics] = useState<Record<string, MetricPoint[]>>({});
  const [events, setEvents] = useState<TelemetryEvent[]>([]);
  const [status, setStatus] = useState<string>("unknown");

  useEffect(() => {
    if (!runId) return;
    const source = new EventSource(api.streamUrl(runId));
    source.addEventListener("metric", (evt) => {
      const point = JSON.parse((evt as MessageEvent).data) as MetricPoint;
      setMetrics((prev) => ({ ...prev, [point.key]: [...(prev[point.key] ?? []).slice(-199), point] }));
    });
    source.addEventListener("event", (evt) => {
      const event = JSON.parse((evt as MessageEvent).data) as TelemetryEvent;
      setEvents((prev) => [event, ...prev].slice(0, 200));
    });
    source.addEventListener("status", (evt) => {
      setStatus(JSON.parse((evt as MessageEvent).data).status);
    });
    return () => source.close();
  }, [runId]);

  return useMemo(() => ({ metrics, events, status }), [metrics, events, status]);
}
