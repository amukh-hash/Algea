"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "@/lib/api";
import { MetricPoint, TelemetryEvent } from "@/lib/types";
import { useEventSource } from "./useEventSource";

export const METRIC_WINDOW = 200;
export const CHART_WINDOW = 1000;
export const EVENT_WINDOW = 200;

export function useRunStream(runId: string | null) {
  const [metrics, setMetrics] = useState<Record<string, MetricPoint[]>>({});
  const [metricsLW, setMetricsLW] = useState<Record<string, { time: number; value: number }[]>>({});
  const [events, setEvents] = useState<TelemetryEvent[]>([]);
  const [status, setStatus] = useState<string>("unknown");
  const [gapDetected, setGapDetected] = useState(false);
  const queue = useRef<Array<{ kind: string; data: unknown; id: number }>>([]);
  const lastSeen = useRef<number>(0);

  const onEvent = useCallback((evt: MessageEvent) => {
    const eventId = Number(evt.lastEventId || Date.now());
    if (eventId <= lastSeen.current) return;
    if (lastSeen.current > 0 && eventId > lastSeen.current + 1) setGapDetected(true);
    lastSeen.current = eventId;

    let kind = evt.type || "message";
    if (kind === "message") {
      try {
        const parsed = JSON.parse(evt.data);
        kind = parsed.type ?? "message";
      } catch {}
    }
    queue.current.push({ kind, data: evt.data, id: eventId });
  }, []);

  const es = useEventSource(runId ? api.streamUrl(runId) : null, onEvent);

  useEffect(() => {
    const timer = setInterval(() => {
      if (queue.current.length === 0) return;
      const batch = queue.current.splice(0, queue.current.length);
      for (const item of batch) {
        if (item.kind === "metric") {
          const point = JSON.parse(item.data as string) as MetricPoint;
          setMetrics((prev) => ({ ...prev, [point.key]: [...(prev[point.key] ?? []).slice(-(METRIC_WINDOW - 1)), point] }));
          const lwPoint = { time: Math.floor(new Date(point.ts).getTime() / 1000), value: point.value };
          setMetricsLW((prev) => ({ ...prev, [point.key]: [...(prev[point.key] ?? []).slice(-(CHART_WINDOW - 1)), lwPoint] }));
        } else if (item.kind === "event") {
          const event = JSON.parse(item.data as string) as TelemetryEvent;
          setEvents((prev) => [event, ...prev].slice(0, EVENT_WINDOW));
        } else if (item.kind === "status") {
          setStatus(JSON.parse(item.data as string).status);
        }
      }
    }, 150);
    return () => clearInterval(timer);
  }, []);

  return useMemo(() => ({ metrics, metricsLW, events, status, connectionState: es.state, lastMessageAt: es.lastMessageAt, gapDetected }), [metrics, metricsLW, events, status, es.state, es.lastMessageAt, gapDetected]);
}
