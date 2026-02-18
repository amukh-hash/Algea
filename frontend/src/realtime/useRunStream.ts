"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { api, LWPoint } from "@/lib/api";
import { MetricPoint, TelemetryEvent } from "@/lib/types";
import { CHART_HISTORY_LIMIT, EVENT_HISTORY_LIMIT, MAX_SIMULTANEOUS_STREAMS, METRIC_HISTORY_LIMIT } from "./constants";
import { useEventSource } from "./useEventSource";
import { useToast } from "@/components/ui/toast";

let activeStreams = 0;
const summary = { state: "closed", lastUpdate: null as number | null };
const listeners = new Set<() => void>();

function updateSummary(next: Partial<typeof summary>) {
  Object.assign(summary, next);
  listeners.forEach((l) => l());
}

export function useConnectionSummary() {
  const [, rerender] = useState(0);
  useEffect(() => {
    const listener = () => rerender((v) => v + 1);
    listeners.add(listener);
    return () => { listeners.delete(listener); };
  }, []);
  return summary;
}

export function useRunStream(runId: string | null, paused = false) {
  const [metrics, setMetrics] = useState<Record<string, MetricPoint[]>>({});
  const [metricsLW, setMetricsLW] = useState<Record<string, LWPoint[]>>({});
  const [events, setEvents] = useState<TelemetryEvent[]>([]);
  const [status, setStatus] = useState<string>("unknown");
  const [lastEventId, setLastEventId] = useState<number>(0);
  const queueRef = useRef<{ type: "metric" | "event"; payload: MetricPoint | TelemetryEvent; id: number }[]>([]);
  const pausedQueue = useRef<typeof queueRef.current>([]);
  const { push } = useToast();

  useEffect(() => {
    const interval = setInterval(() => {
      const queue = paused ? pausedQueue.current : queueRef.current;
      if (!queue.length) return;
      const batch = queue.splice(0, queue.length);
      batch.forEach((item) => {
        if (item.type === "metric") {
          const point = item.payload as MetricPoint;
          setMetrics((prev) => ({ ...prev, [point.key]: [...(prev[point.key] ?? []), point].slice(-METRIC_HISTORY_LIMIT) }));
          const lwPoint: LWPoint = { time: Math.floor(new Date(point.ts).getTime() / 1000), value: point.value };
          setMetricsLW((prev) => ({ ...prev, [point.key]: [...(prev[point.key] ?? []), lwPoint].slice(-CHART_HISTORY_LIMIT) }));
        } else {
          setEvents((prev) => [item.payload as TelemetryEvent, ...prev].slice(0, EVENT_HISTORY_LIMIT));
        }
      });
    }, 150);
    return () => clearInterval(interval);
  }, [paused]);

  const stream = useEventSource(runId ? api.streamUrl(runId) : null, {
    metric: (evt) => {
      const id = Number(evt.lastEventId || 0);
      if (id && id <= lastEventId) return;
      if (id > lastEventId + 1 && lastEventId > 0) push({ message: `Gap detected on ${runId}` });
      if (id) setLastEventId(id);
      queueRef.current.push({ type: "metric", payload: JSON.parse(evt.data), id });
    },
    event: (evt) => {
      const id = Number(evt.lastEventId || 0);
      if (id && id <= lastEventId) return;
      if (id) setLastEventId(id);
      queueRef.current.push({ type: "event", payload: JSON.parse(evt.data), id });
    },
    status: (evt) => setStatus(JSON.parse(evt.data).status),
    heartbeat: () => undefined,
  });

  useEffect(() => {
    activeStreams += 1;
    if (activeStreams > MAX_SIMULTANEOUS_STREAMS) push({ message: `High stream count (${activeStreams})` });
    return () => { activeStreams -= 1; };
  }, [push]);

  useEffect(() => {
    updateSummary({ state: stream.state, lastUpdate: stream.lastMessageAt });
  }, [stream.state, stream.lastMessageAt]);

  return useMemo(() => ({ metrics, metricsLW, events, status, streamState: stream.state }), [metrics, metricsLW, events, status, stream.state]);
}
