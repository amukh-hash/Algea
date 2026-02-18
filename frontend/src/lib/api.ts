import { Artifact, MetricPoint, Run, TelemetryEvent } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

async function fetchJSON<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<T>;
}

export type LWPoint = { time: number; value: number };
export type LWSeriesResponse = { series: Record<string, LWPoint[]> };

export const api = {
  listRuns: (q = "", limit = 50) =>
    fetchJSON<{ items: Run[]; total: number }>(
      `/api/telemetry/runs?limit=${limit}${q ? `&q=${encodeURIComponent(q)}` : ""}`
    ),
  getRun: (runId: string) => fetchJSON<Run>(`/api/telemetry/runs/${runId}`),
  getMetrics: (runId: string, keys: string[]) =>
    fetchJSON<{ series: Record<string, MetricPoint[]> }>(
      `/api/telemetry/runs/${runId}/metrics?keys=${keys.join(",")}`
    ),
  getMetricsLW: (
    runId: string,
    keys: string[],
    opts?: { start?: string; end?: string; every_ms?: number }
  ) => {
    const params = new URLSearchParams({ keys: keys.join(","), format: "lw" });
    if (opts?.start) params.set("start", opts.start);
    if (opts?.end) params.set("end", opts.end);
    if (opts?.every_ms) params.set("every_ms", String(opts.every_ms));
    return fetchJSON<LWSeriesResponse>(
      `/api/telemetry/runs/${runId}/metrics?${params.toString()}`
    );
  },
  getEvents: (runId: string) =>
    fetchJSON<{ items: TelemetryEvent[] }>(`/api/telemetry/runs/${runId}/events`),
  listArtifacts: (runId: string) =>
    fetchJSON<{ items: Artifact[] }>(`/api/telemetry/runs/${runId}/artifacts`),
  artifactUrl: (runId: string, artifactId: string) =>
    `${API_BASE}/api/telemetry/runs/${runId}/artifacts/${artifactId}`,
  streamUrl: (runId: string) => `${API_BASE}/api/telemetry/stream/runs/${runId}`,
  executionStreamUrl: () => `${API_BASE}/api/telemetry/stream/execution`,
};
