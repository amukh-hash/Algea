import { Artifact, MetricPoint, Run, TelemetryEvent } from "./types";
import { fetchJsonWithTimeout } from "./http";
import { getCachedRuntimeConfig, resolveRuntimeConfig } from "@/runtime/config";

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

async function getApiBase(): Promise<string> {
  const cached = getCachedRuntimeConfig();
  if (cached) return cached.apiBaseUrl;
  const cfg = await resolveRuntimeConfig();
  return cfg.apiBaseUrl;
}

export function getApiBaseSync(): string {
  return getCachedRuntimeConfig()?.apiBaseUrl ?? DEFAULT_API_BASE;
}

async function fetchJSON<T>(path: string): Promise<T> {
  const base = await getApiBase();
  return fetchJsonWithTimeout<T>(`${base}${path}`, { cache: "default" }, 15_000);
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
    `${getApiBaseSync()}/api/telemetry/runs/${runId}/artifacts/${artifactId}`,
  streamUrl: (runId: string) => `${getApiBaseSync()}/api/telemetry/stream/runs/${runId}`,
};
