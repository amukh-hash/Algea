export type RunSummary = {
  manifest: Record<string, any>;
  status: Record<string, any> | null;
};

const API_BASE = (import.meta.env.VITE_API_BASE as string) || "";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function listRuns(params?: Record<string, string>): Promise<RunSummary[]> {
  const query = params ? `?${new URLSearchParams(params).toString()}` : "";
  return getJson(`/control-room/runs${query}`);
}

export function getRun(runId: string): Promise<RunSummary> {
  return getJson(`/control-room/runs/${runId}`);
}

export function getEvents(runId: string, tail = 500) {
  return getJson(`/control-room/runs/${runId}/events?tail=${tail}`);
}

export function getMetrics(runId: string) {
  return getJson(`/control-room/runs/${runId}/metrics`);
}

export function getArtifacts(runId: string) {
  return getJson(`/control-room/runs/${runId}/artifacts`);
}

export function getCheckpoints(runId: string) {
  return getJson(`/control-room/runs/${runId}/checkpoints`);
}

export function getReport(runId: string, reportType: "preflight" | "gate") {
  return getJson(`/control-room/runs/${runId}/reports/${reportType}`);
}

export function searchArtifacts(search?: string) {
  const query = search ? `?search=${encodeURIComponent(search)}` : "";
  return getJson(`/control-room/artifacts${query}`);
}
