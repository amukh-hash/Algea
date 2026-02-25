import { fetchJSON } from "./fetchWithTimeout";

export interface OrchHeartbeat { timestamp: string; session: string; state: string; mode: string }
export interface OrchRun { run_id: string; asof_date: string; session: string; started_at: string; ended_at: string | null; status: string; meta?: Record<string, unknown> }

export interface OrchJob {
  run_id: string;
  asof_date: string;
  session: string;
  job_name: string;
  status: string;
  started_at: string | null;
  ended_at: string | null;
  exit_code?: number | null;
  error_summary: string | null;
  last_success_at?: string | null;
}
export interface OrchPosition { symbol: string; qty: number; avg_cost: number }
export interface OrchTarget { symbol: string; target_weight: number; score?: number; side?: string }
export interface CurvePoint { t: string; cum_net_unscaled: number; cum_net_volscaled: number; drawdown: number; rolling_vol: number; rolling_sharpe: number; turnover: number; cost: number }

export interface RiskChecksReport {
  schema_version: "canonical" | "legacy_normalized";
  status: string;
  checked_at?: string;
  asof_date?: string;
  session?: string;
  reason?: string | null;
  missing_sleeves: string[];
  inputs: Record<string, unknown>;
  metrics: Record<string, unknown>;
  limits: Record<string, unknown>;
  violations: Array<{ code: string; message: string; details: Record<string, unknown> }>;
  per_sleeve?: Record<string, unknown>;
  raw?: Record<string, unknown>;
}

export interface JobInfo {
  name: string;
  sessions: string[];
  depends_on: string[];
  min_interval_s: number;
  last_success_at: string | null;
  last_status: string | null;
  last_error: string | null;
  last_duration_s: number | null;
  next_eligible_at: string | null;
}

export interface JobHistoryRow {
  run_id: string;
  asof_date: string;
  session: string;
  name: string;
  started_at: string | null;
  ended_at: string | null;
  min_interval_s: number;
  last_success_at: string | null;
  last_status: string | null;
  last_error: string | null;
  last_duration_s: number | null;
  next_eligible_at: string | null;
}

function validateRisk(payload: any): RiskChecksReport {
  if (!payload || typeof payload !== "object") throw new Error("Invalid risk payload");
  if (!["canonical", "legacy_normalized"].includes(String(payload.schema_version))) throw new Error("Unknown schema_version");
  if (typeof payload.status !== "string") throw new Error("Risk status must be a string");
  if (!Array.isArray(payload.missing_sleeves)) throw new Error("Risk missing_sleeves must be an array");
  if (!Array.isArray(payload.violations)) throw new Error("violations must be array");
  for (const v of payload.violations) {
    if (!v || typeof v !== "object") throw new Error("Violation must be an object");
    if (typeof v.code !== "string" || typeof v.message !== "string") throw new Error("Violation must include string code/message");
  }
  return payload as RiskChecksReport;
}

function validateSeries(series: any): CurvePoint[] {
  if (!Array.isArray(series)) throw new Error("series must be array");
  return series.map((p) => ({
    t: String(p.t),
    cum_net_unscaled: Number(p.cum_net_unscaled ?? 0),
    cum_net_volscaled: Number(p.cum_net_volscaled ?? 0),
    drawdown: Number(p.drawdown ?? 0),
    rolling_vol: Number(p.rolling_vol ?? 0),
    rolling_sharpe: Number(p.rolling_sharpe ?? 0),
    turnover: Number(p.turnover ?? 0),
    cost: Number(p.cost ?? 0),
  }));
}

export const orchApi = {
  getStatus: () => fetchJSON<{ asof_date: string; heartbeat: OrchHeartbeat | null; last_run: OrchRun | null }>("/api/orchestrator/status", { timeout: 8_000 }),
  listRuns: (limit = 20) => fetchJSON<{ items: OrchRun[]; total: number }>(`/api/orchestrator/runs?limit=${limit}`, { timeout: 8_000 }),
  getRunJobs: (runId: string) => fetchJSON<{ items: OrchJob[] }>(`/api/orchestrator/runs/${runId}/jobs`, { timeout: 8_000 }),
  getTargets: (asof?: string) => fetchJSON<{ asof_date: string; sleeves: Record<string, { targets: OrchTarget[] }> }>(`/api/orchestrator/targets${asof ? `?asof=${asof}` : ""}`),
  getPositions: (asof?: string) => fetchJSON<{ asof_date: string; positions: OrchPosition[] }>(`/api/orchestrator/positions${asof ? `?asof=${asof}` : ""}`),
  getFills: (asof?: string) => fetchJSON<{ asof_date: string; fills: unknown[] }>(`/api/orchestrator/fills${asof ? `?asof=${asof}` : ""}`),

  getInstance: (asof: string) => fetchJSON<{ asof: string; asof_date: string; instance: Record<string, unknown>; source: string; sleeves?: Record<string, unknown> }>(`/api/orchestrator/instance?asof=${asof}`),
  getLatestInstance: () => fetchJSON<{ asof: string; asof_date: string; instance: Record<string, unknown>; source: string; sleeves?: Record<string, unknown> }>("/api/orchestrator/instance/latest"),

  getRiskChecks: async (asof: string, session?: string) => {
    const qp = new URLSearchParams({ asof });
    if (session) qp.set("session", session);
    const data = await fetchJSON<{ asof: string; asof_date: string; risk_checks: RiskChecksReport; source: string }>(`/api/orchestrator/risk-checks?${qp.toString()}`);
    return { ...data, risk_checks: validateRisk(data.risk_checks) };
  },
  getLatestRiskChecks: async () => {
    const data = await fetchJSON<{ asof: string; asof_date: string; risk_checks: RiskChecksReport; source: string }>("/api/orchestrator/risk-checks/latest");
    return { ...data, risk_checks: validateRisk(data.risk_checks) };
  },

  getEquitySeries: async (asof: string, sleeve?: string) => {
    const qp = new URLSearchParams({ asof });
    if (sleeve) qp.set("sleeve", sleeve);
    const data = await fetchJSON<{ asof: string; asof_date: string; scope: string; series: CurvePoint[]; source: string }>(`/api/orchestrator/equity-series?${qp.toString()}`);
    return { ...data, series: validateSeries(data.series) };
  },

  getTimeseries: async (asof?: string, sleeve?: string) => {
    if (!asof) return { asof: "", asof_date: "", scope: sleeve ?? "portfolio", series: [] as CurvePoint[], source: "none" };
    return orchApi.getEquitySeries(asof, sleeve);
  },
  getJobRegistry: () => fetchJSON<{ items: JobInfo[]; total: number }>("/api/orchestrator/jobs/registry"),
  getJobs: () => fetchJSON<{ items: JobInfo[]; total: number }>("/api/orchestrator/jobs"),
  getJobHistory: (limit = 100, asof?: string) => fetchJSON<{ items: JobHistoryRow[]; total: number }>(`/api/orchestrator/jobs/history?limit=${limit}${asof ? `&asof=${asof}` : ""}`),

  listArtifacts: (asof?: string) => fetchJSON<{ items: Array<{ asof: string; name: string; relative_path: string; size_bytes: number; modified_at: string; download_url: string }>; total: number }>(`/api/orchestrator/artifacts${asof ? `?asof=${asof}` : ""}`),
  listDates: () => fetchJSON<{ items: string[] }>("/api/orchestrator/dates"),
};
