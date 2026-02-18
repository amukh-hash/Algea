import { LWPoint } from "./api";
import { fetchJSON } from "./fetchWithTimeout";

export interface OrchHeartbeat {
    timestamp: string;
    session: string;
    state: string;
    mode: string;
}

export interface OrchRun {
    run_id: string;
    asof_date: string;
    session: string;
    started_at: string;
    ended_at: string | null;
    status: string;
    meta: Record<string, unknown>;
}

export interface OrchJob {
    run_id: string;
    asof_date: string;
    session: string;
    job_name: string;
    status: string;
    started_at: string | null;
    ended_at: string | null;
    exit_code: number | null;
    error_summary: string | null;
}

export interface OrchPosition {
    symbol: string;
    qty: number;
    avg_cost: number;
}

export interface OrchTarget {
    symbol: string;
    target_weight: number;
    score?: number;
    side?: string;
}

export interface OrchStatus {
    asof_date: string;
    heartbeat: OrchHeartbeat | null;
    last_run: OrchRun | null;
}

export const orchApi = {
    getStatus: () => fetchJSON<OrchStatus>("/api/orchestrator/status", { timeout: 8_000 }),
    listRuns: (limit = 20) =>
        fetchJSON<{ items: OrchRun[]; total: number }>(
            `/api/orchestrator/runs?limit=${limit}`,
            { timeout: 8_000 },
        ),
    getRunJobs: (runId: string) =>
        fetchJSON<{ items: OrchJob[] }>(
            `/api/orchestrator/runs/${runId}/jobs`,
            { timeout: 8_000 },
        ),
    getPositions: (asof?: string) =>
        fetchJSON<{ positions: OrchPosition[]; asof_date: string }>(
            `/api/orchestrator/positions${asof ? `?asof=${asof}` : ""}`,
            { timeout: 8_000 },
        ),
    getTargets: (asof?: string) =>
        fetchJSON<{
            asof_date: string;
            sleeves: Record<string, { targets: OrchTarget[]; status?: string }>;
        }>(`/api/orchestrator/targets${asof ? `?asof=${asof}` : ""}`, { timeout: 8_000 }),
    getFills: (asof?: string) =>
        fetchJSON<{ fills: unknown[]; asof_date: string }>(
            `/api/orchestrator/fills${asof ? `?asof=${asof}` : ""}`,
            { timeout: 8_000 },
        ),
};
