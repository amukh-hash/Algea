/**
 * Control API client — typed wrappers for all /api/control/* endpoints.
 */
import { fetchWithTimeout } from "./fetchWithTimeout";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Types ───────────────────────────────────────────────────────────

export interface ControlState {
    paused: boolean;
    vol_regime_override: string | null;
    blocked_symbols: string[];
    frozen_sleeves: string[];
    gross_exposure_cap: number | null;
    execution_mode: string;
}

export interface BrokerStatus {
    connected: boolean;
    gateway_url?: string;
    paper_only?: boolean;
    account_id?: string;
    mode: string;
    error?: string;
}

export interface JobNode {
    name: string;
    deps: string[];
    sessions: string[];
    timeout_s: number;
    min_interval_s: number;
    last_status: string | null;
    last_started: string | null;
    last_duration_s: number | null;
    last_error: string | null;
}

export interface CalendarInfo {
    date: string;
    current_session: string;
    is_trading_day: boolean;
    current_time: string;
    session_windows: Record<string, { start: string; end: string }>;
}

export interface OrchestratorConfig {
    timezone: string;
    exchange: string;
    mode: string;
    poll_interval_s: number;
    paper_only: boolean;
    account_equity: number;
    max_order_notional: number;
    max_total_order_notional: number;
    max_orders: number;
    enabled_jobs: string[];
    disabled_jobs: string[];
}

export interface AuditEntry {
    ts: string;
    action: string;
    detail: Record<string, unknown>;
}

export interface TickResult {
    ok: boolean;
    run_id?: string;
    ran_jobs?: string[];
    failed_jobs?: string[];
    skipped_jobs?: string[];
}

export interface ManualOrder {
    symbol: string;
    qty: number;
    side: string;
    order_type: string;
    limit_price?: number | null;
}

export interface PortfolioHolding {
    symbol: string;
    root: string;
    instrument_type: "future" | "equity";
    sleeve: "core" | "vrp" | "selector";
    qty: number;
    avg_cost: number;
    last_price: number;
    multiplier: number;
    notional: number;
    margin_posted: number;
    unrealized_pnl: number;
}

export interface SleeveSummary {
    margin: number;
    notional: number;
    unrealized_pnl: number;
    equity_cost: number;
    position_count: number;
}

export interface PortfolioSummary {
    asof_date: string;
    account_equity: number;
    cash: number;
    total_margin: number;
    total_notional: number;
    total_equity_cost: number;
    total_unrealized_pnl: number;
    total_value: number;
    total_pnl: number;
    realized_pnl: number;
    position_count: number;
    fill_count: number;
    holdings: PortfolioHolding[];
    sleeves: Record<string, SleeveSummary>;
}

export interface PortfolioHistoryPoint {
    ts: string;
    total_value: number;
    cash: number;
    core: number;
    vrp: number;
    selector: number;
    total_pnl: number;
    unrealized_pnl: number;
}

export interface PortfolioHistoryResponse {
    asof_date: string;
    points: PortfolioHistoryPoint[];
    count: number;
}

// ── Helpers ─────────────────────────────────────────────────────────

async function get<T>(path: string): Promise<T> {
    const res = await fetchWithTimeout(`${BASE}${path}`);
    return res.json();
}

async function put<T>(path: string, body: unknown): Promise<T> {
    const res = await fetchWithTimeout(`${BASE}${path}`, {
        init: {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        },
    });
    return res.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
    const res = await fetchWithTimeout(`${BASE}${path}`, {
        init: {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        },
    });
    return res.json();
}

// ── API ─────────────────────────────────────────────────────────────

export const controlApi = {
    // State queries
    getState: () => get<ControlState>("/api/control/state"),
    getBrokerStatus: () => get<BrokerStatus>("/api/control/broker-status"),
    getJobGraph: () => get<{ jobs: JobNode[] }>("/api/control/job-graph"),
    getCalendar: () => get<CalendarInfo>("/api/control/calendar"),
    getConfig: () => get<OrchestratorConfig>("/api/control/config"),
    getAudit: (limit = 50) => get<{ items: AuditEntry[] }>(`/api/control/audit?limit=${limit}`),
    getPortfolioSummary: () => get<PortfolioSummary>("/api/control/portfolio-summary"),
    getPortfolioHistory: () => get<PortfolioHistoryResponse>("/api/control/portfolio-history"),

    // State mutations
    pause: () => put<{ ok: boolean }>("/api/control/pause", { paused: true }),
    resume: () => put<{ ok: boolean }>("/api/control/resume", {}),
    setVolRegime: (regime: string | null) =>
        put<{ ok: boolean }>("/api/control/vol-regime", { regime }),
    setBlockedSymbols: (symbols: string[]) =>
        put<{ ok: boolean }>("/api/control/blocked-symbols", { symbols }),
    setFrozenSleeves: (sleeves: string[]) =>
        put<{ ok: boolean }>("/api/control/frozen-sleeves", { sleeves }),
    setExposureCap: (cap: number | null) =>
        put<{ ok: boolean }>("/api/control/exposure-cap", { cap }),
    setExecutionMode: (mode: string) =>
        put<{ ok: boolean }>("/api/control/execution-mode", { mode }),

    // Actions
    triggerTick: (dryRun = true, session?: string) =>
        post<TickResult>("/api/control/trigger-tick", { dry_run: dryRun, session }),
    flatten: (sleeve?: string) =>
        post<{ ok: boolean }>("/api/control/flatten", { sleeve: sleeve ?? null }),
    submitManualOrder: (order: ManualOrder) =>
        post<{ ok: boolean; order: ManualOrder }>("/api/control/manual-order", order),
};
