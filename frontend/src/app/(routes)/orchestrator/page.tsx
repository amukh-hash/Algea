"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { orchApi, OrchJob, OrchRun, OrchTarget } from "@/lib/orch";
import {
    Button,
    Card,
    PageHeader,
    Skeleton,
    StatusBadge,
} from "@/components/ui/primitives";
import { ErrorBanner } from "@/components/ui/ErrorBanner";

/* ── helpers ─────────────────────────────────────────────────────── */

function timeAgo(iso: string | null): string {
    if (!iso) return "—";
    const diff = Date.now() - new Date(iso).getTime();
    if (diff < 60_000) return `${Math.round(diff / 1000)}s ago`;
    if (diff < 3_600_000) return `${Math.round(diff / 60_000)}m ago`;
    return `${Math.round(diff / 3_600_000)}h ago`;
}

function sessionColor(session: string): string {
    const map: Record<string, string> = {
        premarket: "text-blue-400",
        open: "text-green-400",
        intraday: "text-emerald-400",
        preclose: "text-amber-400",
        close: "text-orange-400",
        overnight: "text-indigo-400",
    };
    return map[session] ?? "text-secondary";
}

function jobStatusIcon(status: string): string {
    const icons: Record<string, string> = {
        success: "✅",
        failed: "❌",
        skipped: "⏭️",
        running: "🔄",
    };
    return icons[status] ?? "•";
}

function jobStatusColor(status: string): string {
    const map: Record<string, string> = {
        success: "border-l-green-500",
        failed: "border-l-red-500",
        skipped: "border-l-yellow-500",
        running: "border-l-blue-500",
    };
    return map[status] ?? "border-l-gray-500";
}

/* ── main page ───────────────────────────────────────────────────── */

export default function OrchestratorPage() {
    const status = useQuery({
        queryKey: ["orch-status"],
        queryFn: orchApi.getStatus,
        refetchInterval: 10_000,
    });
    const runs = useQuery({
        queryKey: ["orch-runs"],
        queryFn: () => orchApi.listRuns(15),
        refetchInterval: 15_000,
    });
    const positions = useQuery({
        queryKey: ["orch-positions"],
        queryFn: () => orchApi.getPositions(),
        refetchInterval: 30_000,
    });
    const targets = useQuery({
        queryKey: ["orch-targets"],
        queryFn: () => orchApi.getTargets(),
        refetchInterval: 30_000,
    });

    const hb = status.data?.heartbeat;
    const lastRun = status.data?.last_run;

    return (
        <div className="space-y-5">
            <PageHeader
                title="Orchestrator"
                subtitle="Paper trading operations dashboard"
                actions={
                    <Button onClick={() => { status.refetch(); runs.refetch(); positions.refetch(); targets.refetch(); }}>
                        Refresh
                    </Button>
                }
            />

            {/* ── Global error banner ────────────────────────────────── */}
            {status.error && (
                <ErrorBanner
                    error={status.error as Error}
                    onRetry={() => status.refetch()}
                    isRetrying={status.isFetching}
                />
            )}

            {/* ── Status Banner ──────────────────────────────────────── */}
            <div className="grid grid-cols-2 gap-3 rounded-lg border border-border bg-surface-1 p-4 md:grid-cols-5">
                <StatCell label="Mode" value={hb?.mode ?? "—"} highlight={hb?.mode === "paper" ? "text-amber-400" : undefined} />
                <StatCell label="Session" value={hb?.session ?? "—"} highlight={sessionColor(hb?.session ?? "")} />
                <StatCell label="State" value={hb?.state ?? "—"} highlight={hb?.state === "success" ? "text-green-400" : hb?.state === "failed" ? "text-red-400" : undefined} />
                <StatCell label="Last Tick" value={timeAgo(hb?.timestamp ?? null)} />
                <StatCell label="Date" value={status.data?.asof_date ?? "—"} />
            </div>

            {/* ── Positions ──────────────────────────────────────────── */}
            <Card>
                <h2 className="mb-3 text-lg font-semibold">Positions</h2>
                {positions.error && (
                    <ErrorBanner error={positions.error as Error} onRetry={() => positions.refetch()} isRetrying={positions.isFetching} />
                )}
                {positions.isLoading && <Skeleton className="h-16" />}
                {positions.data && positions.data.positions.length === 0 && (
                    <p className="text-sm text-secondary">No open positions</p>
                )}
                {positions.data && positions.data.positions.length > 0 && (
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b border-border text-left text-xs text-muted">
                                <th className="pb-2">Symbol</th>
                                <th className="pb-2 text-right">Qty</th>
                                <th className="pb-2 text-right">Avg Cost</th>
                            </tr>
                        </thead>
                        <tbody>
                            {positions.data.positions.map((p) => (
                                <tr key={p.symbol} className="border-b border-border/30">
                                    <td className="py-2 font-mono font-medium">{p.symbol}</td>
                                    <td className="py-2 text-right">{p.qty}</td>
                                    <td className="py-2 text-right">${p.avg_cost.toLocaleString()}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </Card>

            {/* ── Targets ────────────────────────────────────────────── */}
            <Card>
                <h2 className="mb-3 text-lg font-semibold">Today&apos;s Targets</h2>
                {targets.error && (
                    <ErrorBanner error={targets.error as Error} onRetry={() => targets.refetch()} isRetrying={targets.isFetching} />
                )}
                {targets.isLoading && <Skeleton className="h-16" />}
                {targets.data && Object.keys(targets.data.sleeves).length === 0 && (
                    <p className="text-sm text-secondary">No targets generated yet today</p>
                )}
                {targets.data && Object.entries(targets.data.sleeves).map(([sleeve, data]) => (
                    <SleeveTargets key={sleeve} sleeve={sleeve} targets={data.targets ?? []} />
                ))}
            </Card>

            {/* ── Recent Runs ────────────────────────────────────────── */}
            <Card>
                <h2 className="mb-3 text-lg font-semibold">Recent Ticks</h2>
                {runs.error && (
                    <ErrorBanner error={runs.error as Error} onRetry={() => runs.refetch()} isRetrying={runs.isFetching} />
                )}
                {runs.isLoading && <Skeleton className="h-32" />}
                {runs.data?.items.map((run) => (
                    <RunRow key={run.run_id} run={run} />
                ))}
                {runs.data?.items.length === 0 && (
                    <p className="text-sm text-secondary">No orchestrator runs recorded yet</p>
                )}
            </Card>
        </div>
    );
}

/* ── sub-components ──────────────────────────────────────────────── */

function StatCell({ label, value, highlight }: { label: string; value: string; highlight?: string }) {
    return (
        <div>
            <div className="text-xs text-muted">{label}</div>
            <div className={`text-sm font-medium ${highlight ?? ""}`}>{value}</div>
        </div>
    );
}

function SleeveTargets({ sleeve, targets }: { sleeve: string; targets: OrchTarget[] }) {
    const [open, setOpen] = useState(false);
    const buys = targets.filter((t) => t.side === "buy");
    const sells = targets.filter((t) => t.side === "sell");

    return (
        <div className="mb-3 rounded border border-border/50 bg-surface-2/50 p-3">
            <button className="flex w-full items-center justify-between text-sm font-medium" onClick={() => setOpen((v) => !v)}>
                <span className="capitalize">{sleeve}</span>
                <span className="text-xs text-muted">
                    {targets.length} targets {open ? "▾" : "▸"}
                </span>
            </button>
            {open && targets.length > 0 && (
                <div className="mt-2 grid gap-4 md:grid-cols-2">
                    {buys.length > 0 && (
                        <div>
                            <div className="mb-1 text-xs font-semibold text-green-400">Long ({buys.length})</div>
                            <div className="space-y-1">
                                {buys.map((t) => (
                                    <div key={t.symbol} className="flex justify-between text-xs">
                                        <span className="font-mono">{t.symbol}</span>
                                        <span className="text-muted">{(t.target_weight * 100).toFixed(1)}%{t.score != null ? ` (${t.score.toFixed(2)})` : ""}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                    {sells.length > 0 && (
                        <div>
                            <div className="mb-1 text-xs font-semibold text-red-400">Short ({sells.length})</div>
                            <div className="space-y-1">
                                {sells.map((t) => (
                                    <div key={t.symbol} className="flex justify-between text-xs">
                                        <span className="font-mono">{t.symbol}</span>
                                        <span className="text-muted">{(t.target_weight * 100).toFixed(1)}%{t.score != null ? ` (${t.score.toFixed(2)})` : ""}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}
            {open && targets.length === 0 && (
                <p className="mt-2 text-xs text-secondary">Empty (no targets for this sleeve)</p>
            )}
        </div>
    );
}

function RunRow({ run }: { run: OrchRun }) {
    const [expanded, setExpanded] = useState(false);
    const jobs = useQuery({
        queryKey: ["orch-run-jobs", run.run_id],
        queryFn: () => orchApi.getRunJobs(run.run_id),
        enabled: expanded,
    });

    const ranJobs = (run.meta as Record<string, unknown>)?.ran_jobs as string[] | undefined;
    const failedJobs = (run.meta as Record<string, unknown>)?.failed_jobs as string[] | undefined;

    return (
        <div className="border-b border-border/30 py-2">
            <button
                className="flex w-full items-center justify-between text-sm"
                onClick={() => setExpanded((v) => !v)}
            >
                <div className="flex items-center gap-3">
                    <StatusBadge status={run.status} />
                    <span className={`font-medium ${sessionColor(run.session)}`}>{run.session}</span>
                    <span className="text-xs text-muted">{run.asof_date}</span>
                </div>
                <div className="flex items-center gap-3 text-xs text-muted">
                    {ranJobs && <span className="text-green-400">{ranJobs.length} ran</span>}
                    {failedJobs && failedJobs.length > 0 && <span className="text-red-400">{failedJobs.length} failed</span>}
                    <span>{timeAgo(run.started_at)}</span>
                    <span>{expanded ? "▾" : "▸"}</span>
                </div>
            </button>
            {expanded && jobs.data && (
                <div className="ml-4 mt-2 space-y-1">
                    {jobs.data.items.map((j) => (
                        <JobRow key={j.job_name} job={j} />
                    ))}
                    {jobs.data.items.length === 0 && (
                        <p className="text-xs text-secondary">No job records for this run</p>
                    )}
                </div>
            )}
            {expanded && jobs.error && (
                <ErrorBanner error={jobs.error as Error} onRetry={() => jobs.refetch()} isRetrying={jobs.isFetching} />
            )}
            {expanded && jobs.isLoading && <Skeleton className="ml-4 mt-2 h-10" />}
        </div>
    );
}

function JobRow({ job }: { job: OrchJob }) {
    return (
        <div
            className={`flex items-center justify-between rounded border-l-2 bg-surface-2/30 px-3 py-1.5 text-xs ${jobStatusColor(job.status)}`}
        >
            <div className="flex items-center gap-2">
                <span>{jobStatusIcon(job.status)}</span>
                <span className="font-mono">{job.job_name}</span>
            </div>
            <div className="flex items-center gap-3 text-muted">
                {job.error_summary && <span className="text-red-400">{job.error_summary}</span>}
                {job.started_at && <span>{timeAgo(job.started_at)}</span>}
            </div>
        </div>
    );
}
