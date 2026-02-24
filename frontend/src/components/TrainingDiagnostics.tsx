"use client";

import { Card } from "@/components/ui/primitives";

interface DiagnosticsData {
    n_days?: number;
    mean?: number;
    vol?: number;
    skew?: number;
    kurtosis?: number;
    worst_1pct?: number;
    cvar_1pct?: number;
    max_drawdown?: number;
    zero_return_frac?: number;
}

function kpi(label: string, value: number | undefined, format: (n: number) => string, good?: (n: number) => boolean) {
    if (value === undefined) return null;
    const color = good ? (good(value) ? "text-green-400" : "text-red-400") : "";
    return (
        <div className="rounded bg-surface-2 p-2 text-center">
            <div className="text-[0.6rem] text-muted uppercase tracking-wider">{label}</div>
            <div className={`text-lg font-semibold ${color}`}>{format(value)}</div>
        </div>
    );
}

export function TrainingDiagnostics({ data }: { data?: DiagnosticsData | null }) {
    if (!data || !data.n_days) return null;

    const sharpe = data.vol && data.vol > 0 ? (data.mean! / data.vol) * Math.sqrt(252) : null;
    const winRate = data.mean !== undefined && data.n_days ? undefined : undefined; // Would need daily sign counts

    return (
        <Card>
            <h2 className="mb-3 text-sm font-semibold">Training Diagnostics</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {sharpe != null && kpi("Annualized Sharpe", sharpe, (n) => n.toFixed(2), (n) => n > 0.5)}
                {kpi("Max Drawdown", data.max_drawdown, (n) => `${(n * 100).toFixed(1)}%`, (n) => n > -0.2)}
                {kpi("Daily Mean", data.mean, (n) => `${(n * 100).toFixed(3)}%`)}
                {kpi("Daily Vol", data.vol, (n) => `${(n * 100).toFixed(3)}%`)}
                {kpi("Skew", data.skew, (n) => n.toFixed(2))}
                {kpi("Kurtosis", data.kurtosis, (n) => n.toFixed(2))}
                {kpi("Worst 1%", data.worst_1pct, (n) => `${(n * 100).toFixed(2)}%`, (n) => n > -0.05)}
                {kpi("CVaR 1%", data.cvar_1pct, (n) => `${(n * 100).toFixed(2)}%`, (n) => n > -0.05)}
                {kpi("N Days", data.n_days, (n) => n.toString(), (n) => n > 100)}
                {kpi("Zero Return %", data.zero_return_frac, (n) => `${(n * 100).toFixed(1)}%`, (n) => n < 0.1)}
            </div>
        </Card>
    );
}
