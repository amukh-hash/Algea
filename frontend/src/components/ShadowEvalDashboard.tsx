"use client";

import { Card } from "@/components/ui/primitives";

interface ShadowData {
    realized_sharpe?: number;
    assumed_sharpe?: number;
    mean_slippage_drift_bps?: number;
    promotion_eligible?: boolean;
    gate_details?: Record<string, { passed: boolean; value: number; threshold: number }>;
}

function GateIndicator({ name, passed, value, threshold }: { name: string; passed: boolean; value: number; threshold: number }) {
    return (
        <div className={`flex items-center justify-between rounded px-3 py-2 text-xs ${passed ? "bg-green-900/20 border border-green-500/30" : "bg-red-900/20 border border-red-500/30"
            }`}>
            <div className="flex items-center gap-2">
                <span>{passed ? "✅" : "❌"}</span>
                <span className="font-medium">{name}</span>
            </div>
            <div className="text-muted">
                {value.toFixed(3)} {passed ? "≤" : ">"} {threshold.toFixed(3)}
            </div>
        </div>
    );
}

export function ShadowEvalDashboard({ data }: { data?: ShadowData | null }) {
    if (!data) {
        return (
            <Card>
                <h2 className="mb-3 text-sm font-semibold">Shadow Evaluation</h2>
                <p className="text-sm text-secondary">No shadow evaluation data available for this run.</p>
            </Card>
        );
    }

    const sharpeGap = (data.realized_sharpe ?? 0) - (data.assumed_sharpe ?? 0);

    return (
        <Card>
            <h2 className="mb-3 text-sm font-semibold">Shadow Evaluation</h2>

            {/* Promotion indicator */}
            <div className={`rounded-lg p-4 mb-4 text-center ${data.promotion_eligible
                    ? "bg-green-900/20 border border-green-500/50"
                    : "bg-red-900/20 border border-red-500/50"
                }`}>
                <div className="text-2xl mb-1">{data.promotion_eligible ? "✅" : "🚫"}</div>
                <div className={`text-sm font-semibold ${data.promotion_eligible ? "text-green-400" : "text-red-400"}`}>
                    {data.promotion_eligible ? "Eligible for Promotion" : "Not Ready for Promotion"}
                </div>
            </div>

            {/* Sharpe comparison */}
            <div className="grid grid-cols-3 gap-3 mb-4 text-xs">
                <div className="rounded bg-surface-2 p-3 text-center">
                    <div className="text-muted">Assumed Sharpe</div>
                    <div className="text-xl font-semibold">{data.assumed_sharpe?.toFixed(2) ?? "—"}</div>
                </div>
                <div className="rounded bg-surface-2 p-3 text-center">
                    <div className="text-muted">Realized Sharpe</div>
                    <div className="text-xl font-semibold">{data.realized_sharpe?.toFixed(2) ?? "—"}</div>
                </div>
                <div className="rounded bg-surface-2 p-3 text-center">
                    <div className="text-muted">Gap</div>
                    <div className={`text-xl font-semibold ${sharpeGap < 0 ? "text-red-400" : "text-green-400"}`}>
                        {sharpeGap > 0 ? "+" : ""}{sharpeGap.toFixed(2)}
                    </div>
                </div>
            </div>

            {/* Slippage drift */}
            {data.mean_slippage_drift_bps != null && (
                <div className="rounded bg-surface-2 p-3 mb-4 text-xs">
                    <span className="text-muted">Mean Slippage Drift: </span>
                    <span className={`font-semibold ${Math.abs(data.mean_slippage_drift_bps) > 5 ? "text-red-400" : "text-green-400"}`}>
                        {data.mean_slippage_drift_bps.toFixed(1)} bps
                    </span>
                </div>
            )}

            {/* Gate details */}
            {data.gate_details && (
                <div className="space-y-1">
                    <div className="text-xs font-semibold text-muted mb-1">Promotion Gates</div>
                    {Object.entries(data.gate_details).map(([name, gate]) => (
                        <GateIndicator key={name} name={name} {...gate} />
                    ))}
                </div>
            )}
        </Card>
    );
}
