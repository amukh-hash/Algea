"use client";

import { useQuery } from "@tanstack/react-query";
import { orchApi, OrchPosition, OrchTarget } from "@/lib/orch";
import { Card, PageHeader, Skeleton } from "@/components/ui/primitives";
import { ErrorBanner } from "@/components/ui/ErrorBanner";
import { ManualOverrideForm } from "@/components/ManualOverrideForm";
import { TearOffWrapper } from "@/components/TearOffWrapper";
import { FlattenControls } from "@/components/FlattenControls";
import { ExposureBreakdown } from "@/components/ExposureBreakdown";
import { FillQuality } from "@/components/FillQuality";
import { useMemo } from "react";

// Helper to check if date is stale
function isStale(asofDate?: string) {
    if (!asofDate) return false;
    const today = new Date().toISOString().split("T")[0];
    return asofDate < today;
}

export default function PortfolioPage() {
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
    const fillsQuery = useQuery({
        queryKey: ["orch-fills"],
        queryFn: () => orchApi.getFills(),
        refetchInterval: 30_000,
    });

    // Compute Target vs Actual matrix
    const matrix = useMemo(() => {
        if (!positions.data || !targets.data) return [];
        const rows: Record<string, { symbol: string; actualQty: number; targetQty: number; }> = {};

        // 1. Populate actuals
        positions.data.positions.forEach(p => {
            rows[p.symbol] = { symbol: p.symbol, actualQty: p.qty, targetQty: 0 };
        });

        // 2. We don't have absolute target QTY from the API, we have target_weights.
        // For a true Target vs Actual matrix, we'd need AUM to convert weights > qty.
        // For now, we just list symbols that have a target weight but no actual position,
        // or just visualize the targets side-by-side.
        Object.values(targets.data.sleeves).forEach(sleeve => {
            sleeve.targets?.forEach(t => {
                if (!rows[t.symbol]) {
                    rows[t.symbol] = { symbol: t.symbol, actualQty: 0, targetQty: 1 }; // Indicator only
                }
            });
        });

        return Object.values(rows).sort((a, b) => a.symbol.localeCompare(b.symbol));
    }, [positions.data, targets.data]);

    const anyStale = isStale(positions.data?.asof_date) || isStale(targets.data?.asof_date);

    return (
        <div className="space-y-6">
            <PageHeader
                title="Portfolio & Execution"
                subtitle="Live holdings, aggregate targets, and daily fills blotter"
                actions={
                    <button
                        onClick={() => { positions.refetch(); targets.refetch(); fillsQuery.refetch(); }}
                        className="rounded bg-primary px-3 py-1.5 text-xs font-semibold text-primary-foreground hover:bg-primary/90"
                    >
                        Refresh
                    </button>
                }
            />

            {anyStale && (
                <div className="rounded border-l-4 border-amber-500 bg-amber-500/10 p-3 text-sm text-amber-200">
                    <strong>Warning:</strong> The data currently displayed has an as-of date from a previous session.
                </div>
            )}

            <FlattenControls />
            <ExposureBreakdown />
            <ManualOverrideForm />

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                    <h2 className="mb-4 text-lg font-semibold flex items-center justify-between">
                        <span>Target vs Actual Matrix</span>
                        {positions.data?.asof_date && <span className="text-xs font-normal text-muted">As of: {positions.data.asof_date}</span>}
                    </h2>

                    {positions.isLoading || targets.isLoading ? (
                        <Skeleton className="h-64" />
                    ) : positions.error ? (
                        <ErrorBanner error={positions.error as Error} />
                    ) : (
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className="border-b border-border text-left text-xs text-muted">
                                        <th className="pb-2">Symbol</th>
                                        <th className="pb-2 text-right">Actual Qty</th>
                                        <th className="pb-2 text-right">Target Action</th>
                                        <th className="pb-2 text-right">Drift</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {matrix.length === 0 ? (
                                        <tr>
                                            <td colSpan={4} className="py-4 text-center text-secondary">No data available</td>
                                        </tr>
                                    ) : (
                                        matrix.map(row => {
                                            const hasActual = row.actualQty !== 0;
                                            const hasTarget = row.targetQty !== 0; // Simplified for MVP
                                            let drift = "In Sync";
                                            let driftClass = "text-green-400";
                                            if (hasTarget && !hasActual) { drift = "Missing Pos"; driftClass = "text-red-400"; }
                                            if (hasActual && !hasTarget) { drift = "Stale Pos"; driftClass = "text-amber-400"; }

                                            return (
                                                <tr key={row.symbol} className="border-b border-border/30">
                                                    <td className="py-2 font-mono font-medium">{row.symbol}</td>
                                                    <td className="py-2 text-right">{row.actualQty}</td>
                                                    <td className="py-2 text-right">{hasTarget ? "Has Target" : "None"}</td>
                                                    <td className={`py-2 text-right ${driftClass}`}>{drift}</td>
                                                </tr>
                                            );
                                        })
                                    )}
                                </tbody>
                            </table>
                        </div>
                    )}
                </Card>

                <TearOffWrapper id="fills" title="Fills Blotter">
                    <Card>
                        <h2 className="mb-4 text-lg font-semibold flex items-center justify-between">
                            <span>Fills Blotter</span>
                            {fillsQuery.data?.asof_date && <span className="text-xs font-normal text-muted">As of: {fillsQuery.data.asof_date}</span>}
                        </h2>

                        {fillsQuery.isLoading ? (
                            <Skeleton className="h-64" />
                        ) : fillsQuery.error ? (
                            <ErrorBanner error={fillsQuery.error as Error} />
                        ) : (
                            <div className="overflow-x-auto">
                                <table className="w-full text-sm">
                                    <thead>
                                        <tr className="border-b border-border text-left text-xs text-muted">
                                            <th className="pb-2">Time</th>
                                            <th className="pb-2">Symbol</th>
                                            <th className="pb-2 text-right">Qty</th>
                                            <th className="pb-2 text-right">Price</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {(!fillsQuery.data?.fills || fillsQuery.data.fills.length === 0) ? (
                                            <tr>
                                                <td colSpan={4} className="py-4 text-center text-secondary">No fills recorded today</td>
                                            </tr>
                                        ) : (
                                            (fillsQuery.data.fills as any[]).map((f, i) => (
                                                <tr key={i} className="border-b border-border/30">
                                                    <td className="py-2 text-muted">{new Date(f.time ?? Date.now()).toLocaleTimeString()}</td>
                                                    <td className="py-2 font-mono font-medium">{f.symbol || "UNK"}</td>
                                                    <td className={`py-2 text-right ${f.qty > 0 ? "text-green-400" : "text-red-400"}`}>
                                                        {f.qty > 0 ? "+" : ""}{f.qty}
                                                    </td>
                                                    <td className="py-2 text-right">${f.price?.toLocaleString()}</td>
                                                </tr>
                                            ))
                                        )}
                                    </tbody>
                                </table>
                            </div>
                        )}
                    </Card>
                </TearOffWrapper>
            </div>

            <FillQuality fills={(fillsQuery.data?.fills ?? []) as any[]} />
        </div>
    );
}
