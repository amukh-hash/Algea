"use client";

import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { orchApi } from "@/lib/orch";

export default function TearOffPage({ params }: { params: { component: string } }) {
    const searchParams = useSearchParams();
    const runId = searchParams.get("runId");

    // For MVP, we will render specific tearoff components based on the ID.
    if (params.component === "live-events") {
        return (
            <div className="h-full flex flex-col">
                <h1 className="text-lg font-bold mb-4 font-mono">Live Events Tear-Off</h1>
                <p className="text-sm text-secondary mb-4 border-b border-border pb-2">
                    Monitoring events globally or for run: {runId ?? "Global"}
                </p>
                <div className="flex-1 overflow-auto bg-surface-2/50 rounded p-4 font-mono text-xs">
                    {/* Simplified tear-off view, real implementation would mount the exact component used in the main app */}
                    <div className="animate-pulse text-info">Connecting to EventSource...</div>
                </div>
            </div>
        );
    }

    if (params.component === "fills") {
        return (
            <div className="h-full flex flex-col p-2">
                <h1 className="text-lg font-bold mb-4">Fills Blotter Tear-Off</h1>
                <FillsTearOff />
            </div>
        );
    }

    return (
        <div className="flex h-full items-center justify-center">
            <p className="text-secondary">Unknown tear-off component: {params.component}</p>
        </div>
    );
}

function FillsTearOff() {
    const fillsQuery = useQuery({
        queryKey: ["orch-fills"],
        queryFn: () => orchApi.getFills(),
        refetchInterval: 10_000,
    });

    if (fillsQuery.isLoading) return <div>Loading fills...</div>;

    return (
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
    );
}
