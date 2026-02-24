"use client";

import { useQuery } from "@tanstack/react-query";
import { orchApi } from "@/lib/orch";
import { Card } from "@/components/ui/primitives";

const SLEEVE_ALLOC: Record<string, number> = { core: 50, vrp: 30, selector: 20 };
const SLEEVE_COLORS: Record<string, string> = {
    core: "bg-blue-500",
    vrp: "bg-purple-500",
    selector: "bg-emerald-500",
};

export function ExposureBreakdown() {
    const targets = useQuery({
        queryKey: ["orch-targets"],
        queryFn: () => orchApi.getTargets(),
        refetchInterval: 30_000,
    });

    if (!targets.data) return null;

    const sleeves = targets.data.sleeves ?? {};
    const sleeveNames = Object.keys(sleeves);
    if (sleeveNames.length === 0) return null;

    // Compute per-sleeve metrics
    const rows = sleeveNames.map((name) => {
        const tgts = sleeves[name]?.targets ?? [];
        const gross = tgts.reduce((sum: number, t: any) => sum + Math.abs(t.target_weight ?? 0), 0);
        const net = tgts.reduce((sum: number, t: any) => sum + (t.target_weight ?? 0), 0);
        const nLong = tgts.filter((t: any) => (t.target_weight ?? 0) > 0).length;
        const nShort = tgts.filter((t: any) => (t.target_weight ?? 0) < 0).length;
        return { name, gross, net, nLong, nShort, allocation: SLEEVE_ALLOC[name] ?? 0 };
    });

    // Cross-sleeve concentration
    const symbolMap: Record<string, string[]> = {};
    for (const name of sleeveNames) {
        for (const t of sleeves[name]?.targets ?? []) {
            const sym = t.symbol ?? "";
            if (sym) {
                if (!symbolMap[sym]) symbolMap[sym] = [];
                symbolMap[sym].push(name);
            }
        }
    }
    const crossSleeve = Object.entries(symbolMap).filter(([, s]) => s.length > 1);

    return (
        <Card>
            <h2 className="mb-3 text-sm font-semibold">Sleeve Exposure Breakdown</h2>

            {/* Allocation bar */}
            <div className="flex h-4 rounded overflow-hidden mb-3">
                {rows.map((r) => (
                    <div
                        key={r.name}
                        className={`${SLEEVE_COLORS[r.name] ?? "bg-gray-500"} flex items-center justify-center text-[0.55rem] font-bold text-white`}
                        style={{ width: `${r.allocation}%` }}
                    >
                        {r.name} {r.allocation}%
                    </div>
                ))}
            </div>

            {/* Per-sleeve table */}
            <table className="w-full text-xs">
                <thead>
                    <tr className="border-b border-border text-muted text-left">
                        <th className="pb-1">Sleeve</th>
                        <th className="pb-1 text-right">Alloc %</th>
                        <th className="pb-1 text-right">Gross</th>
                        <th className="pb-1 text-right">Net</th>
                        <th className="pb-1 text-right">Long</th>
                        <th className="pb-1 text-right">Short</th>
                    </tr>
                </thead>
                <tbody>
                    {rows.map((r) => (
                        <tr key={r.name} className="border-b border-border/30">
                            <td className="py-1 font-medium capitalize">{r.name}</td>
                            <td className="py-1 text-right">{r.allocation}%</td>
                            <td className="py-1 text-right">{(r.gross * 100).toFixed(1)}%</td>
                            <td className={`py-1 text-right ${r.net > 0 ? "text-green-400" : r.net < 0 ? "text-red-400" : ""}`}>
                                {(r.net * 100).toFixed(1)}%
                            </td>
                            <td className="py-1 text-right text-green-400">{r.nLong}</td>
                            <td className="py-1 text-right text-red-400">{r.nShort}</td>
                        </tr>
                    ))}
                </tbody>
            </table>

            {/* Cross-sleeve concentration warning */}
            {crossSleeve.length > 0 && (
                <div className="mt-3 rounded border border-amber-500/50 bg-amber-500/10 p-2 text-xs text-amber-200">
                    <strong>Cross-sleeve concentration:</strong>{" "}
                    {crossSleeve.map(([sym, sleeves]) => `${sym} (${sleeves.join(", ")})`).join("; ")}
                </div>
            )}
        </Card>
    );
}
