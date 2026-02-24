"use client";

import { Card } from "@/components/ui/primitives";

interface Fill {
    symbol?: string;
    qty?: number;
    price?: number;
    time?: string;
    expected_price?: number;
}

export function FillQuality({ fills }: { fills: Fill[] }) {
    if (!fills || fills.length === 0) return null;

    const withSlippage = fills
        .filter((f) => f.expected_price && f.price && f.expected_price > 0)
        .map((f) => ({
            ...f,
            slippage_bps: Math.round(((f.price! - f.expected_price!) / f.expected_price!) * 10_000),
        }));

    const totalSlippage = withSlippage.length > 0
        ? withSlippage.reduce((sum, f) => sum + Math.abs(f.slippage_bps), 0) / withSlippage.length
        : 0;
    const worstSlippage = withSlippage.length > 0
        ? Math.max(...withSlippage.map((f) => Math.abs(f.slippage_bps)))
        : 0;

    return (
        <Card>
            <h2 className="mb-3 text-sm font-semibold">Fill Quality</h2>

            {/* Summary stats */}
            <div className="grid grid-cols-3 gap-3 mb-3 text-xs">
                <div className="rounded bg-surface-2 p-2">
                    <div className="text-muted">Total Fills</div>
                    <div className="text-lg font-semibold">{fills.length}</div>
                </div>
                <div className="rounded bg-surface-2 p-2">
                    <div className="text-muted">Avg Slippage</div>
                    <div className={`text-lg font-semibold ${totalSlippage > 5 ? "text-red-400" : totalSlippage > 2 ? "text-amber-400" : "text-green-400"}`}>
                        {totalSlippage.toFixed(1)} bps
                    </div>
                </div>
                <div className="rounded bg-surface-2 p-2">
                    <div className="text-muted">Worst Fill</div>
                    <div className={`text-lg font-semibold ${worstSlippage > 10 ? "text-red-400" : "text-amber-400"}`}>
                        {worstSlippage.toFixed(1)} bps
                    </div>
                </div>
            </div>

            {/* Detail table */}
            {withSlippage.length > 0 && (
                <table className="w-full text-xs">
                    <thead>
                        <tr className="border-b border-border text-muted text-left">
                            <th className="pb-1">Symbol</th>
                            <th className="pb-1 text-right">Expected</th>
                            <th className="pb-1 text-right">Actual</th>
                            <th className="pb-1 text-right">Slippage</th>
                        </tr>
                    </thead>
                    <tbody>
                        {withSlippage.slice(0, 20).map((f, i) => (
                            <tr key={i} className="border-b border-border/30">
                                <td className="py-1 font-mono">{f.symbol}</td>
                                <td className="py-1 text-right">${f.expected_price?.toFixed(2)}</td>
                                <td className="py-1 text-right">${f.price?.toFixed(2)}</td>
                                <td className={`py-1 text-right font-semibold ${Math.abs(f.slippage_bps) > 5 ? "text-red-400" : "text-green-400"
                                    }`}>
                                    {f.slippage_bps > 0 ? "+" : ""}{f.slippage_bps} bps
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}

            {withSlippage.length === 0 && (
                <p className="text-xs text-secondary">No expected-price data available for slippage analysis.</p>
            )}
        </Card>
    );
}
