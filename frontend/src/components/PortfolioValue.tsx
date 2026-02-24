"use client";

import { useQuery } from "@tanstack/react-query";
import { useState, useMemo } from "react";
import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
} from "recharts";
import {
    controlApi,
    PortfolioHolding,
    PortfolioSummary,
    PortfolioHistoryResponse,
} from "@/lib/control";

// ── Constants ──────────────────────────────────────────────────────────

type SleeveTab = "all" | "core" | "vrp" | "selector";

const SLEEVE_TABS: { id: SleeveTab; label: string; color: string }[] = [
    { id: "all", label: "All", color: "#6366f1" },
    { id: "core", label: "Core", color: "#f59e0b" },
    { id: "vrp", label: "VRP", color: "#10b981" },
    { id: "selector", label: "Selector", color: "#3b82f6" },
];

const SLEEVE_COLORS: Record<string, string> = {
    core: "#f59e0b",
    vrp: "#10b981",
    selector: "#3b82f6",
};

// ── Formatters ─────────────────────────────────────────────────────────

const fmt = (n: number) =>
    n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

const fmtCompact = (n: number) => {
    if (n === 0) return "—";
    const abs = Math.abs(n);
    if (abs >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
    if (abs >= 1_000) return `$${(n / 1_000).toFixed(1)}K`;
    return `$${fmt(n)}`;
};

const fmtTime = (ts: string) => {
    const d = new Date(ts);
    return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
};

// ── Sub-components ─────────────────────────────────────────────────────

function PnlValue({ value }: { value: number }) {
    const color = value > 0 ? "text-emerald-400" : value < 0 ? "text-red-400" : "text-muted";
    const sign = value > 0 ? "+" : "";
    return <span className={`${color} font-semibold tabular-nums`}>{sign}${fmt(value)}</span>;
}

function TypeBadge({ type }: { type: string }) {
    const cls = type === "future"
        ? "bg-amber-500/20 text-amber-400 border-amber-500/30"
        : "bg-blue-500/20 text-blue-400 border-blue-500/30";
    const label = type === "future" ? "FUT" : "EQ";
    return (
        <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded border ${cls}`}>
            {label}
        </span>
    );
}

function SleeveBadge({ sleeve }: { sleeve: string }) {
    const color = SLEEVE_COLORS[sleeve] ?? "#6b7280";
    return (
        <span
            className="text-[10px] font-bold px-1.5 py-0.5 rounded border"
            style={{
                backgroundColor: `${color}20`,
                color: color,
                borderColor: `${color}50`,
            }}
        >
            {sleeve.toUpperCase()}
        </span>
    );
}

// ── Chart component ────────────────────────────────────────────────────

function IntradayChart({
    data,
    activeTab,
}: {
    data: PortfolioHistoryResponse | undefined;
    activeTab: SleeveTab;
}) {
    if (!data || data.points.length < 2) {
        return (
            <div className="h-40 flex items-center justify-center text-xs text-muted">
                Chart available after 2+ snapshots (auto-updates every 60s)
            </div>
        );
    }

    const chartData = data.points.map((p) => ({
        time: fmtTime(p.ts),
        total: p.total_value,
        core: p.core,
        vrp: p.vrp,
        selector: p.selector,
        pnl: p.unrealized_pnl,
    }));

    return (
        <div className="h-40">
            <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                    <defs>
                        <linearGradient id="gradTotal" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#6366f1" stopOpacity={0.3} />
                            <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="gradCore" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.3} />
                            <stop offset="100%" stopColor="#f59e0b" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="gradVrp" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#10b981" stopOpacity={0.3} />
                            <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="gradSelector" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.3} />
                            <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis
                        dataKey="time"
                        tick={{ fill: "#94a3b8", fontSize: 10 }}
                        tickLine={false}
                        axisLine={{ stroke: "#334155" }}
                    />
                    <YAxis
                        tick={{ fill: "#94a3b8", fontSize: 10 }}
                        tickLine={false}
                        axisLine={false}
                        tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
                        width={55}
                    />
                    <Tooltip
                        contentStyle={{
                            backgroundColor: "#1e293b",
                            border: "1px solid #334155",
                            borderRadius: "8px",
                            fontSize: "12px",
                        }}
                        labelStyle={{ color: "#94a3b8" }}
                        formatter={(value: number) => [`$${fmt(value)}`, undefined]}
                    />
                    {activeTab === "all" ? (
                        <Area
                            type="monotone"
                            dataKey="total"
                            stroke="#6366f1"
                            fill="url(#gradTotal)"
                            strokeWidth={2}
                            name="Total Value"
                        />
                    ) : activeTab === "core" ? (
                        <Area
                            type="monotone"
                            dataKey="core"
                            stroke="#f59e0b"
                            fill="url(#gradCore)"
                            strokeWidth={2}
                            name="Core"
                        />
                    ) : activeTab === "vrp" ? (
                        <Area
                            type="monotone"
                            dataKey="vrp"
                            stroke="#10b981"
                            fill="url(#gradVrp)"
                            strokeWidth={2}
                            name="VRP"
                        />
                    ) : (
                        <Area
                            type="monotone"
                            dataKey="selector"
                            stroke="#3b82f6"
                            fill="url(#gradSelector)"
                            strokeWidth={2}
                            name="Selector"
                        />
                    )}
                </AreaChart>
            </ResponsiveContainer>
        </div>
    );
}

// ── Main component ─────────────────────────────────────────────────────

export function PortfolioValue() {
    const [activeTab, setActiveTab] = useState<SleeveTab>("all");

    const { data, isLoading, error } = useQuery({
        queryKey: ["portfolio-summary"],
        queryFn: controlApi.getPortfolioSummary,
        refetchInterval: 15_000, // live M2M every 15s
    });

    const { data: historyData } = useQuery({
        queryKey: ["portfolio-history"],
        queryFn: controlApi.getPortfolioHistory,
        refetchInterval: 60_000, // match snapshot interval
    });

    // Filter holdings by active tab
    const filteredHoldings = useMemo(() => {
        if (!data) return [];
        if (activeTab === "all") return data.holdings;
        return data.holdings.filter((h: PortfolioHolding) => h.sleeve === activeTab);
    }, [data, activeTab]);

    // Calculate tab-level totals
    const tabTotals = useMemo(() => {
        if (!data) return { margin: 0, notional: 0, pnl: 0, equity: 0, count: 0 };
        if (activeTab === "all") {
            return {
                margin: data.total_margin,
                notional: data.total_notional,
                pnl: data.total_unrealized_pnl,
                equity: data.total_equity_cost,
                count: data.position_count,
            };
        }
        const sleeve = data.sleeves?.[activeTab];
        return sleeve
            ? { margin: sleeve.margin, notional: sleeve.notional, pnl: sleeve.unrealized_pnl, equity: sleeve.equity_cost, count: sleeve.position_count }
            : { margin: 0, notional: 0, pnl: 0, equity: 0, count: 0 };
    }, [data, activeTab]);

    if (isLoading) {
        return (
            <div className="rounded-lg border border-border bg-surface-1 p-5">
                <div className="text-xs uppercase tracking-wider text-muted mb-3">Paper Portfolio Value</div>
                <div className="space-y-2">
                    <div className="h-8 bg-surface-2 rounded animate-pulse" />
                    <div className="h-4 bg-surface-2 rounded animate-pulse w-2/3" />
                    <div className="h-40 bg-surface-2 rounded animate-pulse" />
                </div>
            </div>
        );
    }

    if (error || !data) {
        return (
            <div className="rounded-lg border border-red-500/30 bg-surface-1 p-5 text-sm text-red-400">
                Portfolio data unavailable
            </div>
        );
    }

    const pnlPct = data.account_equity > 0 ? (data.total_pnl / data.account_equity) * 100 : 0;
    const hasFutures = data.holdings.some((h: PortfolioHolding) => h.instrument_type === "future");

    return (
        <div className="rounded-lg border border-border bg-gradient-to-r from-surface-1 to-surface-1/80 overflow-hidden">
            {/* ── Header with total value ─────────────────────────────── */}
            <div className="px-5 pt-4 pb-3 flex items-center justify-between">
                <div>
                    <div className="text-xs uppercase tracking-wider text-muted mb-1">Paper Portfolio Value</div>
                    <div className="text-3xl font-bold tabular-nums">${fmt(data.total_value)}</div>
                    <div className="text-xs text-muted mt-0.5">
                        As of {data.asof_date} · {data.position_count} position{data.position_count !== 1 ? "s" : ""} · {data.fill_count} fills today
                    </div>
                </div>
                <div className="text-right">
                    <PnlValue value={data.total_pnl} />
                    <div className={`text-xs mt-0.5 ${pnlPct >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                        {pnlPct >= 0 ? "+" : ""}{pnlPct.toFixed(2)}%
                    </div>
                </div>
            </div>

            {/* ── Sleeve Tabs ────────────────────────────────────────── */}
            <div className="px-5 border-t border-border/50">
                <div className="flex gap-0.5 -mb-px">
                    {SLEEVE_TABS.map((tab) => {
                        const isActive = activeTab === tab.id;
                        const count = tab.id === "all"
                            ? data.position_count
                            : (data.sleeves?.[tab.id]?.position_count ?? 0);

                        // Skip tabs with no positions (unless "All")
                        if (tab.id !== "all" && count === 0) return null;

                        return (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`
                                    px-4 py-2.5 text-xs font-semibold tracking-wide uppercase
                                    transition-all duration-150 border-b-2
                                    ${isActive
                                        ? "border-current text-white"
                                        : "border-transparent text-muted hover:text-white/70 hover:border-white/20"
                                    }
                                `}
                                style={isActive ? { color: tab.color, borderColor: tab.color } : undefined}
                            >
                                {tab.label}
                                {count > 0 && (
                                    <span className="ml-1.5 text-[10px] opacity-60">({count})</span>
                                )}
                            </button>
                        );
                    })}
                </div>
            </div>

            {/* ── Intraday Chart ──────────────────────────────────────── */}
            <div className="px-5 py-3 border-t border-border/50">
                <IntradayChart data={historyData} activeTab={activeTab} />
            </div>

            {/* ── Breakdown strip ─────────────────────────────────────── */}
            <div className="px-5 py-3 border-t border-border/50 grid grid-cols-2 md:grid-cols-6 gap-3 text-sm">
                <div>
                    <div className="text-xs text-muted">Starting Equity</div>
                    <div className="font-semibold tabular-nums">${fmt(data.account_equity)}</div>
                </div>
                <div>
                    <div className="text-xs text-muted">Cash Available</div>
                    <div className="font-semibold tabular-nums">${fmt(data.cash)}</div>
                </div>
                {hasFutures && (
                    <>
                        <div>
                            <div className="text-xs text-muted">Margin Posted</div>
                            <div className="font-semibold tabular-nums">${fmt(tabTotals.margin)}</div>
                        </div>
                        <div>
                            <div className="text-xs text-muted">Notional Exposure</div>
                            <div className="font-semibold tabular-nums">{fmtCompact(tabTotals.notional)}</div>
                        </div>
                    </>
                )}
                <div>
                    <div className="text-xs text-muted">Unrealized P&L</div>
                    <PnlValue value={tabTotals.pnl} />
                </div>
                <div>
                    <div className="text-xs text-muted">Realized P&L</div>
                    <PnlValue value={data.realized_pnl} />
                </div>
            </div>

            {/* ── Holdings table ──────────────────────────────────────── */}
            {filteredHoldings.length > 0 && (
                <div className="px-5 py-3 border-t border-border/50">
                    <div className="text-xs uppercase tracking-wider text-muted mb-2">Holdings</div>
                    <table className="w-full text-xs">
                        <thead>
                            <tr className="text-muted text-left">
                                <th className="pb-1.5 pr-3">Symbol</th>
                                <th className="pb-1.5 pr-3">Type</th>
                                {activeTab === "all" && <th className="pb-1.5 pr-3">Sleeve</th>}
                                <th className="pb-1.5 text-right pr-3">Qty</th>
                                <th className="pb-1.5 text-right pr-3">Mult</th>
                                <th className="pb-1.5 text-right pr-3">Avg Cost</th>
                                <th className="pb-1.5 text-right pr-3">Last Price</th>
                                <th className="pb-1.5 text-right pr-3">Margin</th>
                                <th className="pb-1.5 text-right pr-3">Notional</th>
                                <th className="pb-1.5 text-right">Unreal P&L</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredHoldings.map((h: PortfolioHolding) => {
                                const priceChanged = Math.abs(h.last_price - h.avg_cost) > 0.001;
                                return (
                                    <tr key={h.symbol} className="border-t border-border/30">
                                        <td className="py-1.5 pr-3 font-mono font-medium">{h.symbol}</td>
                                        <td className="py-1.5 pr-3"><TypeBadge type={h.instrument_type} /></td>
                                        {activeTab === "all" && (
                                            <td className="py-1.5 pr-3"><SleeveBadge sleeve={h.sleeve} /></td>
                                        )}
                                        <td className="py-1.5 text-right pr-3 tabular-nums">{h.qty}</td>
                                        <td className="py-1.5 text-right pr-3 tabular-nums text-muted">
                                            {h.multiplier !== 1 ? `×${h.multiplier}` : "—"}
                                        </td>
                                        <td className="py-1.5 text-right pr-3 tabular-nums">${fmt(h.avg_cost)}</td>
                                        <td className={`py-1.5 text-right pr-3 tabular-nums ${priceChanged ? "text-cyan-400" : ""}`}>
                                            ${fmt(h.last_price)}
                                            {priceChanged && <span className="ml-1 text-[10px]">●</span>}
                                        </td>
                                        <td className="py-1.5 text-right pr-3 tabular-nums">{fmtCompact(h.margin_posted)}</td>
                                        <td className="py-1.5 text-right pr-3 tabular-nums">{fmtCompact(h.notional)}</td>
                                        <td className="py-1.5 text-right"><PnlValue value={h.unrealized_pnl} /></td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            )}

            {/* ── Live indicator ──────────────────────────────────────── */}
            <div className="px-5 py-2 border-t border-border/50 flex items-center justify-between text-[10px] text-muted">
                <div className="flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    Live M2M · refreshes every 15s
                </div>
                <div>Chart snapshots every 60s</div>
            </div>
        </div>
    );
}
