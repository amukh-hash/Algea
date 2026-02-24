"use client";

import { useQuery } from "@tanstack/react-query";
import { controlApi, BrokerStatus, ControlState, CalendarInfo } from "@/lib/control";

const SESSION_COLORS: Record<string, string> = {
    premarket: "bg-blue-500",
    open: "bg-green-500",
    intraday: "bg-emerald-500",
    preclose: "bg-amber-500",
    close: "bg-orange-500",
    overnight: "bg-indigo-500",
    afterhours: "bg-purple-500",
    closed: "bg-gray-500",
};

function Dot({ color }: { color: string }) {
    return <span className={`inline-block h-2 w-2 rounded-full ${color}`} />;
}

function PulsingDot({ color }: { color: string }) {
    return <span className={`inline-block h-2 w-2 rounded-full ${color} animate-pulse`} />;
}

export function SystemHealthBar() {
    const broker = useQuery({
        queryKey: ["broker-status"],
        queryFn: controlApi.getBrokerStatus,
        refetchInterval: 5_000,
        retry: 1,
    });

    const state = useQuery({
        queryKey: ["control-state"],
        queryFn: controlApi.getState,
        refetchInterval: 5_000,
        retry: 1,
    });

    const calendar = useQuery({
        queryKey: ["calendar"],
        queryFn: controlApi.getCalendar,
        refetchInterval: 10_000,
        retry: 1,
    });

    const bs: Partial<BrokerStatus> = broker.data ?? {};
    const cs: Partial<ControlState> = state.data ?? {};
    const cal: Partial<CalendarInfo> = calendar.data ?? {};

    const sessionKey = (cal.current_session ?? "closed").toLowerCase();
    const sessionColor = SESSION_COLORS[sessionKey] ?? "bg-gray-500";

    return (
        <div className="flex items-center gap-3 text-xs">
            {/* Broker */}
            <div className="flex items-center gap-1.5" title={bs.gateway_url ?? "unknown"}>
                {bs.connected ? <Dot color="bg-green-500" /> : <Dot color="bg-red-500" />}
                <span className={bs.connected ? "text-green-400" : "text-red-400"}>
                    {bs.connected ? "Broker" : "Disconnected"}
                </span>
            </div>

            {/* Session */}
            <div className="flex items-center gap-1.5">
                <Dot color={sessionColor} />
                <span className="uppercase tracking-wider font-semibold" style={{ fontSize: "0.65rem" }}>
                    {cal.current_session?.replace("_", " ") ?? "—"}
                </span>
            </div>

            {/* Mode */}
            <span className={`rounded px-1.5 py-0.5 text-[0.65rem] font-bold uppercase ${cs.execution_mode === "ibkr" ? "bg-red-600 text-white" :
                    cs.execution_mode === "paper" ? "bg-amber-600 text-white" :
                        "bg-gray-600 text-gray-200"
                }`}>
                {cs.execution_mode ?? "—"}
            </span>

            {/* Paused */}
            {cs.paused && (
                <div className="flex items-center gap-1">
                    <PulsingDot color="bg-amber-400" />
                    <span className="text-amber-400 font-semibold">PAUSED</span>
                </div>
            )}

            {/* Vol regime override */}
            {cs.vol_regime_override && (
                <span className={`rounded px-1.5 py-0.5 text-[0.65rem] font-bold ${cs.vol_regime_override === "CRASH_RISK" ? "bg-red-900 text-red-200" : "bg-amber-900 text-amber-200"
                    }`}>
                    {cs.vol_regime_override}
                </span>
            )}

            {/* Time */}
            <span className="text-secondary ml-auto">{cal.current_time ?? ""}</span>
        </div>
    );
}
