"use client";

import { useQuery } from "@tanstack/react-query";
import { controlApi, CalendarInfo } from "@/lib/control";

const SESSION_COLORS: Record<string, string> = {
    PREMARKET: "bg-blue-500/20 border-blue-500",
    OPEN: "bg-green-500/20 border-green-500",
    INTRADAY: "bg-emerald-500/20 border-emerald-500",
    PRECLOSE: "bg-amber-500/20 border-amber-500",
    CLOSE: "bg-orange-500/20 border-orange-500",
    AFTERHOURS: "bg-purple-500/20 border-purple-500",
};

function timeToMinutes(t: string): number {
    const [h, m] = t.split(":").map(Number);
    return h * 60 + m;
}

export function CalendarSchedule() {
    const { data, isLoading } = useQuery({
        queryKey: ["calendar-schedule"],
        queryFn: controlApi.getCalendar,
        refetchInterval: 10_000,
    });

    if (isLoading || !data) return null;

    const windows = Object.entries(data.session_windows);
    const dayStart = 7 * 60; // 07:00
    const dayEnd = 20 * 60; // 20:00
    const dayRange = dayEnd - dayStart;

    const currentMinutes = timeToMinutes(data.current_time);
    const markerPct = Math.max(0, Math.min(100, ((currentMinutes - dayStart) / dayRange) * 100));

    return (
        <div className="rounded-lg border border-border bg-surface-1 p-3">
            <div className="flex items-center justify-between mb-2">
                <div className="text-xs font-semibold text-secondary">
                    {data.date} · {data.is_trading_day ? "Trading Day" : "Non-Trading Day"}
                </div>
                <div className="text-xs text-muted">
                    Current: <span className="font-semibold text-primary">{data.current_session.replace("_", " ")}</span>
                    {" · "}{data.current_time}
                </div>
            </div>

            {/* Timeline bar */}
            <div className="relative h-8 rounded bg-surface-2 overflow-hidden">
                {windows.map(([name, win]) => {
                    const start = timeToMinutes(win.start);
                    const end = timeToMinutes(win.end);
                    const left = ((start - dayStart) / dayRange) * 100;
                    const width = ((end - start) / dayRange) * 100;
                    const colors = SESSION_COLORS[name] ?? "bg-gray-500/20 border-gray-500";
                    const isActive = name.toLowerCase() === data.current_session.toLowerCase();

                    return (
                        <div
                            key={name}
                            className={`absolute top-0 h-full border-l ${colors} flex items-center justify-center ${isActive ? "ring-1 ring-primary z-10" : ""
                                }`}
                            style={{ left: `${left}%`, width: `${width}%` }}
                            title={`${name}: ${win.start} — ${win.end}`}
                        >
                            <span className="text-[0.55rem] font-semibold truncate px-1">
                                {name}
                            </span>
                        </div>
                    );
                })}

                {/* Current time marker */}
                <div
                    className="absolute top-0 h-full w-0.5 bg-primary z-20"
                    style={{ left: `${markerPct}%` }}
                />
            </div>

            {/* Time labels */}
            <div className="flex justify-between mt-1 text-[0.55rem] text-muted">
                <span>07:00</span>
                <span>09:30</span>
                <span>12:00</span>
                <span>16:00</span>
                <span>20:00</span>
            </div>
        </div>
    );
}
