"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { controlApi, ControlState } from "@/lib/control";
import { Card } from "@/components/ui/primitives";

const REGIMES = [
    { value: "", label: "Normal (No Override)" },
    { value: "CAUTION", label: "⚠ Caution" },
    { value: "CRASH_RISK", label: "🔴 Crash Risk" },
];

const SLEEVES = ["core", "vrp", "selector"];

export function RiskPanel() {
    const qc = useQueryClient();
    const state = useQuery({
        queryKey: ["control-state"],
        queryFn: controlApi.getState,
        refetchInterval: 5_000,
    });

    const regimeMut = useMutation({
        mutationFn: (regime: string | null) => controlApi.setVolRegime(regime),
        onSuccess: () => qc.invalidateQueries({ queryKey: ["control-state"] }),
    });

    const freezeMut = useMutation({
        mutationFn: (sleeves: string[]) => controlApi.setFrozenSleeves(sleeves),
        onSuccess: () => qc.invalidateQueries({ queryKey: ["control-state"] }),
    });

    const capMut = useMutation({
        mutationFn: (cap: number | null) => controlApi.setExposureCap(cap),
        onSuccess: () => qc.invalidateQueries({ queryKey: ["control-state"] }),
    });

    const cs: ControlState = state.data ?? {
        paused: false, vol_regime_override: null, blocked_symbols: [],
        frozen_sleeves: [], gross_exposure_cap: null, execution_mode: "paper",
    };

    const [capInput, setCapInput] = useState("");

    return (
        <Card>
            <h2 className="mb-3 text-sm font-semibold">Risk Controls</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Vol Regime */}
                <div>
                    <label className="block text-xs text-muted mb-1">Vol Regime Override</label>
                    <select
                        value={cs.vol_regime_override ?? ""}
                        onChange={(e) => regimeMut.mutate(e.target.value || null)}
                        className="w-full rounded border border-border bg-surface-2 px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
                    >
                        {REGIMES.map((r) => (
                            <option key={r.value} value={r.value}>{r.label}</option>
                        ))}
                    </select>
                    <div className={`mt-1 h-2 rounded-full ${cs.vol_regime_override === "CRASH_RISK" ? "bg-red-500" :
                            cs.vol_regime_override === "CAUTION" ? "bg-amber-500" :
                                "bg-green-500"
                        }`} />
                </div>

                {/* Frozen Sleeves */}
                <div>
                    <label className="block text-xs text-muted mb-1">Freeze Sleeves</label>
                    <div className="space-y-1">
                        {SLEEVES.map((s) => (
                            <label key={s} className="flex items-center gap-2 text-xs">
                                <input
                                    type="checkbox"
                                    checked={cs.frozen_sleeves.includes(s)}
                                    onChange={(e) => {
                                        const next = e.target.checked
                                            ? [...cs.frozen_sleeves, s]
                                            : cs.frozen_sleeves.filter((x) => x !== s);
                                        freezeMut.mutate(next);
                                    }}
                                />
                                <span className={cs.frozen_sleeves.includes(s) ? "text-amber-400 line-through" : ""}>
                                    {s}
                                </span>
                            </label>
                        ))}
                    </div>
                </div>

                {/* Exposure Cap */}
                <div>
                    <label className="block text-xs text-muted mb-1">
                        Gross Exposure Cap {cs.gross_exposure_cap != null ? `(${cs.gross_exposure_cap})` : "(none)"}
                    </label>
                    <div className="flex gap-1">
                        <input
                            type="number"
                            step="0.1"
                            min="0.1"
                            placeholder="e.g. 1.5"
                            value={capInput}
                            onChange={(e) => setCapInput(e.target.value)}
                            className="flex-1 rounded border border-border bg-surface-2 px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
                        />
                        <button
                            onClick={() => capMut.mutate(capInput ? parseFloat(capInput) : null)}
                            className="rounded bg-primary px-2 py-1 text-xs text-primary-foreground hover:bg-primary/90"
                        >
                            Set
                        </button>
                        {cs.gross_exposure_cap != null && (
                            <button
                                onClick={() => { capMut.mutate(null); setCapInput(""); }}
                                className="rounded border border-border px-2 py-1 text-xs hover:bg-surface-2"
                            >
                                Clear
                            </button>
                        )}
                    </div>
                </div>
            </div>

            {/* Blocked symbols inline display */}
            {cs.blocked_symbols.length > 0 && (
                <div className="mt-3 text-xs">
                    <span className="text-muted">Blocked: </span>
                    {cs.blocked_symbols.map((s) => (
                        <span key={s} className="inline-block rounded bg-red-900/30 px-1.5 py-0.5 mr-1 text-red-300 font-mono">{s}</span>
                    ))}
                </div>
            )}
        </Card>
    );
}
