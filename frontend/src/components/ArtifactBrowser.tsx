"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { orchApi } from "@/lib/orch";

const KNOWN_DIRS = ["signals", "targets", "orders", "fills", "reports", "runs", "jobs"];

export function ArtifactBrowser() {
    const today = new Date().toISOString().split("T")[0];
    const [day, setDay] = useState(today);
    const [selectedPath, setSelectedPath] = useState<string | null>(null);

    const artifact = useQuery({
        queryKey: ["artifact-file", day, selectedPath],
        queryFn: async () => {
            if (!selectedPath) return null;
            const resp = await fetch(
                `${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/api/orchestrator/artifacts/${day}/${selectedPath}`
            );
            if (!resp.ok) return { error: `HTTP ${resp.status}` };
            const text = await resp.text();
            try { return JSON.parse(text); } catch { return text; }
        },
        enabled: !!selectedPath,
        staleTime: 30_000,
    });

    return (
        <div className="grid grid-cols-[200px_1fr] gap-4 min-h-[300px]">
            {/* Tree sidebar */}
            <div className="space-y-2">
                <input
                    type="date"
                    value={day}
                    onChange={(e) => { setDay(e.target.value); setSelectedPath(null); }}
                    className="w-full rounded border border-border bg-surface-2 px-2 py-1 text-xs"
                />
                <div className="space-y-0.5">
                    {KNOWN_DIRS.map((dir) => (
                        <TreeDir key={dir} dir={dir} day={day} onSelect={setSelectedPath} selectedPath={selectedPath} />
                    ))}
                    <button
                        onClick={() => setSelectedPath("heartbeat.json")}
                        className={`block w-full text-left text-xs px-2 py-1 rounded hover:bg-surface-2 ${selectedPath === "heartbeat.json" ? "bg-surface-2 text-primary" : "text-secondary"
                            }`}
                    >
                        📄 heartbeat.json
                    </button>
                </div>
            </div>

            {/* Content viewer */}
            <div className="overflow-auto rounded border border-border bg-surface-2/30 p-3">
                {!selectedPath && <p className="text-sm text-secondary">Select a file to view</p>}
                {selectedPath && artifact.isLoading && <p className="text-sm text-muted animate-pulse">Loading...</p>}
                {selectedPath && artifact.data && (
                    <pre className="text-xs font-mono whitespace-pre-wrap break-all text-secondary">
                        {typeof artifact.data === "string" ? artifact.data : JSON.stringify(artifact.data, null, 2)}
                    </pre>
                )}
                {selectedPath && artifact.error && (
                    <p className="text-sm text-red-400">Failed to load artifact</p>
                )}
            </div>
        </div>
    );
}

function TreeDir({ dir, day, onSelect, selectedPath }: { dir: string; day: string; onSelect: (p: string) => void; selectedPath: string | null }) {
    const [open, setOpen] = useState(false);
    const KNOWN_FILES: Record<string, string[]> = {
        signals: ["core_signals.json", "vrp_signals.json", "selector_signals.json", "data_refresh.json"],
        targets: ["core_targets.json", "vrp_targets.json", "selector_targets.json"],
        orders: ["orders.json", "routed.json", "rejected.json"],
        fills: ["fills.json", "positions.json"],
        reports: ["risk_checks.json", "eod_summary.json"],
        runs: [],
        jobs: [],
    };
    const files = KNOWN_FILES[dir] ?? [];
    return (
        <div>
            <button onClick={() => setOpen((v) => !v)} className="flex items-center gap-1 text-xs text-secondary hover:text-primary w-full px-2 py-1 rounded hover:bg-surface-2">
                <span>{open ? "📂" : "📁"}</span>
                <span>{dir}/</span>
            </button>
            {open && files.map((f) => (
                <button
                    key={f}
                    onClick={() => onSelect(`${dir}/${f}`)}
                    className={`block w-full text-left text-xs pl-6 py-0.5 rounded hover:bg-surface-2 ${selectedPath === `${dir}/${f}` ? "bg-surface-2 text-primary" : "text-muted"
                        }`}
                >
                    📄 {f}
                </button>
            ))}
        </div>
    );
}
