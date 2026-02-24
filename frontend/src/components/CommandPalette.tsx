"use client";

import { useEffect, useState, useMemo, useCallback } from "react";
import { useRouter } from "next/navigation";
import { controlApi } from "@/lib/control";
import { useToasts } from "@/components/ui/ToastProvider";

interface Command {
    label: string;
    href?: string;
    action?: () => Promise<void>;
    icon?: string;
    category: string;
}

export function CommandPalette({ onClose }: { onClose: () => void }) {
    const router = useRouter();
    const { addToast } = useToasts();
    const [query, setQuery] = useState("");
    const [index, setIndex] = useState(0);
    const [executing, setExecuting] = useState(false);

    const staticCommands: Command[] = useMemo(() => [
        // Navigation
        { label: "Go to Execution", href: "/execution", icon: "📊", category: "Navigate" },
        { label: "Go to Orchestrator", href: "/orchestrator", icon: "⚙️", category: "Navigate" },
        { label: "Go to Portfolio", href: "/portfolio", icon: "💼", category: "Navigate" },
        { label: "Go to Research", href: "/research", icon: "🔬", category: "Navigate" },
        { label: "Go to Compare", href: "/compare", icon: "📈", category: "Navigate" },
        { label: "Go to Settings", href: "/settings", icon: "🔧", category: "Navigate" },
        // Actions
        {
            label: "Trigger Orchestrator Tick (Dry Run)",
            icon: "⚡",
            category: "Action",
            action: async () => {
                const r = await controlApi.triggerTick(true);
                addToast({ type: "success", title: "Tick Triggered", description: `Run ${r.run_id?.slice(0, 12)} — ${r.ran_jobs?.length ?? 0} jobs` });
            },
        },
        {
            label: "Flatten All Positions",
            icon: "🚨",
            category: "Action",
            action: async () => {
                await controlApi.flatten();
                addToast({ type: "success", title: "Flatten Submitted", description: "Flatten ALL order queued" });
            },
        },
        {
            label: "Pause Orchestrator",
            icon: "⏸",
            category: "Action",
            action: async () => {
                await controlApi.pause();
                addToast({ type: "info", title: "Orchestrator Paused" });
            },
        },
        {
            label: "Resume Orchestrator",
            icon: "▶️",
            category: "Action",
            action: async () => {
                await controlApi.resume();
                addToast({ type: "info", title: "Orchestrator Resumed" });
            },
        },
        {
            label: "Switch to NOOP Mode",
            icon: "🔇",
            category: "Action",
            action: async () => {
                await controlApi.setExecutionMode("noop");
                addToast({ type: "info", title: "Mode Changed", description: "Switched to NOOP" });
            },
        },
        {
            label: "Switch to Paper Mode",
            icon: "📝",
            category: "Action",
            action: async () => {
                await controlApi.setExecutionMode("paper");
                addToast({ type: "info", title: "Mode Changed", description: "Switched to PAPER" });
            },
        },
        {
            label: "Set Vol Regime: CRASH_RISK",
            icon: "🔴",
            category: "Action",
            action: async () => {
                await controlApi.setVolRegime("CRASH_RISK");
                addToast({ type: "warning", title: "Vol Override", description: "Regime set to CRASH_RISK" });
            },
        },
        {
            label: "Clear Vol Regime Override",
            icon: "🟢",
            category: "Action",
            action: async () => {
                await controlApi.setVolRegime(null);
                addToast({ type: "info", title: "Vol Override Cleared" });
            },
        },
    ], [addToast]);

    const options = useMemo(() => {
        let filtered = staticCommands;
        if (query) {
            filtered = filtered.filter((c) => c.label.toLowerCase().includes(query.toLowerCase()));
            if (query.length > 8 && query.includes("-")) {
                filtered.push({ href: `/runs/${query}`, label: `View Run: ${query}`, icon: "🔍", category: "Navigate" });
            }
        }
        return filtered;
    }, [query, staticCommands]);

    useEffect(() => {
        if (index >= options.length) setIndex(Math.max(0, options.length - 1));
    }, [options.length, index]);

    const execute = useCallback(async (opt: Command) => {
        if (opt.action) {
            setExecuting(true);
            try {
                await opt.action();
            } catch (e: any) {
                addToast({ type: "error", title: "Command Failed", description: e.message });
            } finally {
                setExecuting(false);
            }
            onClose();
        } else if (opt.href) {
            router.push(opt.href);
            onClose();
        }
    }, [addToast, onClose, router]);

    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if (e.key === "Escape") onClose();
            if (e.key === "ArrowDown") { e.preventDefault(); setIndex((i) => Math.min(i + 1, options.length - 1)); }
            if (e.key === "ArrowUp") { e.preventDefault(); setIndex((i) => Math.max(i - 1, 0)); }
            if (e.key === "Enter" && options.length > 0) { e.preventDefault(); execute(options[index]); }
        };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [onClose, options, index, execute]);

    // Group by category
    const categories = useMemo(() => {
        const cats: Record<string, Command[]> = {};
        options.forEach((opt) => {
            if (!cats[opt.category]) cats[opt.category] = [];
            cats[opt.category].push(opt);
        });
        return cats;
    }, [options]);

    let flatIdx = 0;

    return (
        <div className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh] bg-black/60 backdrop-blur-sm" onClick={onClose}>
            <div className="w-full max-w-2xl overflow-hidden rounded-lg border border-border bg-surface-1 shadow-2xl" onClick={(e) => e.stopPropagation()}>
                <div className="flex items-center border-b border-border px-4 font-mono">
                    <span className="text-secondary select-none text-lg mr-3">{">"}</span>
                    <input
                        autoFocus
                        className="flex-1 bg-transparent py-4 text-lg outline-none placeholder:text-secondary focus:ring-0"
                        placeholder="Search commands, runs, or tickers..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        disabled={executing}
                    />
                    {executing && <span className="text-xs text-primary animate-pulse">Running...</span>}
                    <kbd className="hidden sm:inline-block text-xs text-secondary border border-border px-2 py-0.5 rounded">ESC</kbd>
                </div>

                {options.length > 0 ? (
                    <div className="max-h-[60vh] overflow-y-auto p-2">
                        {Object.entries(categories).map(([cat, cmds]) => (
                            <div key={cat}>
                                <div className="px-2 pb-1 pt-2 text-xs font-semibold text-secondary uppercase tracking-wider">{cat}</div>
                                <div className="space-y-0.5">
                                    {cmds.map((opt) => {
                                        const myIdx = flatIdx++;
                                        return (
                                            <button
                                                key={(opt.href ?? "") + opt.label}
                                                className={`flex w-full items-center gap-3 rounded px-3 py-2.5 text-left text-sm transition-colors ${myIdx === index ? "bg-primary/20 text-primary border-l-2 border-primary" : "text-primary/70 border-l-2 border-transparent hover:bg-surface-2"}`}
                                                onClick={() => execute(opt)}
                                                onMouseEnter={() => setIndex(myIdx)}
                                            >
                                                <span className="text-base w-6 text-center">{opt.icon}</span>
                                                <span className="font-medium">{opt.label}</span>
                                                {opt.action && <span className="ml-auto text-[0.6rem] rounded bg-surface-2 px-1.5 py-0.5 text-muted">ACTION</span>}
                                                {myIdx === index && <span className="ml-auto text-xs text-secondary font-mono">↵</span>}
                                            </button>
                                        );
                                    })}
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="py-14 text-center text-sm text-secondary">
                        No results found.
                    </div>
                )}
            </div>
        </div>
    );
}
