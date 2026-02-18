"use client";

import { ReactNode } from "react";

interface ErrorBannerProps {
    error: Error | null;
    onRetry?: () => void;
    isRetrying?: boolean;
    children?: ReactNode;
}

/**
 * Reusable error banner with retry action.
 * Shows a clear error message and a retry button for failed queries.
 */
export function ErrorBanner({ error, onRetry, isRetrying, children }: ErrorBannerProps) {
    if (!error) return null;

    const isTimeout = error.name === "TimeoutError";
    const message = isTimeout
        ? "The server took too long to respond. It may be overloaded or unreachable."
        : error.message || "An unexpected error occurred.";

    return (
        <div className="rounded-lg border border-red-500/30 bg-red-950/20 px-4 py-3">
            <div className="flex items-start justify-between gap-3">
                <div className="flex items-start gap-2 text-sm">
                    <span className="mt-0.5 text-red-400">⚠</span>
                    <div>
                        <p className="font-medium text-red-300">
                            {isTimeout ? "Connection Timeout" : "Error"}
                        </p>
                        <p className="mt-0.5 text-red-400/80">{message}</p>
                    </div>
                </div>
                {onRetry && (
                    <button
                        onClick={onRetry}
                        disabled={isRetrying}
                        className="shrink-0 rounded border border-red-500/40 bg-red-950/40 px-3 py-1.5 text-xs font-medium text-red-300 transition-colors hover:bg-red-900/40 disabled:opacity-50"
                    >
                        {isRetrying ? "Retrying…" : "Retry"}
                    </button>
                )}
            </div>
            {children}
        </div>
    );
}

/**
 * Inline connection status indicator for SSE streams.
 */
export function ConnectionStatus({ state }: { state: string }) {
    const config: Record<string, { color: string; label: string }> = {
        connecting: { color: "text-yellow-400", label: "Connecting…" },
        open: { color: "text-green-400", label: "Live" },
        reconnecting: { color: "text-amber-400", label: "Reconnecting…" },
        rehydrating: { color: "text-blue-400", label: "Rehydrating…" },
        closed: { color: "text-gray-500", label: "Disconnected" },
        error: { color: "text-red-400", label: "Connection Error" },
    };
    const { color, label } = config[state] ?? { color: "text-gray-500", label: state };

    return (
        <span className={`inline-flex items-center gap-1.5 text-xs ${color}`}>
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-current" />
            {label}
        </span>
    );
}
