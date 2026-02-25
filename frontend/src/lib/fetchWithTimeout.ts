/**
 * Fetch wrapper that uses AbortController with configurable timeout.
 * Prevents indefinite hangs from unresponsive backends.
 */

import { globalToastBus } from "@/components/ui/ToastProvider";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export class TimeoutError extends Error {
    constructor(url: string, timeoutMs: number) {
        super(`Request to ${url} timed out after ${timeoutMs}ms`);
        this.name = "TimeoutError";
    }
}

export class ApiError extends Error {
    status: number;
    body: string;
    detail: unknown;

    constructor(url: string, status: number, body: string) {
        super(`API ${status}: ${body}`);
        this.name = "ApiError";
        this.status = status;
        this.body = body;
        let detail: unknown = null;
        try {
            const parsed = JSON.parse(body);
            detail = parsed?.detail ?? parsed;
        } catch {
            detail = null;
        }
        this.detail = detail;
    }
}

interface FetchOptions {
    timeout?: number;
    init?: RequestInit;
}

const DEFAULT_TIMEOUT = 20_000;

// Throttle repeated timeout/error toasts — show at most once per 30s per category
let _lastTimeoutToast = 0;
let _lastCongestedToast = 0;
const TOAST_THROTTLE_MS = 30_000;

/**
 * Fetch with an AbortController-based timeout.
 * Throws `TimeoutError` on timeout, `ApiError` on non-2xx responses.
 */
export async function fetchWithTimeout(
    url: string,
    opts: FetchOptions = {},
): Promise<Response> {
    const { timeout = DEFAULT_TIMEOUT, init = {} } = opts;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);

    try {
        const response = await fetch(url, {
            ...init,
            signal: controller.signal,
            cache: "no-store",
        });
        if (!response.ok) {
            const body = await response.text().catch(() => "");
            throw new ApiError(url, response.status, body);
        }
        return response;
    } catch (err) {
        if (err instanceof ApiError) {
            if (err.status === 504 || err.status === 503) {
                const now = Date.now();
                if (now - _lastCongestedToast > TOAST_THROTTLE_MS) {
                    _lastCongestedToast = now;
                    globalToastBus.addToast({
                        type: "error",
                        title: "Backend Congested",
                        description: `Service responded with ${err.status}.`,
                    });
                }
            }
            throw err;
        }
        if ((err as Error).name === "AbortError") {
            const now = Date.now();
            if (now - _lastTimeoutToast > TOAST_THROTTLE_MS) {
                _lastTimeoutToast = now;
                globalToastBus.addToast({
                    type: "error",
                    title: "Timeout",
                    description: `Request to backend timed out after ${timeout}ms.`,
                });
            }
            throw new TimeoutError(url, timeout);
        }
        throw err;
    } finally {
        clearTimeout(timer);
    }
}

/**
 * Typed JSON fetch with timeout. Convenience wrapper used by api.ts and orch.ts.
 */
export async function fetchJSON<T>(
    path: string,
    opts: FetchOptions = {},
): Promise<T> {
    const response = await fetchWithTimeout(`${API_BASE}${path}`, opts);
    return response.json() as Promise<T>;
}
