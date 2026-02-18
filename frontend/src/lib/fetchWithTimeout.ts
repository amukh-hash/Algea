/**
 * Fetch wrapper that uses AbortController with configurable timeout.
 * Prevents indefinite hangs from unresponsive backends.
 */

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

    constructor(url: string, status: number, body: string) {
        super(`API ${status}: ${body}`);
        this.name = "ApiError";
        this.status = status;
        this.body = body;
    }
}

interface FetchOptions {
    timeout?: number;
    init?: RequestInit;
}

const DEFAULT_TIMEOUT = 8_000;

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
        if (err instanceof ApiError) throw err;
        if ((err as Error).name === "AbortError") {
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
