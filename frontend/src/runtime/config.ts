export type RuntimeConfig = { apiBaseUrl: string; desktop: boolean };

declare global {
  interface Window {
    __TAURI__?: {
      core?: { invoke?: (cmd: string, args?: Record<string, unknown>) => Promise<unknown> };
      invoke?: (cmd: string, args?: Record<string, unknown>) => Promise<unknown>;
    };
  }
}

let cached: RuntimeConfig | null = null;

async function invoke<T>(cmd: string): Promise<T> {
  const fn = window.__TAURI__?.core?.invoke ?? window.__TAURI__?.invoke;
  if (!fn) throw new Error("tauri invoke unavailable");
  return (await fn(cmd)) as T;
}

export async function resolveRuntimeConfig(): Promise<RuntimeConfig> {
  if (cached) return cached;
  const webBase = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
  if (typeof window === "undefined") {
    cached = { apiBaseUrl: webBase, desktop: false };
    return cached;
  }
  try {
    const url = await invoke<string>("get_backend_base_url");
    cached = { apiBaseUrl: url, desktop: true };
  } catch {
    cached = { apiBaseUrl: webBase, desktop: false };
  }
  return cached;
}

export function getCachedRuntimeConfig() {
  return cached;
}
