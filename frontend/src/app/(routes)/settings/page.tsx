"use client";

import { PageHeader, Card, Button } from "@/components/ui/primitives";
import { useFeatureFlags } from "@/lib/feature_flags";
import { useEffect, useState } from "react";
import { getCachedRuntimeConfig } from "@/runtime/config";

type BackendStatus = { state: string; pid?: number; port?: number; last_health_ok_at?: number };

declare global {
  interface Window {
    __TAURI__?: {
      core?: { invoke?: (cmd: string, args?: Record<string, unknown>) => Promise<unknown> };
      invoke?: (cmd: string, args?: Record<string, unknown>) => Promise<unknown>;
    };
  }
}

const invoke = async <T,>(cmd: string): Promise<T> => {
  const fn = window.__TAURI__?.core?.invoke ?? window.__TAURI__?.invoke;
  if (!fn) throw new Error("Tauri invoke unavailable");
  return (await fn(cmd)) as T;
};

export default function SettingsPage() {
  const flags = useFeatureFlags();
  const [status, setStatus] = useState<BackendStatus | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const runtime = getCachedRuntimeConfig();

  async function refreshDiagnostics() {
    try {
      setStatus(await invoke<BackendStatus>("get_backend_status"));
      setLogs(await invoke<string[]>("get_backend_logs"));
    } catch {
      setStatus(null);
      setLogs([]);
    }
  }

  useEffect(() => {
    refreshDiagnostics();
  }, []);

  return (
    <div className="space-y-4">
      <PageHeader title="Settings" subtitle="Feature flags and desktop diagnostics" actions={<Button onClick={refreshDiagnostics}>Refresh</Button>} />
      <Card>
        <h2 className="mb-2 text-sm font-semibold">Feature Flags</h2>
        <ul className="space-y-2 text-sm">
          {Object.entries(flags).map(([k, v]) => <li key={k}><strong>{k}</strong>: {String(v)}</li>)}
        </ul>
      </Card>
      <Card>
        <h2 className="mb-2 text-sm font-semibold">Desktop Backend</h2>
        <p className="text-sm text-secondary">Mode: {runtime?.desktop ? "desktop" : "web"}</p>
        <p className="text-sm text-secondary">API base: {runtime?.apiBaseUrl ?? "unresolved"}</p>
        <p className="text-sm">Status: {status?.state ?? "n/a"}, PID: {status?.pid ?? "n/a"}, Port: {status?.port ?? "n/a"}</p>
        <p className="text-sm text-secondary">Last health: {status?.last_health_ok_at ? new Date(status.last_health_ok_at * 1000).toLocaleString() : "n/a"}</p>
        <div className="mt-2 flex gap-2"><Button onClick={async () => { await invoke("restart_backend"); await refreshDiagnostics(); }}>Restart backend</Button></div>
        <pre className="mt-3 max-h-64 overflow-auto rounded bg-surface-2 p-2 text-xs">{logs.slice(-200).join("\n") || "No logs"}</pre>
      </Card>
    </div>
  );
}
