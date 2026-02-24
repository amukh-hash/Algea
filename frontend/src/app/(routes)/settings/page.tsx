"use client";

import { PageHeader, Card, Button } from "@/components/ui/primitives";
import { useFeatureFlags } from "@/lib/feature_flags";
import { Suspense, useEffect, useState } from "react";
import { getCachedRuntimeConfig } from "@/runtime/config";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { controlApi, AuditEntry } from "@/lib/control";
import { useToasts } from "@/components/ui/ToastProvider";

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
  return (
    <Suspense fallback={<div className="h-32 animate-pulse rounded-md bg-surface-2" />}>
      <SettingsPageInner />
    </Suspense>
  );
}

function SettingsPageInner() {
  const flags = useFeatureFlags();
  const [status, setStatus] = useState<BackendStatus | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const runtime = getCachedRuntimeConfig();
  const { addToast } = useToasts();
  const qc = useQueryClient();

  // Control API data
  const controlState = useQuery({
    queryKey: ["control-state"],
    queryFn: controlApi.getState,
    refetchInterval: 5_000,
  });
  const config = useQuery({
    queryKey: ["orch-config"],
    queryFn: controlApi.getConfig,
    refetchInterval: 60_000,
  });
  const audit = useQuery({
    queryKey: ["control-audit"],
    queryFn: () => controlApi.getAudit(100),
    refetchInterval: 10_000,
  });
  const broker = useQuery({
    queryKey: ["broker-status"],
    queryFn: controlApi.getBrokerStatus,
    refetchInterval: 5_000,
  });

  const modeMut = useMutation({
    mutationFn: (mode: string) => controlApi.setExecutionMode(mode),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["control-state"] });
      addToast({ type: "success", title: "Mode Changed" });
    },
    onError: (e: Error) => addToast({ type: "error", title: "Mode Change Failed", description: e.message }),
  });

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
      <PageHeader title="Settings" subtitle="System controls, configuration, and diagnostics" actions={
        <div className="flex gap-2">
          <Button onClick={() => { audit.refetch(); controlState.refetch(); broker.refetch(); }}>Refresh</Button>
        </div>
      } />

      {/* Current Control State */}
      <Card>
        <h2 className="mb-3 text-sm font-semibold">Active Control State</h2>
        {controlState.data ? (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
            <div>
              <div className="text-xs text-muted">Execution Mode</div>
              <div className="flex items-center gap-2">
                <span className={`font-semibold ${controlState.data.execution_mode === "ibkr" ? "text-red-400" :
                  controlState.data.execution_mode === "paper" ? "text-amber-400" : "text-green-400"
                  }`}>
                  {controlState.data.execution_mode.toUpperCase()}
                </span>
                <select
                  value={controlState.data.execution_mode}
                  onChange={(e) => modeMut.mutate(e.target.value)}
                  className="rounded border border-border bg-surface-2 px-2 py-1 text-xs"
                >
                  <option value="noop">NOOP</option>
                  <option value="paper">PAPER</option>
                  <option value="ibkr">IBKR (Live)</option>
                </select>
              </div>
            </div>
            <div>
              <div className="text-xs text-muted">Paused</div>
              <div className={controlState.data.paused ? "text-amber-400 font-semibold" : "text-green-400"}>
                {controlState.data.paused ? "⏸ Paused" : "▶ Active"}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted">Vol Regime Override</div>
              <div className={controlState.data.vol_regime_override ? "text-red-400" : "text-muted"}>
                {controlState.data.vol_regime_override ?? "None"}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted">Broker</div>
              <div className={broker.data?.connected ? "text-green-400" : "text-red-400"}>
                {broker.data?.connected ? `Connected (${broker.data.mode})` : "Disconnected"}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted">Frozen Sleeves</div>
              <div className={controlState.data.frozen_sleeves.length > 0 ? "text-amber-400" : "text-muted"}>
                {controlState.data.frozen_sleeves.length > 0 ? controlState.data.frozen_sleeves.join(", ") : "None"}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted">Exposure Cap</div>
              <div className="text-muted">
                {controlState.data.gross_exposure_cap ?? "None"}
              </div>
            </div>
          </div>
        ) : (
          <p className="text-sm text-secondary">Loading control state...</p>
        )}
      </Card>

      {/* Orchestrator Configuration */}
      <Card>
        <h2 className="mb-3 text-sm font-semibold">Orchestrator Configuration</h2>
        {config.data ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            {Object.entries(config.data).map(([k, v]) => (
              <div key={k}>
                <div className="text-muted">{k.replace(/_/g, " ")}</div>
                <div className="font-mono">{Array.isArray(v) ? v.join(", ") || "[]" : String(v)}</div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-secondary">Loading config...</p>
        )}
      </Card>

      {/* Audit Log */}
      <Card>
        <h2 className="mb-3 text-sm font-semibold">Audit Log (Recent Actions)</h2>
        {audit.data && audit.data.items.length > 0 ? (
          <div className="max-h-96 overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-surface-1">
                <tr className="border-b border-border text-muted text-left">
                  <th className="pb-2 pr-4">Timestamp</th>
                  <th className="pb-2 pr-4">Action</th>
                  <th className="pb-2">Detail</th>
                </tr>
              </thead>
              <tbody>
                {audit.data.items.map((entry: AuditEntry, i: number) => (
                  <tr key={i} className="border-b border-border/30">
                    <td className="py-1.5 pr-4 text-muted whitespace-nowrap">{new Date(entry.ts).toLocaleString()}</td>
                    <td className="py-1.5 pr-4 font-medium">{entry.action}</td>
                    <td className="py-1.5 text-muted font-mono truncate max-w-[300px]" title={JSON.stringify(entry.detail)}>
                      {Object.keys(entry.detail).length > 0 ? JSON.stringify(entry.detail) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-sm text-secondary">No audit entries yet</p>
        )}
      </Card>

      {/* Feature Flags */}
      <Card>
        <h2 className="mb-2 text-sm font-semibold">Feature Flags</h2>
        <ul className="space-y-2 text-sm">
          {Object.entries(flags).map(([k, v]) => <li key={k}><strong>{k}</strong>: {String(v)}</li>)}
        </ul>
      </Card>

      {/* Desktop Backend Diagnostics */}
      <Card>
        <h2 className="mb-2 text-sm font-semibold">Desktop Backend</h2>
        <p className="text-sm text-secondary">Mode: {runtime?.desktop ? "desktop" : "web"}</p>
        <p className="text-sm text-secondary">API base: {runtime?.apiBaseUrl ?? "unresolved"}</p>
        <p className="text-sm">Status: {status?.state ?? "n/a"}, PID: {status?.pid ?? "n/a"}, Port: {status?.port ?? "n/a"}</p>
        <p className="text-sm text-secondary">Last health: {status?.last_health_ok_at ? new Date(status.last_health_ok_at * 1000).toLocaleString() : "n/a"}</p>
        <div className="mt-2 flex gap-2">
          <Button onClick={async () => { await invoke("restart_backend"); await refreshDiagnostics(); }}>Restart backend</Button>
          <Button onClick={refreshDiagnostics}>Refresh diagnostics</Button>
        </div>
        <pre className="mt-3 max-h-64 overflow-auto rounded bg-surface-2 p-2 text-xs">{logs.slice(-200).join("\n") || "No logs"}</pre>
      </Card>
    </div>
  );
}
