"use client";

import { useQuery } from "@tanstack/react-query";
import { api, LWPoint } from "@/lib/api";
import dynamic from "next/dynamic";
import { EventsTimeline } from "@/components/events_timeline";
import { ArtifactViewer } from "@/components/artifact_viewer";
import { useState } from "react";
import type { SeriesSpec } from "@/components/TimeSeriesChartLW";
import Link from "next/link";

const TimeSeriesChartLW = dynamic(
  () => import("@/components/TimeSeriesChartLW").then((m) => m.TimeSeriesChartLW),
  { ssr: false, loading: () => <div className="h-52 animate-pulse rounded bg-slate-900" /> }
);

// All canonical metric keys we check for
const ALL_METRIC_KEYS = [
  "pnl_net", "equity", "cum_net",
  "gross_exposure", "net_exposure",
  "gate_scale", "risk_scale",
  "train_loss", "val_loss",
  "model_ic", "baseline_ic", "all_passed",
  "spearman_correlation", "spearman_pvalue",
  "account_equity", "account_cash",
  "capital_core", "capital_vrp", "capital_selector",
  "dd_current", "vol_realized", "turnover",
];

export default function RunDetailPage({ params }: { params: { runId: string } }) {
  const runId = params.runId;
  const run = useQuery({ queryKey: ["run", runId], queryFn: () => api.getRun(runId) });
  const metricsQ = useQuery({
    queryKey: ["metrics-lw-detail", runId],
    queryFn: () => api.getMetricsLW(runId, ALL_METRIC_KEYS),
  });
  const events = useQuery({ queryKey: ["events", runId], queryFn: () => api.getEvents(runId) });
  const artifacts = useQuery({ queryKey: ["artifacts", runId], queryFn: () => api.listArtifacts(runId) });

  const [activeTab, setActiveTab] = useState<"metrics" | "events" | "artifacts">("metrics");

  const series = metricsQ.data?.series ?? {};
  const availableKeys = Object.keys(series).filter((k) => series[k].length > 0);

  // Group metrics into chart panels
  const pnlKeys = availableKeys.filter((k) => ["pnl_net", "equity", "cum_net"].includes(k));
  const exposureKeys = availableKeys.filter((k) => ["gross_exposure", "net_exposure"].includes(k));
  const gateKeys = availableKeys.filter((k) => ["gate_scale", "risk_scale"].includes(k));
  const trainKeys = availableKeys.filter((k) => ["train_loss", "val_loss"].includes(k));
  const otherKeys = availableKeys.filter(
    (k) => ![...pnlKeys, ...exposureKeys, ...gateKeys, ...trainKeys].includes(k)
  );

  const toSeriesSpecs = (keys: string[]): SeriesSpec[] =>
    keys.map((k) => ({ key: k, name: k.replace(/_/g, " "), data: series[k] }));

  const statusColor =
    run.data?.status === "completed" ? "text-green-400" :
      run.data?.status === "running" ? "text-sky-400" :
        run.data?.status === "error" ? "text-red-400" : "text-slate-400";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <Link href="/research" className="text-xs text-slate-500 hover:text-slate-300">
            ← All Runs
          </Link>
          <h1 className="mt-1 text-2xl font-bold text-slate-100">
            {run.data?.name ?? "Loading…"}
          </h1>
          <div className="mt-1 flex gap-3 text-xs text-slate-500">
            <span className={statusColor}>{run.data?.status}</span>
            <span>{run.data?.run_type}</span>
            {run.data?.started_at && (
              <span>{new Date(run.data.started_at).toLocaleString()}</span>
            )}
          </div>
          {run.data?.tags && run.data.tags.length > 0 && (
            <div className="mt-2 flex gap-1">
              {run.data.tags.map((t) => (
                <span key={t} className="rounded bg-slate-800 px-2 py-0.5 text-[10px] text-slate-400">
                  {t}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-slate-800 pb-1">
        {(["metrics", "events", "artifacts"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`rounded-t px-4 py-2 text-sm font-medium transition ${activeTab === tab
                ? "bg-slate-800 text-slate-100"
                : "text-slate-500 hover:bg-slate-900 hover:text-slate-300"
              }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Metrics Tab */}
      {activeTab === "metrics" && (
        <div className="space-y-4">
          {metricsQ.isLoading && (
            <div className="h-52 animate-pulse rounded bg-slate-900" />
          )}
          {availableKeys.length === 0 && !metricsQ.isLoading && (
            <div className="rounded-lg border border-slate-800 bg-slate-900 p-6 text-center text-sm text-slate-500">
              No metrics recorded for this run.
            </div>
          )}
          {pnlKeys.length > 0 && (
            <TimeSeriesChartLW title="PnL / Equity" series={toSeriesSpecs(pnlKeys)} height={240} />
          )}
          {exposureKeys.length > 0 && (
            <TimeSeriesChartLW title="Exposure" series={toSeriesSpecs(exposureKeys)} height={180} />
          )}
          {gateKeys.length > 0 && (
            <TimeSeriesChartLW title="Gate / Risk Scale" series={toSeriesSpecs(gateKeys)} height={180} />
          )}
          {trainKeys.length > 0 && (
            <TimeSeriesChartLW title="Training Loss" series={toSeriesSpecs(trainKeys)} height={200} />
          )}
          {otherKeys.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-sm font-medium text-slate-400">Other Metrics</h3>
              <div className="grid gap-3 md:grid-cols-2">
                {otherKeys.map((k) => (
                  <div key={k} className="rounded border border-slate-800 bg-slate-900/50 p-3">
                    <span className="text-xs text-slate-500">{k.replace(/_/g, " ")}</span>
                    <div className="mt-1 text-lg font-semibold text-slate-200">
                      {series[k][series[k].length - 1]?.value?.toLocaleString(undefined, {
                        maximumFractionDigits: 4,
                      }) ?? "—"}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Events Tab */}
      {activeTab === "events" && (
        <EventsTimeline events={events.data?.items ?? []} />
      )}

      {/* Artifacts Tab */}
      {activeTab === "artifacts" && (
        <ArtifactViewer runId={runId} artifacts={artifacts.data?.items ?? []} />
      )}
    </div>
  );
}
