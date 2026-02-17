"use client";

import { useSearchParams } from "next/navigation";
import { useQueries, useQuery } from "@tanstack/react-query";
import { api, LWPoint } from "@/lib/api";
import dynamic from "next/dynamic";
import { useState, Suspense } from "react";
import type { SeriesSpec } from "@/components/TimeSeriesChartLW";

const TimeSeriesChartLW = dynamic(
  () => import("@/components/TimeSeriesChartLW").then((m) => m.TimeSeriesChartLW),
  { ssr: false, loading: () => <div className="h-72 animate-pulse rounded bg-slate-900" /> }
);

const COMPARE_KEYS = ["pnl_net", "equity", "cum_net", "model_ic", "train_loss", "gross_exposure", "spearman_correlation"];

export default function ComparePage() {
  return (
    <Suspense fallback={<div className="h-72 animate-pulse rounded bg-slate-900" />}>
      <ComparePageInner />
    </Suspense>
  );
}

function ComparePageInner() {
  const params = useSearchParams();
  const runIds = (params.get("runIds") ?? "").split(",").filter(Boolean).slice(0, 5);
  const [selectedKey, setSelectedKey] = useState("pnl_net");

  // Fetch all data
  const metricsQueries = useQueries({
    queries: runIds.map((runId) => ({
      queryKey: ["compare-lw", runId],
      queryFn: () => api.getMetricsLW(runId, COMPARE_KEYS),
    })),
  });

  const runQueries = useQueries({
    queries: runIds.map((runId) => ({
      queryKey: ["run", runId],
      queryFn: () => api.getRun(runId),
    })),
  });

  // Build overlay series
  const overlaySeries: SeriesSpec[] = metricsQueries
    .map((q, idx) => {
      const data = q.data?.series[selectedKey] ?? [];
      if (data.length === 0) return null;
      const name = runQueries[idx]?.data?.name ?? runIds[idx].slice(0, 8);
      return { key: runIds[idx], name, data } satisfies SeriesSpec;
    })
    .filter(Boolean) as SeriesSpec[];

  // Find available keys across all runs
  const allAvailableKeys = new Set<string>();
  metricsQueries.forEach((q) => {
    if (q.data) {
      Object.keys(q.data.series).forEach((k) => {
        if (q.data!.series[k].length > 0) allAvailableKeys.add(k);
      });
    }
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-100">Compare Runs</h1>
        <p className="text-sm text-slate-500">
          {runIds.length > 0
            ? `Comparing ${runIds.length} runs`
            : "Add ?runIds=id1,id2 to URL to compare runs"}
        </p>
      </div>

      {runIds.length > 0 && (
        <>
          {/* Key selector */}
          <div className="flex flex-wrap gap-2">
            {Array.from(allAvailableKeys).map((key) => (
              <button
                key={key}
                onClick={() => setSelectedKey(key)}
                className={`rounded px-3 py-1.5 text-xs font-medium transition ${selectedKey === key
                  ? "bg-sky-500/20 text-sky-400 ring-1 ring-sky-500/30"
                  : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                  }`}
              >
                {key.replace(/_/g, " ")}
              </button>
            ))}
          </div>

          {/* Chart */}
          {overlaySeries.length > 0 ? (
            <TimeSeriesChartLW
              title={selectedKey.replace(/_/g, " ")}
              series={overlaySeries}
              height={360}
            />
          ) : (
            <div className="rounded-lg border border-slate-800 bg-slate-900 p-8 text-center text-slate-500">
              No data for key &quot;{selectedKey}&quot; in selected runs.
            </div>
          )}

          {/* Run list */}
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-slate-400">Selected Runs</h3>
            {runQueries.map((q, idx) => (
              <div
                key={runIds[idx]}
                className="flex items-center gap-3 rounded border border-slate-800 bg-slate-900/50 p-3"
              >
                <span
                  className="inline-block h-3 w-3 rounded-full"
                  style={{
                    backgroundColor: [
                      "#38bdf8", "#22c55e", "#f97316", "#a78bfa", "#f43f5e",
                    ][idx],
                  }}
                />
                <div>
                  <div className="text-sm text-slate-200">{q.data?.name ?? "Loading…"}</div>
                  <div className="text-[10px] text-slate-500">
                    {runIds[idx]} · {q.data?.status} · {q.data?.run_type}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
