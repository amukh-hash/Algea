"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";

import { api, LWPoint } from "@/lib/api";
import { useRunStream } from "@/lib/sse";
import dynamic from "next/dynamic";
import { EventsTimeline } from "@/components/events_timeline";
import type { SeriesSpec } from "@/components/TimeSeriesChartLW";
import type { Run } from "@/lib/types";

const TimeSeriesChartLW = dynamic(
  () => import("@/components/TimeSeriesChartLW").then((m) => m.TimeSeriesChartLW),
  { ssr: false, loading: () => <div className="h-36 animate-pulse rounded bg-slate-900" /> }
);



/* ── Family Run Card (multi-day curves) ─────────────────────────────────── */

function FamilyRunCard({ run }: { run: Run }) {
  const { events } = useRunStream(run.run_id);
  const { data: metricsData, isLoading } = useQuery({
    queryKey: ["metrics-lw-family", run.run_id],
    queryFn: () =>
      api.getMetricsLW(run.run_id, [
        "equity",
        "cash",
        "buying_power",
        "sleeve_capital.total",
        "sleeve_capital.core",
        "sleeve_capital.vrp",
        "sleeve_capital.selector",
        "sleeve.selector.intent_notional_sum",
        "sleeve.selector.intents_count",
        "sleeve.core.orders_count",
        "sleeve.vrp.orders_count",
        "sleeve.selector.orders_count",
        "sleeve.selector.num_longs",
        "sleeve.selector.num_shorts",
      ]),
  });

  const s = (key: string): LWPoint[] => metricsData?.series[key] ?? [];
  const hasPts = (key: string) => s(key).length > 0;

  const equitySeries: SeriesSpec[] = [
    ...(hasPts("equity") ? [{ key: "equity", name: "Equity", data: s("equity") }] : []),
    ...(hasPts("cash") ? [{ key: "cash", name: "Cash", data: s("cash"), color: "#22c55e" }] : []),
    ...(hasPts("buying_power") ? [{ key: "bp", name: "Buying Power", data: s("buying_power"), color: "#a855f7" }] : []),
  ];

  const capitalSeries: SeriesSpec[] = [
    ...(hasPts("sleeve_capital.total") ? [{ key: "total", name: "Total", data: s("sleeve_capital.total") }] : []),
    ...(hasPts("sleeve_capital.core") ? [{ key: "core", name: "Core", data: s("sleeve_capital.core"), color: "#f97316" }] : []),
    ...(hasPts("sleeve_capital.vrp") ? [{ key: "vrp", name: "VRP", data: s("sleeve_capital.vrp"), color: "#06b6d4" }] : []),
    ...(hasPts("sleeve_capital.selector") ? [{ key: "sel", name: "Selector", data: s("sleeve_capital.selector"), color: "#eab308" }] : []),
  ];

  const members = (run.meta as Record<string, unknown>)?.family_members;

  return (
    <div className="col-span-full rounded-xl border border-indigo-800/40 bg-gradient-to-br from-indigo-950/60 via-slate-900 to-slate-950 p-5 shadow-lg">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-slate-100">{run.name}</h2>
          <p className="text-xs text-slate-500">
            {run.run_id.slice(0, 12)} · {members ? `${members} daily reports` : "family run"} ·{" "}
            <span className="text-indigo-400">multi-day curves</span>
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className={`inline-block h-2.5 w-2.5 rounded-full ${run.status === "running" ? "bg-green-400 animate-pulse" :
            run.status === "completed" ? "bg-sky-400" : "bg-slate-600"
            }`} />
          <span className="text-xs text-slate-400">{run.status}</span>
          <Link href={`/runs/${run.run_id}`} className="text-xs text-sky-400 hover:underline">
            detail →
          </Link>
        </div>
      </div>




      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-48 animate-pulse rounded-lg bg-slate-800/50" />
          ))}
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {equitySeries.length > 0 && (
            <div>
              <TimeSeriesChartLW title="Account (Equity / Cash / BP)" series={equitySeries} height={180} />
            </div>
          )}
          {capitalSeries.length > 0 && (
            <div>
              <TimeSeriesChartLW title="Sleeve Capital Allocation" series={capitalSeries} height={180} />
            </div>
          )}
        </div>
      )}

      {events.length > 0 && (
        <div className="mt-4">
          <EventsTimeline events={events.slice(0, 10)} />
        </div>
      )}
    </div>
  );
}

/* ── Standard Sleeve Card (single run) ──────────────────────────────────── */

function SleeveCard({ runId, name }: { runId: string; name: string }) {
  const { metricsLW, events, status } = useRunStream(runId);
  const { data: initialMetrics } = useQuery({
    queryKey: ["metrics-lw", runId],
    queryFn: () =>
      api.getMetricsLW(runId, [
        "pnl_net",
        "equity",
        "gross_exposure",
        "net_exposure",
        "gate_scale",
        "risk_scale",
      ]),
  });

  const merge = (key: string): LWPoint[] => {
    const initial = initialMetrics?.series[key] ?? [];
    const live = metricsLW[key] ?? [];
    if (live.length === 0) return initial;
    const lastInitial = initial.length > 0 ? initial[initial.length - 1].time : 0;
    return [...initial, ...live.filter((p) => p.time > lastInitial)];
  };

  const pnlData = merge("pnl_net").length > 0 ? merge("pnl_net") : merge("equity");
  const pnlSeries: SeriesSpec[] = pnlData.length > 0 ? [{ key: "pnl", name: "PnL / Equity", data: pnlData }] : [];

  const exposureSeries: SeriesSpec[] = [
    ...(merge("gross_exposure").length > 0 ? [{ key: "gross_exposure", name: "Gross", data: merge("gross_exposure") }] : []),
    ...(merge("net_exposure").length > 0 ? [{ key: "net_exposure", name: "Net", data: merge("net_exposure"), color: "#22c55e" }] : []),
  ];

  const gateSeries: SeriesSpec[] = [
    ...(merge("gate_scale").length > 0 ? [{ key: "gate_scale", name: "Gate Scale", data: merge("gate_scale") }] : []),
    ...(merge("risk_scale").length > 0 ? [{ key: "risk_scale", name: "Risk Scale", data: merge("risk_scale"), color: "#f97316" }] : []),
  ];

  return (
    <div className="rounded-lg border border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 p-4 shadow-lg">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <div className="text-sm font-semibold text-slate-200">{name}</div>
          <div className="text-[10px] text-slate-500">{runId.slice(0, 12)}</div>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`inline-block h-2 w-2 rounded-full ${status === "running" ? "bg-green-400 animate-pulse" : status === "completed" ? "bg-sky-400" : "bg-slate-600"
              }`}
          />
          <span className="text-xs text-slate-400">{status}</span>
          <Link href={`/runs/${runId}`} className="ml-2 text-xs text-sky-400 hover:underline">
            detail →
          </Link>
        </div>
      </div>
      {pnlSeries.length > 0 && <TimeSeriesChartLW title="PnL / Equity" series={pnlSeries} height={160} />}
      {exposureSeries.length > 0 && (
        <div className="mt-2">
          <TimeSeriesChartLW title="Exposure" series={exposureSeries} height={120} />
        </div>
      )}
      {gateSeries.length > 0 && (
        <div className="mt-2">
          <TimeSeriesChartLW title="Gate / Risk Scale" series={gateSeries} height={120} />
        </div>
      )}
      {events.length > 0 && (
        <div className="mt-3">
          <EventsTimeline events={events.slice(0, 8)} />
        </div>
      )}
    </div>
  );
}

/* ── Page ────────────────────────────────────────────────────────────────── */

export default function ExecutionPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["sleeve-runs"],
    queryFn: () => api.listRuns("Sleeve", 20),
  });

  const allRuns = data?.items ?? [];
  const familyRuns = allRuns.filter((r) => (r.tags ?? []).includes("family"));
  const otherRuns = allRuns.filter((r) => !(r.tags ?? []).includes("family")).slice(0, 6);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-100">Trading Ops / Execution</h1>
        <p className="text-sm text-slate-500">Live and recent sleeve runs</p>
      </div>



      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-64 animate-pulse rounded-lg bg-slate-900" />
          ))}
        </div>
      ) : allRuns.length === 0 ? (
        <div className="rounded-lg border border-slate-800 bg-slate-900 p-8 text-center text-slate-500">
          No sleeve runs found. Start a paper/live sleeve to see data here.
        </div>
      ) : (
        <>
          {familyRuns.length > 0 && (
            <div className="grid gap-4">
              {familyRuns.map((run) => (
                <FamilyRunCard key={run.run_id} run={run} />
              ))}
            </div>
          )}

          {otherRuns.length > 0 && (
            <>
              <h2 className="text-lg font-semibold text-slate-300 mt-2">Individual Runs</h2>
              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                {otherRuns.map((run) => (
                  <SleeveCard key={run.run_id} runId={run.run_id} name={run.name} />
                ))}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
