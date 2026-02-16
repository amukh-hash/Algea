"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useRunStream } from "@/lib/sse";
import { MetricChart } from "@/components/metric_chart";
import { EventsTimeline } from "@/components/events_timeline";

function SleeveCard({ runId }: { runId: string }) {
  const { metrics, events, status } = useRunStream(runId);
  return (
    <div className="rounded border border-slate-800 bg-slate-900 p-3">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-sm font-semibold">{runId.slice(0, 8)} · {status}</div>
        <Link href={`/runs/${runId}`} className="text-xs underline">open detail</Link>
      </div>
      <MetricChart data={metrics.pnl_net ?? []} />
      <MetricChart data={metrics.gross_exposure ?? []} color="#facc15" />
      <EventsTimeline events={events.slice(0, 8)} />
    </div>
  );
}

export default function ExecutionPage() {
  const { data } = useQuery({ queryKey: ["sleeve-runs"], queryFn: () => api.listRuns("Sleeve") });
  const runIds = (data?.items ?? []).slice(0, 3).map((run) => run.run_id);
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold">Trading Ops / Execution</h1>
      <div className="grid gap-3 md:grid-cols-3">
        {runIds.map((runId) => <SleeveCard key={runId} runId={runId} />)}
      </div>
    </div>
  );
}
