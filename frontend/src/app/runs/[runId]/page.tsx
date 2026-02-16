"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { MetricChart } from "@/components/metric_chart";
import { EventsTimeline } from "@/components/events_timeline";
import { ArtifactViewer } from "@/components/artifact_viewer";

export default function RunDetailPage({ params }: { params: { runId: string } }) {
  const runId = params.runId;
  const run = useQuery({ queryKey: ["run", runId], queryFn: () => api.getRun(runId) });
  const metrics = useQuery({ queryKey: ["metrics", runId], queryFn: () => api.getMetrics(runId, ["pnl_net", "cum_net", "train_loss", "val_loss"]) });
  const events = useQuery({ queryKey: ["events", runId], queryFn: () => api.getEvents(runId) });
  const artifacts = useQuery({ queryKey: ["artifacts", runId], queryFn: () => api.listArtifacts(runId) });
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold">Run {run.data?.name}</h1>
      <div className="rounded border border-slate-800 p-3 text-sm">status: {run.data?.status} · type: {run.data?.run_type}</div>
      <MetricChart data={metrics.data?.series.pnl_net ?? metrics.data?.series.cum_net ?? []} />
      <EventsTimeline events={events.data?.items ?? []} />
      <ArtifactViewer runId={runId} artifacts={artifacts.data?.items ?? []} />
    </div>
  );
}
