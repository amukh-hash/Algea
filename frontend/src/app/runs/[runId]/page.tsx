"use client";

import { useParams } from "next/navigation";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { ArtifactViewer } from "@/components/artifact_viewer";
import { EventsTimeline } from "@/components/events_timeline";
import dynamic from "next/dynamic";
import { Button, Card, EmptyState, PageHeader, StatusBadge } from "@/components/ui/primitives";
import { Tabs } from "@/components/ui/tabs";

const TimeSeriesChartLW = dynamic(() => import("@/components/TimeSeriesChartLW").then((m) => m.TimeSeriesChartLW), { ssr: false });

export default function RunDetailPage() {
  const params = useParams<{ runId: string }>();
  const runId = params.runId;
  const [alignment, setAlignment] = useState<"absolute" | "relative">("absolute");
  const [normalize, setNormalize] = useState<"raw" | "index100">("raw");

  const run = useQuery({ queryKey: ["run", runId], queryFn: () => api.getRun(runId), staleTime: 300_000 });
  const metrics = useQuery({ queryKey: ["run-metrics", runId], queryFn: () => api.getMetricsLW(runId, ["pnl_net", "equity", "gross_exposure", "net_exposure", "risk_scale", "gate_scale", "train_loss"]), staleTime: 30_000 });
  const events = useQuery({ queryKey: ["run-events", runId], queryFn: () => api.getEvents(runId), staleTime: 30_000 });
  const artifacts = useQuery({ queryKey: ["run-artifacts", runId], queryFn: () => api.listArtifacts(runId), staleTime: 600_000 });

  const chartSeries = useMemo(() => Object.entries(metrics.data?.series ?? {}).filter(([, v]) => v.length > 0).map(([key, data]) => ({ key, name: key, data })), [metrics.data]);

  return (
    <div className="space-y-4">
      <PageHeader title={run.data?.name ?? "Run detail"} subtitle={runId} actions={<div className="flex gap-2"><Button onClick={() => navigator.clipboard.writeText(runId)}>Copy ID</Button><Button onClick={() => navigator.clipboard.writeText(window.location.href)}>Share link</Button></div>} />
      {run.data && <Card className="flex flex-wrap gap-3 text-sm"><StatusBadge status={run.data.status} /><span>{run.data.run_type}</span><span>{new Date(run.data.started_at).toLocaleString()}</span></Card>}
      <Tabs
        items={[
          { id: "metrics", label: "Metrics", panel: (
            <div className="space-y-3">
              <Card className="flex flex-wrap gap-2"><Button onClick={() => setAlignment((a) => a === "absolute" ? "relative" : "absolute")}>Alignment: {alignment}</Button><Button onClick={() => setNormalize((n) => n === "raw" ? "index100" : "raw")}>Normalize: {normalize}</Button></Card>
              {chartSeries.length === 0 ? <EmptyState title="No metrics yet" message="Run may not emit metrics yet." /> : <TimeSeriesChartLW title="Metrics" series={chartSeries} mode="full" alignment={alignment} normalize={normalize} />}
            </div>
          ) },
          { id: "events", label: "Events", panel: <EventsTimeline events={events.data?.items ?? []} /> },
          { id: "artifacts", label: "Artifacts", panel: <ArtifactViewer runId={runId} artifacts={artifacts.data?.items ?? []} /> },
        ]}
      />
    </div>
  );
}
