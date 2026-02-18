"use client";

import Link from "next/link";
import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { auditLog } from "@/lib/audit";
import { EventsTimeline } from "@/components/events_timeline";
import { ArtifactViewer } from "@/components/artifact_viewer";
import { Button, EmptyState, PageHeader, StatusBadge, Tabs } from "@/components/ui/primitives";

const Chart = dynamic(() => import("@/components/TimeSeriesChartLW").then((m) => m.TimeSeriesChartLW), { ssr: false });
const ALL_KEYS = ["pnl_net", "equity", "cum_net", "gross_exposure", "net_exposure", "gate_scale", "risk_scale", "train_loss", "val_loss"];

export default function RunDetailPage({ params }: { params: { runId: string } }) {
  const runId = params.runId;
  const [activeTab, setActiveTab] = useState("metrics");
  const [align, setAlign] = useState<"absolute" | "relative">("absolute");
  const [normalize, setNormalize] = useState<"raw" | "index100">("raw");
  const [crosshair, setCrosshair] = useState<number | null>(null);
  const [eventLevel, setEventLevel] = useState<"all" | "info" | "warn" | "error">("all");

  const run = useQuery({ queryKey: ["run", runId], queryFn: () => api.getRun(runId), staleTime: 300_000 });
  const metrics = useQuery({ queryKey: ["metrics", runId], queryFn: () => api.getMetricsLW(runId, ALL_KEYS), staleTime: 60_000 });
  const events = useQuery({ queryKey: ["events", runId], queryFn: () => api.getEvents(runId) });
  const artifacts = useQuery({ queryKey: ["artifacts", runId], queryFn: () => api.listArtifacts(runId), staleTime: 600_000 });

  const series = metrics.data?.series ?? {};
  const anns = useMemo(() => (events.data?.items ?? []).filter((e) => ["warn", "error"].includes(e.level)).slice(0, 20).map((e) => ({ time: Math.floor(new Date(e.ts).getTime() / 1000), label: e.type, kind: e.level as "warn" | "error" })), [events.data?.items]);

  return (
    <div className="space-y-4">
      <PageHeader title={run.data?.name ?? "Run detail"} subtitle={`${runId.slice(0, 12)} · ${run.data?.run_type ?? ""}`} actions={<><Button onClick={() => (auditLog("copy_run_id", { runId }), navigator.clipboard.writeText(runId))}>Copy run ID</Button><Button onClick={() => (auditLog("share_link"), navigator.clipboard.writeText(window.location.href))}>Share link</Button><Button onClick={() => (auditLog("export_run", { runId }), navigator.clipboard.writeText("export"))}>Export</Button></>} />
      <div className="text-sm"><Link className="text-info" href="/research">← Back to research</Link> <StatusBadge status={run.data?.status ?? "unknown"} /></div>
      <Tabs active={activeTab} onChange={setActiveTab} items={[
        {
          id: "metrics", label: "Metrics", panel: (
            <div className="space-y-3">
              <div className="flex flex-wrap gap-2"><Button onClick={() => setAlign((v) => v === "absolute" ? "relative" : "absolute")}>Alignment: {align}</Button><Button onClick={() => setNormalize((v) => v === "raw" ? "index100" : "raw")}>Normalize: {normalize}</Button></div>
              {!Object.values(series).some((s) => s.length) && <EmptyState title="No metrics" message="Run may not emit metrics yet." />}
              <Chart title="PnL / Equity" series={[{ key: "pnl", name: "PnL", data: series.pnl_net ?? [] }, { key: "equity", name: "Equity", data: series.equity ?? [] }]} alignment={align} normalize={normalize} annotations={anns} onCrosshairMove={setCrosshair} externalCrosshairTime={crosshair} />
              <Chart title="Exposure" series={[{ key: "gross", name: "Gross", data: series.gross_exposure ?? [] }, { key: "net", name: "Net", data: series.net_exposure ?? [] }]} alignment={align} normalize={normalize} onCrosshairMove={setCrosshair} externalCrosshairTime={crosshair} />
            </div>
          )
        },
        {
          id: "events", label: "Events", panel: (
            <div className="space-y-2">
              <div className="flex gap-2"><Button onClick={() => setEventLevel("all")}>All</Button><Button onClick={() => setEventLevel("warn")}>Warn</Button><Button onClick={() => setEventLevel("error")}>Error</Button></div>
              <EventsTimeline events={events.data?.items ?? []} filterLevel={eventLevel} onSelectEvent={(event) => { setCrosshair(Math.floor(new Date(event.ts).getTime() / 1000)); setActiveTab("metrics"); }} />
            </div>
          )
        },
        { id: "artifacts", label: "Artifacts", panel: <ArtifactViewer runId={runId} artifacts={artifacts.data?.items ?? []} /> }
      ]} />
    </div>
  );
}
