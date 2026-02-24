"use client";

import Link from "next/link";
import { useMemo, useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useRunStream } from "@/lib/sse";
import { Button, EmptyState, PageHeader, SearchInput, Skeleton, StatusBadge } from "@/components/ui/primitives";
import { EventsTimeline } from "@/components/events_timeline";
import { RiskPanel } from "@/components/RiskPanel";
import { PortfolioValue } from "@/components/PortfolioValue";
import dynamic from "next/dynamic";

const Chart = dynamic(() => import("@/components/TimeSeriesChartLW").then((m) => m.TimeSeriesChartLW), { ssr: false });

export default function ExecutionPage() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const [paused, setPaused] = useState(false);
  const [q, setQ] = useState("");
  const [display, setDisplay] = useState<"cards" | "list">("cards");
  const { data, isLoading, refetch } = useQuery({ queryKey: ["execution-runs", q], queryFn: () => api.listRuns(q || "Sleeve", 40) });
  const runs = data?.items ?? [];
  const family = runs.filter((r) => r.tags.includes("family"));
  const other = runs.filter((r) => !r.tags.includes("family"));

  const warnings = useMemo(() => runs.filter((r) => ["error", "paused"].includes(r.status)).length, [runs]);
  const running = runs.filter((r) => r.status === "running").length;

  return (
    <div className="space-y-4">
      <PageHeader title="Trading Ops" subtitle="Live execution and recent sleeve runs" actions={<><Button onClick={() => setPaused((v) => !v)}>{paused ? "Resume" : "Pause live updates"}</Button><Button onClick={() => refetch()}>Refresh snapshot</Button></>} />
      <div className="sticky top-[90px] z-10 grid grid-cols-2 gap-3 rounded-md border border-border bg-surface-1 p-3 text-sm md:grid-cols-4">
        <div><div className="text-muted">Connection</div><div>Live stream</div></div>
        <div><div className="text-muted">Last update</div><div>{mounted ? new Date().toLocaleTimeString() : "--:--:--"}</div></div>
        <div><div className="text-muted">Running runs</div><div>{running}</div></div>
        <div><div className="text-muted">Warnings/errors</div><div>{warnings}</div></div>
      </div>
      <PortfolioValue />
      <RiskPanel />
      <div className="flex flex-wrap gap-2">
        <SearchInput value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search runs" />
        <Button variant={display === "cards" ? "primary" : "secondary"} onClick={() => setDisplay("cards")}>Cards</Button>
        <Button variant={display === "list" ? "primary" : "secondary"} onClick={() => setDisplay("list")}>List</Button>
      </div>
      {isLoading && <Skeleton className="h-40" />}
      {!isLoading && runs.length === 0 && <EmptyState title="No runs" message="Start a sleeve run to populate this dashboard." cta={<Link className="text-info" href="/research">Open research registry</Link>} />}
      <RunGroup title="Family runs" runs={family} paused={paused} display={display} />
      <RunGroup title="All other runs" runs={other} paused={paused} display={display} />
    </div>
  );
}

function RunGroup({ title, runs, paused, display }: { title: string; runs: Awaited<ReturnType<typeof api.listRuns>>["items"]; paused: boolean; display: "cards" | "list"; }) {
  const [collapsed, setCollapsed] = useState(false);
  if (!runs.length) return null;
  return (
    <section>
      <button className="mb-2 text-sm font-semibold" onClick={() => setCollapsed((v) => !v)}>{title} ({runs.length}) {collapsed ? "▸" : "▾"}</button>
      {!collapsed && <div className={display === "cards" ? "grid gap-3 md:grid-cols-2" : "space-y-2"}>{runs.map((run) => <RunItem key={run.run_id} runId={run.run_id} name={run.name} status={run.status} tags={run.tags} paused={paused} compact={display === "list"} />)}</div>}
    </section>
  );
}

function RunItem({ runId, name, status, tags, paused, compact }: { runId: string; name: string; status: string; tags: string[]; paused: boolean; compact: boolean; }) {
  const { metricsLW, events } = useRunStream(runId, paused);
  const pnl = metricsLW.pnl_net ?? metricsLW.equity ?? [];
  return (
    <article className="rounded-md border border-border bg-surface-1 p-3">
      <div className="flex items-center justify-between">
        <div><Link href={`/runs/${runId}`} className="font-medium text-primary">{name}</Link><div className="text-xs text-muted">{runId.slice(0, 12)}</div></div>
        <StatusBadge status={status} />
      </div>
      <div className="my-2 flex flex-wrap gap-1">{tags.slice(0, 4).map((tag) => <span key={tag} className="rounded bg-surface-2 px-2 py-0.5 text-xs text-secondary">{tag}</span>)}</div>
      {!compact && pnl.length > 0 && <Chart title="PnL / Equity" mode="compact" series={[{ key: "pnl", name: "PnL", data: pnl }]} height={120} />}
      <div className="mt-2"><EventsTimeline events={events.slice(0, 5)} filterLevel="warn" /></div>
    </article>
  );
}
