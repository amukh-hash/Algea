"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useRunStream } from "@/lib/sse";
import { Button, Card, EmptyState, PageHeader, StatusBadge } from "@/components/ui/primitives";
import { useToast } from "@/components/ui/toast";

function RunCard({ runId, name, paused }: { runId: string; name: string; paused: boolean }) {
  const stream = useRunStream(runId);
  const latest = (k: string) => stream.metricsLW[k]?.at(-1)?.value;
  return (
    <Card className="space-y-3">
      <div className="flex items-start justify-between gap-2"><div><Link href={`/runs/${runId}`} className="font-medium text-info hover:underline">{name}</Link><p className="text-xs text-secondary">{runId.slice(0, 12)}</p></div><StatusBadge status={stream.status} /></div>
      <div className="grid grid-cols-3 gap-2 text-xs">
        <div><p className="text-secondary">PnL</p><p>{latest("pnl_net")?.toFixed(2) ?? "—"}</p></div>
        <div><p className="text-secondary">Gross</p><p>{latest("gross_exposure")?.toFixed(2) ?? "—"}</p></div>
        <div><p className="text-secondary">Risk</p><p>{latest("risk_scale")?.toFixed(2) ?? "—"}</p></div>
      </div>
      <p className="text-xs text-secondary">Connection: {stream.connectionState}{paused ? " · buffered" : ""}</p>
    </Card>
  );
}

export default function ExecutionPage() {
  const [search, setSearch] = useState("");
  const [paused, setPaused] = useState(false);
  const { push } = useToast();
  const runs = useQuery({ queryKey: ["runs", search], queryFn: () => api.listRuns(search, 200), staleTime: 30_000 });

  const filtered = useMemo(() => (runs.data?.items ?? []).filter((run) => run.name.toLowerCase().includes(search.toLowerCase()) || run.run_id.includes(search)), [runs.data, search]);
  const family = filtered.filter((r) => r.tags.includes("family"));
  const other = filtered.filter((r) => !r.tags.includes("family"));

  return (
    <div className="space-y-4">
      <PageHeader title="Trading Ops" subtitle="Live execution and recent sleeve runs" actions={<div className="flex gap-2"><Button onClick={() => setPaused((v) => !v)}>{paused ? "Resume" : "Pause live updates"}</Button><Button onClick={() => runs.refetch()}>Refresh snapshot</Button></div>} />
      <Card className="sticky top-14 z-10 flex flex-wrap gap-4 text-sm">
        <span className="rounded-full border border-success/40 px-2 py-1 text-success">Live stream</span>
        <span>Running: {filtered.filter((r) => r.status === "running").length}</span>
        <span>Warn/Error: {filtered.filter((r) => ["error", "paused"].includes(r.status)).length}</span>
      </Card>
      <Card>
        <label className="text-xs text-secondary" htmlFor="run-search">Search runs</label>
        <input id="run-search" value={search} onChange={(e) => setSearch(e.target.value)} className="mt-1 w-full rounded-md border border-border bg-surface-2 p-2" />
      </Card>
      {runs.isError && <EmptyState title="Failed to load runs" message="Check API connectivity and retry." cta={<Button onClick={() => runs.refetch()}>Retry</Button>} />}
      <section className="space-y-2"><h2 className="text-sm font-semibold">Family runs ({family.length})</h2><div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">{family.map((run) => <RunCard key={run.run_id} runId={run.run_id} name={run.name} paused={paused} />)}</div></section>
      <section className="space-y-2"><h2 className="text-sm font-semibold">Other runs ({other.length})</h2><div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">{other.map((run) => <RunCard key={run.run_id} runId={run.run_id} name={run.name} paused={paused} />)}</div></section>
      {paused && <Button variant="ghost" onClick={() => push("Buffered updates discarded")}>Resume (discard buffered)</Button>}
    </div>
  );
}
