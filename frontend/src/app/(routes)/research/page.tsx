"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Button, Card, EmptyState, PageHeader, StatusBadge } from "@/components/ui/primitives";
import { useRouter } from "next/navigation";

export default function ResearchPage() {
  const [q, setQ] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const router = useRouter();
  const runs = useQuery({ queryKey: ["research-runs", q], queryFn: () => api.listRuns(q, 300), staleTime: 60_000 });

  const items = useMemo(() => [...(runs.data?.items ?? [])].sort((a, b) => sortDir === "desc" ? b.started_at.localeCompare(a.started_at) : a.started_at.localeCompare(b.started_at)), [runs.data, sortDir]);

  return (
    <div className="space-y-4">
      <PageHeader title="Research" subtitle="Registry of runs" actions={<Button onClick={() => runs.refetch()}>Refresh</Button>} />
      <Card className="sticky top-14 z-10">
        <div className="flex flex-wrap items-center gap-2"><input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search by id or name" className="min-w-72 flex-1 rounded-md border border-border bg-surface-2 p-2" /><Button onClick={() => setSortDir((s) => s === "asc" ? "desc" : "asc")}>Sort started {sortDir}</Button><Button disabled={selected.length < 2 || selected.length > 5} onClick={() => router.push(`/compare?runIds=${selected.join(",")}`)}>Compare ({selected.length})</Button></div>
      </Card>
      {runs.isError && <EmptyState title="Unable to fetch registry" message="Try manual refresh." cta={<Button onClick={() => runs.refetch()}>Retry</Button>} />}
      <div className="overflow-auto rounded-lg border border-border">
        <table className="min-w-full text-sm">
          <thead className="bg-surface-2 text-secondary"><tr><th className="p-2"><input aria-label="Select all" type="checkbox" checked={items.length > 0 && selected.length === items.length} onChange={(e) => setSelected(e.target.checked ? items.map((r) => r.run_id) : [])} /></th><th className="p-2 text-left">Name</th><th className="p-2 text-left">Type</th><th className="p-2 text-left">Status</th><th className="p-2 text-left" aria-sort={sortDir === "asc" ? "ascending" : "descending"}>Started</th></tr></thead>
          <tbody>
            {items.slice(0, 150).map((run) => (
              <tr key={run.run_id} className="border-t border-border-subtle hover:bg-surface-2/60">
                <td className="p-2"><input aria-label={`select ${run.name}`} type="checkbox" checked={selected.includes(run.run_id)} onChange={(e) => setSelected((prev) => e.target.checked ? [...prev, run.run_id] : prev.filter((id) => id !== run.run_id))} /></td>
                <td className="p-2"><Link className="text-info hover:underline" href={`/runs/${run.run_id}`}>{run.name}</Link></td>
                <td className="p-2">{run.run_type}</td>
                <td className="p-2"><StatusBadge status={run.status} /></td>
                <td className="p-2">{new Date(run.started_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
