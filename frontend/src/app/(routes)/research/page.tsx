"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { auditLog } from "@/lib/audit";
import { Button, PageHeader, SearchInput, StatusBadge } from "@/components/ui/primitives";

export default function ResearchPage() {
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<"name" | "type" | "status" | "started">("started");
  const [selected, setSelected] = useState<string[]>([]);
  const router = useRouter();
  const { data, refetch } = useQuery({ queryKey: ["registry", q], queryFn: () => api.listRuns(q, 200), staleTime: 60_000 });

  const items = useMemo(() => [...(data?.items ?? [])].sort((a, b) => {
    if (sort === "started") return a.started_at < b.started_at ? 1 : -1;
    if (sort === "name") return a.name.localeCompare(b.name);
    if (sort === "type") return a.run_type.localeCompare(b.run_type);
    return a.status.localeCompare(b.status);
  }), [data?.items, sort]);

  return (
    <div className="space-y-4">
      <PageHeader title="Research Run Registry" subtitle={`${data?.total ?? 0} total runs`} actions={<><Button onClick={() => refetch()}>Refresh</Button><Button onClick={() => (auditLog("copy_ids", { count: selected.length }), navigator.clipboard.writeText(selected.join(",")))}>Copy IDs</Button></>} />
      <div className="sticky top-[90px] z-10 flex flex-wrap items-center gap-2 rounded-md border border-border bg-surface-1 p-3">
        <SearchInput value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search runs" />
        <Button onClick={() => setSort("name")}>Sort name</Button>
        <Button onClick={() => setSort("type")}>Sort type</Button>
        <Button onClick={() => setSort("status")}>Sort status</Button>
        <Button onClick={() => setSort("started")}>Sort started</Button>
        <Button variant="primary" disabled={selected.length < 2 || selected.length > 5} onClick={() => router.push(`/compare?runIds=${selected.join(",")}`)}>Compare ({selected.length})</Button>
      </div>
      <div className="overflow-auto rounded-md border border-border">
        <table className="w-full text-sm">
          <thead className="bg-surface-2 text-left text-xs text-secondary">
            <tr>
              <th className="p-2"><input type="checkbox" aria-label="Select all" onChange={(e) => setSelected(e.target.checked ? items.map((r) => r.run_id) : [])} checked={selected.length > 0 && selected.length === items.length} /></th>
              <SortTh label="Name" active={sort === "name"} onClick={() => setSort("name")} />
              <SortTh label="Type" active={sort === "type"} onClick={() => setSort("type")} />
              <SortTh label="Status" active={sort === "status"} onClick={() => setSort("status")} />
              <SortTh label="Started" active={sort === "started"} onClick={() => setSort("started")} />
              <th className="p-2">Tags</th>
              <th className="p-2">Action</th>
            </tr>
          </thead>
          <tbody>
            {items.slice(0, 100).map((run, idx) => (
              <tr key={run.run_id} tabIndex={0} className="border-t border-border-subtle" onKeyDown={(e) => {
                if (e.key === " ") { e.preventDefault(); setSelected((s) => s.includes(run.run_id) ? s.filter((id) => id !== run.run_id) : [...s, run.run_id].slice(0, 5)); }
                if (e.key === "Enter") router.push(`/runs/${run.run_id}`);
                if (e.shiftKey && e.key === "ArrowDown" && items[idx + 1]) setSelected((s) => Array.from(new Set([...s, run.run_id, items[idx + 1].run_id])).slice(0, 5));
              }}>
                <td className="p-2"><input type="checkbox" checked={selected.includes(run.run_id)} onChange={() => setSelected((s) => s.includes(run.run_id) ? s.filter((id) => id !== run.run_id) : [...s, run.run_id].slice(0, 5))} /></td>
                <td className="p-2"><Link href={`/runs/${run.run_id}`} className="text-info">{run.name}</Link></td>
                <td className="p-2">{run.run_type}</td>
                <td className="p-2"><StatusBadge status={run.status} /></td>
                <td className="p-2">{new Date(run.started_at).toLocaleString()}</td>
                <td className="p-2">{run.tags.slice(0, 3).join(", ")}</td>
                <td className="p-2"><Link href={`/runs/${run.run_id}`} className="text-info">Open</Link></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SortTh({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return <th className="p-2" aria-sort={active ? "ascending" : "none"}><button onClick={onClick}>{label}</button></th>;
}
