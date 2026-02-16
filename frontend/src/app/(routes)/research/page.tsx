"use client";

import Link from "next/link";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export default function ResearchPage() {
  const [q, setQ] = useState("");
  const { data } = useQuery({ queryKey: ["runs", q], queryFn: () => api.listRuns(q) });
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold">Research Run Registry</h1>
      <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="search" className="rounded bg-slate-900 px-3 py-2" />
      <table className="w-full text-sm">
        <thead><tr><th>Name</th><th>Type</th><th>Status</th><th>Started</th><th /></tr></thead>
        <tbody>
          {(data?.items ?? []).map((run) => (
            <tr key={run.run_id} className="border-t border-slate-800">
              <td>{run.name}</td><td>{run.run_type}</td><td>{run.status}</td><td>{new Date(run.started_at).toLocaleString()}</td>
              <td><Link href={`/runs/${run.run_id}`} className="underline">open</Link></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
