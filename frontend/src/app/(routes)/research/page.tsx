"use client";

import Link from "next/link";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

const RUN_TYPE_COLORS: Record<string, string> = {
  sleeve_live: "bg-green-500/20 text-green-400",
  sleeve_paper: "bg-sky-500/20 text-sky-400",
  backtest: "bg-violet-500/20 text-violet-400",
  train: "bg-amber-500/20 text-amber-400",
};

const STATUS_COLORS: Record<string, string> = {
  completed: "text-green-400",
  running: "text-sky-400",
  error: "text-red-400",
  stopped: "text-slate-500",
  starting: "text-yellow-400",
};

export default function ResearchPage() {
  const [q, setQ] = useState("");
  const [typeFilter, setTypeFilter] = useState<string>("");
  const { data, isLoading } = useQuery({
    queryKey: ["runs", q],
    queryFn: () => api.listRuns(q, 100),
  });

  const filteredItems = (data?.items ?? []).filter(
    (run) => !typeFilter || run.run_type === typeFilter
  );

  const typeCounts: Record<string, number> = {};
  (data?.items ?? []).forEach((r) => {
    typeCounts[r.run_type] = (typeCounts[r.run_type] ?? 0) + 1;
  });

  // Compare functionality
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else if (next.size < 5) next.add(id);
      return next;
    });
  };

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Research Run Registry</h1>
          <p className="text-sm text-slate-500">{data?.total ?? 0} total runs</p>
        </div>
        {selectedIds.size >= 2 && (
          <Link
            href={`/compare?runIds=${Array.from(selectedIds).join(",")}`}
            className="rounded bg-sky-600 px-4 py-2 text-sm font-medium text-white shadow hover:bg-sky-500 transition"
          >
            Compare {selectedIds.size} runs →
          </Link>
        )}
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search runs…"
          className="rounded-lg border border-slate-800 bg-slate-900 px-4 py-2 text-sm text-slate-200 placeholder-slate-600 outline-none focus:border-sky-600 focus:ring-1 focus:ring-sky-600"
        />
        <div className="flex gap-1">
          <button
            onClick={() => setTypeFilter("")}
            className={`rounded px-3 py-1.5 text-xs font-medium transition ${!typeFilter ? "bg-slate-700 text-slate-200" : "bg-slate-900 text-slate-500 hover:text-slate-300"
              }`}
          >
            All
          </button>
          {Object.keys(typeCounts).map((type) => (
            <button
              key={type}
              onClick={() => setTypeFilter(type === typeFilter ? "" : type)}
              className={`rounded px-3 py-1.5 text-xs font-medium transition ${typeFilter === type
                  ? RUN_TYPE_COLORS[type] ?? "bg-slate-700 text-slate-200"
                  : "bg-slate-900 text-slate-500 hover:text-slate-300"
                }`}
            >
              {type} ({typeCounts[type]})
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      {isLoading ? (
        <div className="space-y-2">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="h-12 animate-pulse rounded bg-slate-900" />
          ))}
        </div>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-slate-800">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/50 text-left text-xs text-slate-500">
                <th className="p-3 w-8" />
                <th className="p-3">Name</th>
                <th className="p-3">Type</th>
                <th className="p-3">Status</th>
                <th className="p-3">Started</th>
                <th className="p-3">Tags</th>
                <th className="p-3" />
              </tr>
            </thead>
            <tbody>
              {filteredItems.map((run) => (
                <tr
                  key={run.run_id}
                  className="border-t border-slate-800/50 transition hover:bg-slate-900/30"
                >
                  <td className="p-3">
                    <input
                      type="checkbox"
                      checked={selectedIds.has(run.run_id)}
                      onChange={() => toggleSelect(run.run_id)}
                      className="h-3.5 w-3.5 rounded border-slate-600 bg-slate-900"
                    />
                  </td>
                  <td className="p-3">
                    <Link href={`/runs/${run.run_id}`} className="text-slate-200 hover:text-sky-400">
                      {run.name}
                    </Link>
                  </td>
                  <td className="p-3">
                    <span className={`rounded px-2 py-0.5 text-[10px] font-medium ${RUN_TYPE_COLORS[run.run_type] ?? "bg-slate-800 text-slate-400"}`}>
                      {run.run_type}
                    </span>
                  </td>
                  <td className="p-3">
                    <span className={`text-xs font-medium ${STATUS_COLORS[run.status] ?? "text-slate-400"}`}>
                      {run.status}
                    </span>
                  </td>
                  <td className="p-3 text-xs text-slate-500">
                    {new Date(run.started_at).toLocaleString()}
                  </td>
                  <td className="p-3">
                    <div className="flex flex-wrap gap-1">
                      {run.tags.slice(0, 3).map((t) => (
                        <span key={t} className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-500">
                          {t}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="p-3">
                    <Link href={`/runs/${run.run_id}`} className="text-xs text-sky-400 hover:underline">
                      open →
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
