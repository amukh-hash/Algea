"use client";

import dynamic from "next/dynamic";
import { Suspense, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useQueries } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { auditLog } from "@/lib/audit";
import { Button, EmptyState, PageHeader, SearchInput } from "@/components/ui/primitives";

const Chart = dynamic(() => import("@/components/TimeSeriesChartLW").then((m) => m.TimeSeriesChartLW), { ssr: false });
const keys = ["pnl_net", "equity", "gross_exposure", "net_exposure", "train_loss"];

export default function ComparePage() {
  return <Suspense fallback={<div className="h-32 animate-pulse rounded-md bg-surface-2" />}><ComparePageInner /></Suspense>;
}

function ComparePageInner() {
  const params = useSearchParams();
  const runIds = (params.get("runIds") ?? "").split(",").filter(Boolean).slice(0, 10);
  const [metric, setMetric] = useState("pnl_net");
  const [align, setAlign] = useState<"absolute" | "relative">("absolute");
  const [normalize, setNormalize] = useState<"raw" | "index100">("raw");
  const [search, setSearch] = useState("");
  const queries = useQueries({ queries: runIds.map((runId) => ({ queryKey: ["cmp", runId], queryFn: () => api.getMetricsLW(runId, keys) })) });

  const series = useMemo(() => queries.map((q, i) => ({ key: runIds[i], name: runIds[i].slice(0, 8), data: q.data?.series[metric] ?? [], visible: true })), [queries, metric, runIds]);
  const coverage = useMemo(() => keys.map((k) => `${k} (${queries.filter((q) => (q.data?.series[k] ?? []).length > 0).length}/${queries.length})`), [queries]);

  return (
    <div className="space-y-4">
      <PageHeader title="Compare Runs" subtitle={runIds.length ? `Comparing ${runIds.length} runs` : "Select runs to begin"} actions={<><Button onClick={() => (auditLog("share_link"), navigator.clipboard.writeText(window.location.href))}>Copy share link</Button><Button onClick={() => (auditLog("export_compare_data", { metric }), navigator.clipboard.writeText(JSON.stringify(series)))}>Export data</Button></>} />
      {!runIds.length && <EmptyState title="No run IDs supplied" message="Use /compare?runIds=id1,id2 or open selector from research." />}
      {runIds.length > 0 && (
        <>
          {runIds.length > 5 && <div className="rounded border border-warning p-2 text-warning">More than 5 runs may reduce chart readability.</div>}
          <div className="flex flex-wrap gap-2">
            <SearchInput placeholder="Search metric" value={search} onChange={(e) => setSearch(e.target.value)} />
            <select value={metric} onChange={(e) => setMetric(e.target.value)} className="rounded border border-border bg-surface-2 px-3 py-2 text-sm">
              {coverage.filter((c) => c.includes(search)).map((label) => <option key={label} value={label.split(" ")[0]}>{label}</option>)}
            </select>
            <Button onClick={() => setAlign((v) => v === "absolute" ? "relative" : "absolute")}>Alignment: {align}</Button>
            <Button onClick={() => setNormalize((v) => v === "raw" ? "index100" : "raw")}>Normalize: {normalize}</Button>
          </div>
          <Chart title={metric} series={series} alignment={align} normalize={normalize} />
          <div className="space-y-2">
            {series.map((s, idx) => <div key={s.key} className="rounded border border-border bg-surface-1 p-2 text-sm"><label><input type="checkbox" defaultChecked /> <span className="text-secondary">Run {idx + 1}</span> {s.name}</label></div>)}
          </div>
        </>
      )}
    </div>
  );
}
