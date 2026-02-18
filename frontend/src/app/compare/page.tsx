"use client";

import { Suspense, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useQueries } from "@tanstack/react-query";
import dynamic from "next/dynamic";
import { api } from "@/lib/api";
import { Button, Card, EmptyState, PageHeader } from "@/components/ui/primitives";

const TimeSeriesChartLW = dynamic(() => import("@/components/TimeSeriesChartLW").then((m) => m.TimeSeriesChartLW), { ssr: false });

export default function ComparePage() {
  return <Suspense><CompareInner /></Suspense>;
}

function CompareInner() {
  const params = useSearchParams();
  const runIds = (params.get("runIds") ?? "").split(",").filter(Boolean).slice(0, 8);
  const [metric, setMetric] = useState("equity");
  const [alignment, setAlignment] = useState<"absolute" | "relative">("absolute");
  const [normalize, setNormalize] = useState<"raw" | "index100">("raw");
  const qs = useQueries({ queries: runIds.map((runId) => ({ queryKey: ["compare", runId], queryFn: () => api.getMetricsLW(runId, [metric, "equity", "pnl_net", "gross_exposure"]) })) });

  const series = useMemo(() => qs.map((q, idx) => ({ key: runIds[idx], name: runIds[idx].slice(0, 8), data: q.data?.series[metric] ?? [] })).filter((s) => s.data.length > 0), [qs, runIds, metric]);

  return (
    <div className="space-y-4">
      <PageHeader title="Compare Runs" subtitle={runIds.length ? `Comparing ${runIds.length} runs` : "Pick runs from research"} actions={<div className="flex gap-2"><Button onClick={() => navigator.clipboard.writeText(window.location.href)}>Copy share link</Button><Button onClick={() => navigator.clipboard.writeText(JSON.stringify(series))}>Export data</Button></div>} />
      {runIds.length === 0 ? <EmptyState title="No run IDs" message="Use /compare?runIds=id1,id2 or pick rows in Research." /> : (
        <>
          <Card className="flex flex-wrap gap-2"><select value={metric} onChange={(e) => setMetric(e.target.value)} className="rounded border border-border bg-surface-2 p-2"><option value="equity">equity</option><option value="pnl_net">pnl_net</option><option value="gross_exposure">gross_exposure</option></select><Button onClick={() => setAlignment((v) => v === "absolute" ? "relative" : "absolute")}>Alignment: {alignment}</Button><Button onClick={() => setNormalize((v) => v === "raw" ? "index100" : "raw")}>Normalize: {normalize}</Button></Card>
          {series.length > 0 ? <TimeSeriesChartLW title={metric} series={series} mode="full" alignment={alignment} normalize={normalize} /> : <EmptyState title="No data for selected metric" message="Choose another metric or run set." />}
        </>
      )}
    </div>
  );
}
