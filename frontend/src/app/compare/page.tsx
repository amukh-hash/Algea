"use client";

import { useSearchParams } from "next/navigation";
import { useQueries } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function ComparePage() {
  const params = useSearchParams();
  const runIds = (params.get("runIds") ?? "").split(",").filter(Boolean).slice(0, 5);
  const metricQueries = useQueries({
    queries: runIds.map((runId) => ({ queryKey: ["compare", runId], queryFn: () => api.getMetrics(runId, ["pnl_net", "cum_net", "train_loss"]) })),
  });

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold">Compare Runs</h1>
      <div className="h-72 rounded border border-slate-700 p-2">
        <ResponsiveContainer>
          <LineChart>
            <XAxis dataKey="ts" hide /><YAxis hide /><Tooltip />
            {metricQueries.map((query, idx) => {
              const data = query.data?.series.pnl_net ?? query.data?.series.cum_net ?? query.data?.series.train_loss ?? [];
              return <Line key={runIds[idx]} data={data} dataKey="value" name={runIds[idx]} stroke={["#38bdf8", "#22c55e", "#f97316", "#f43f5e", "#a78bfa"][idx]} dot={false} />;
            })}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
