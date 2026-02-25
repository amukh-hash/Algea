"use client";

export const dynamic = "force-dynamic";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { orchApi } from "@/lib/orch";
import { PageHeader } from "@/components/ui/primitives";
import { OpsHeader } from "@/components/OpsHeader";
import { useOpsFilters } from "@/lib/ops_filters";

function healthFrom(risk: any, history: any[]) {
  if (risk?.status !== "ok") return "red" as const;
  if (history.some((h) => ["failed", "error"].includes(String(h.status)))) return "red" as const;
  if (!history.length) return "yellow" as const;
  return "green" as const;
}

export default function OverviewPage() {
  const { asof } = useOpsFilters();
  const history = useQuery({ queryKey: ["job-history", asof], queryFn: () => orchApi.getJobHistory(100, asof || undefined) });
  const risk = useQuery({ queryKey: ["risk", asof], queryFn: () => (asof ? orchApi.getRiskChecks(asof) : orchApi.getLatestRiskChecks()) });
  const health = healthFrom(risk.data?.risk_checks, history.data?.items ?? []);

  return (
    <div className="space-y-4">
      <PageHeader title="Overview" subtitle="Ops health + global triage" />
      <OpsHeader health={health} />
      <div className="rounded border border-border bg-surface-1 p-4 text-sm">
        <div>Global health: <span className="font-semibold">{health.toUpperCase()}</span></div>
        <div>Risk status: {risk.data?.risk_checks?.status ?? "unknown"}</div>
      </div>
      <div className="rounded border border-border bg-surface-1 p-4">
        <h3 className="mb-2 font-semibold">Run timeline</h3>
        <div className="space-y-1 text-xs">
          {(history.data?.items ?? []).slice(0, 30).map((j, i) => (
            <div className="grid grid-cols-5 gap-2 border-b border-border/30 pb-1" key={`${j.run_id}-${j.name}-${i}`}>
              <span>{j.name}</span><span>{j.last_status}</span><span>{j.started_at ?? "-"}</span><span>{j.last_duration_s ?? "-"}s</span><span>{j.last_error ?? "-"}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
        <Link href="/sleeves" className="rounded border border-border p-3">Sleeves</Link>
        <Link href="/performance" className="rounded border border-border p-3">Performance</Link>
        <Link href="/risk" className="rounded border border-border p-3">Risk</Link>
        <Link href="/jobs" className="rounded border border-border p-3">Jobs</Link>
      </div>
    </div>
  );
}
