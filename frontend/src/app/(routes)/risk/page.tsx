"use client";

export const dynamic = "force-dynamic";

import { useQuery } from "@tanstack/react-query";
import { PageHeader } from "@/components/ui/primitives";
import { orchApi } from "@/lib/orch";
import { normalizeRiskChecks } from "@/lib/risk_schema";
import { useOpsFilters } from "@/lib/ops_filters";
import { OpsHeader } from "@/components/OpsHeader";

export default function RiskPage() {
  const { asof, session } = useOpsFilters();
  const risk = useQuery({ queryKey: ["risk", asof, session], queryFn: () => (asof ? orchApi.getRiskChecks(asof, session || undefined) : orchApi.getLatestRiskChecks()) });

  let parsed: any = null;
  let parseError: string | null = null;
  try {
    parsed = risk.data?.risk_checks ? normalizeRiskChecks(risk.data.risk_checks) : null;
  } catch (e: any) {
    parseError = e.message;
  }

  return (
    <div className="space-y-4">
      <PageHeader title="Risk" subtitle="Canonical + legacy normalized risk checks" />
      <OpsHeader health={parsed?.status === "ok" ? "green" : "red"} />
      {parseError && <div className="rounded border border-danger p-3 text-danger">{parseError}</div>}
      <div className="rounded border border-border bg-surface-1 p-4 text-sm">
        <div>Status: {parsed?.status ?? "missing"}</div>
        <div>Schema: {parsed?.schema_version ?? "-"}</div>
        <div>Reason: {parsed?.reason ?? "-"}</div>
      </div>
      <div className="rounded border border-border bg-surface-1 p-4 text-sm">
        <h3 className="mb-2 font-semibold">Violations</h3>
        <ul className="list-disc pl-5">{(parsed?.violations ?? []).map((v: any, i: number) => <li key={`${v.code}-${i}`}>{v.code}: {v.message}</li>)}</ul>
      </div>
      <div className="rounded border border-border bg-surface-1 p-4"><pre className="overflow-auto text-xs">{JSON.stringify(parsed ?? risk.data, null, 2)}</pre></div>
    </div>
  );
}
