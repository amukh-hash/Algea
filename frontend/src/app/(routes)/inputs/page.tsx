"use client";

export const dynamic = "force-dynamic";

import { useQuery } from "@tanstack/react-query";
import { PageHeader } from "@/components/ui/primitives";
import { orchApi } from "@/lib/orch";
import { useOpsFilters } from "@/lib/ops_filters";
import { OpsHeader } from "@/components/OpsHeader";

export default function InputsPage() {
  const { asof } = useOpsFilters();
  const instance = useQuery({ queryKey: ["instance", asof], queryFn: () => (asof ? orchApi.getInstance(asof) : orchApi.getLatestInstance()) });
  const artifacts = useQuery({ queryKey: ["artifacts", asof], queryFn: () => orchApi.listArtifacts(asof || undefined) });
  const quoteMissing = (artifacts.data?.items ?? []).filter((a) => a.relative_path.includes("quote") || a.relative_path.includes("price")).length === 0;

  return (
    <div className="space-y-4">
      <PageHeader title="Inputs" subtitle="Data / market inputs" />
      <OpsHeader health={quoteMissing ? "yellow" : "green"} />
      <div className="rounded border border-border bg-surface-1 p-4 text-sm">
        <div>Instance source: {instance.data?.source ?? "-"}</div>
        <div>Quote coverage: {quoteMissing ? "MISSING (see artifacts)" : "present"}</div>
      </div>
      <div className="rounded border border-border bg-surface-1 p-4"><pre className="overflow-auto text-xs">{JSON.stringify(instance.data?.instance ?? {}, null, 2)}</pre></div>
    </div>
  );
}
