"use client";

export const dynamic = "force-dynamic";

import { useQuery } from "@tanstack/react-query";
import { PageHeader } from "@/components/ui/primitives";
import { orchApi } from "@/lib/orch";
import { useOpsFilters } from "@/lib/ops_filters";
import { OpsHeader } from "@/components/OpsHeader";
import { TrueCurvePanel } from "@/components/TrueCurvePanel";

const sleeves = ["core", "vrp", "selector"];

function mapStatus(targets: any[] | undefined, risk: any) {
  if (!targets) return "inputs_missing";
  if (!targets.length) return "noop";
  if (risk?.status !== "ok") return "blocked";
  return "live";
}

export default function SleevesPage() {
  const { asof } = useOpsFilters();
  const targets = useQuery({ queryKey: ["targets", asof], queryFn: () => orchApi.getTargets(asof || undefined), refetchInterval: 15000 });
  const risk = useQuery({ queryKey: ["risk", asof], queryFn: () => (asof ? orchApi.getRiskChecks(asof) : orchApi.getLatestRiskChecks()), refetchInterval: 15000 });

  return (
    <div className="space-y-4">
      <PageHeader title="Sleeves" subtitle="core / vrp / selector" />
      <OpsHeader />
      {sleeves.map((s) => {
        const rows = targets.data?.sleeves?.[s]?.targets;
        return (
          <div key={s} className="rounded border border-border bg-surface-1 p-4">
            <div className="mb-2 flex items-center justify-between"><h3 className="font-semibold capitalize">{s}</h3><span>{mapStatus(rows, risk.data?.risk_checks)}</span></div>
            <table className="mb-3 w-full text-xs"><thead><tr><th>symbol</th><th>target_weight</th></tr></thead><tbody>{(rows ?? []).map((r: any, i: number) => <tr key={`${r.symbol}-${i}`}><td>{r.symbol}</td><td>{r.target_weight}</td></tr>)}</tbody></table>
            <SleeveCurve sleeve={s} asof={asof} />
          </div>
        );
      })}
    </div>
  );
}

function SleeveCurve({ sleeve, asof }: { sleeve: string; asof: string }) {
  const q = useQuery({ queryKey: ["sleeve-curve", sleeve, asof], queryFn: () => asof ? orchApi.getEquitySeries(asof, sleeve) : Promise.resolve({ asof: "", asof_date: "", scope: sleeve, source: "", series: [] }) });
  return <TrueCurvePanel title={`${sleeve} curves`} series={q.data?.series ?? []} />;
}
