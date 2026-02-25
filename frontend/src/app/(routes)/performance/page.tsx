"use client";

export const dynamic = "force-dynamic";

import { useQuery } from "@tanstack/react-query";
import { PageHeader } from "@/components/ui/primitives";
import { orchApi } from "@/lib/orch";
import { useOpsFilters } from "@/lib/ops_filters";
import { OpsHeader } from "@/components/OpsHeader";
import { TrueCurvePanel } from "@/components/TrueCurvePanel";

export default function PerformancePage() {
  const { asof } = useOpsFilters();
  const curve = useQuery({ queryKey: ["portfolio-curve", asof], queryFn: () => asof ? orchApi.getEquitySeries(asof) : Promise.resolve({ asof: "", asof_date: "", scope: "portfolio", source: "", series: [] }) });

  return (
    <div className="space-y-4">
      <PageHeader title="Performance" subtitle="True curves from emitted equity artifacts" />
      <OpsHeader />
      <TrueCurvePanel title="Portfolio (unscaled + volscaled + diagnostics)" series={curve.data?.series ?? []} />
    </div>
  );
}
