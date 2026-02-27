"use client";

export const dynamic = "force-dynamic";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { PageHeader, Button } from "@/components/ui/primitives";
import { orchApi } from "@/lib/orch";
import { controlApi } from "@/lib/control";
import { useOpsFilters } from "@/lib/ops_filters";
import { OpsHeader } from "@/components/OpsHeader";

export default function JobsPage() {
  const { asof } = useOpsFilters();
  const qc = useQueryClient();
  const registry = useQuery({ queryKey: ["jobs-registry"], queryFn: orchApi.getJobs, refetchInterval: 60000 });
  const history = useQuery({ queryKey: ["jobs-history", asof], queryFn: () => orchApi.getJobHistory(200, asof || undefined), refetchInterval: 15000 });
  const dryRun = useMutation({ mutationFn: () => controlApi.triggerTick(true), onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs-history"] }) });

  return (
    <div className="space-y-4">
      <PageHeader title="Jobs" subtitle="Registry + history + guarded trigger" actions={<Button onClick={() => dryRun.mutate()}>Dry-run trigger</Button>} />
      <OpsHeader />
      <div className="rounded border border-border bg-surface-1 p-4">
        <h3 className="mb-2 font-semibold">Registry</h3>
        <table className="w-full text-xs"><thead><tr><th>job</th><th>sessions</th><th>min_interval_s</th><th>cooldown_s</th></tr></thead><tbody>{(registry.data?.items ?? []).map((j) => <tr key={j.name}><td>{j.name}</td><td>{j.sessions.join(",")}</td><td>{j.min_interval_s}</td><td>{j.min_interval_s}</td></tr>)}</tbody></table>
      </div>
      <div className="rounded border border-border bg-surface-1 p-4">
        <h3 className="mb-2 font-semibold">History</h3>
        <table className="w-full text-xs"><thead><tr><th>job</th><th>status</th><th>duration</th><th>last_success_at</th><th>next_eligible</th><th>error</th></tr></thead><tbody>{(history.data?.items ?? []).map((j, i) => <tr key={`${j.run_id}-${j.name}-${i}`}><td>{j.name}</td><td>{j.last_status}</td><td>{j.last_duration_s ?? "-"}</td><td>{j.last_success_at ?? "-"}</td><td>{j.next_eligible_at ?? "-"}</td><td>{j.last_error ?? "-"}</td></tr>)}</tbody></table>
      </div>
      <div className="rounded border border-border bg-surface-1 p-3 text-xs">Stage-level weekly plan artifacts were not found; displaying job-level progress.</div>
    </div>
  );
}
