"use client";

export const dynamic = "force-dynamic";

import { PageHeader } from "@/components/ui/primitives";
import { useQuery } from "@tanstack/react-query";
import { orchApi } from "@/lib/orch";
import { useOpsFilters } from "@/lib/ops_filters";
import { OpsHeader } from "@/components/OpsHeader";

export default function ArtifactsPage() {
  const { asof } = useOpsFilters();
  const artifacts = useQuery({ queryKey: ["artifacts", asof], queryFn: () => orchApi.listArtifacts(asof || undefined) });

  return (
    <div className="space-y-4">
      <PageHeader title="Artifacts" subtitle="Raw emitted files" />
      <OpsHeader />
      <div className="rounded border border-border bg-surface-1 p-4">
        <table className="w-full text-xs"><thead><tr><th>asof</th><th>path</th><th>size</th><th>modified</th><th>download</th></tr></thead><tbody>{(artifacts.data?.items ?? []).map((a, i) => <tr key={`${a.asof}-${a.relative_path}-${i}`}><td>{a.asof}</td><td>{a.relative_path}</td><td>{a.size_bytes}</td><td>{a.modified_at}</td><td><a className="text-info" href={a.download_url} target="_blank">open</a></td></tr>)}</tbody></table>
      </div>
    </div>
  );
}
