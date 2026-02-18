"use client";

import { useMemo, useState } from "react";
import { api } from "@/lib/api";
import { Artifact } from "@/lib/types";

export function ArtifactViewer({ runId, artifacts }: { runId: string; artifacts: Artifact[] }) {
  const [selected, setSelected] = useState<Artifact | null>(artifacts[0] ?? null);
  const groups = useMemo(() => artifacts.reduce<Record<string, Artifact[]>>((acc, artifact) => {
    acc[artifact.kind] = acc[artifact.kind] ?? [];
    acc[artifact.kind].push(artifact);
    return acc;
  }, {}), [artifacts]);

  return (
    <div className="grid gap-4 md:grid-cols-[280px_1fr]">
      <aside className="max-h-[28rem] space-y-2 overflow-auto rounded-lg border border-border bg-surface-1 p-2">
        {Object.entries(groups).map(([kind, group]) => (
          <div key={kind}>
            <p className="px-2 py-1 text-xs text-secondary">{kind}</p>
            {group.map((artifact) => (
              <button key={artifact.artifact_id} onClick={() => setSelected(artifact)} className={`block w-full rounded-md p-2 text-left text-xs ${selected?.artifact_id === artifact.artifact_id ? "bg-surface-2" : "hover:bg-surface-2"}`}>
                {artifact.path.split("/").pop()}
              </button>
            ))}
          </div>
        ))}
      </aside>
      <section className="rounded-lg border border-border bg-surface-1 p-3">
        {!selected ? "Select artifact" : selected.mime.startsWith("image/") ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={api.artifactUrl(runId, selected.artifact_id)} alt={selected.path} className="max-h-[30rem]" />
        ) : (
          <div className="space-y-2 text-sm">
            <p>{selected.path}</p>
            <p className="text-secondary">{selected.mime} · {selected.bytes} bytes · {new Date(selected.created_at).toLocaleString()}</p>
            <a className="text-info underline" href={api.artifactUrl(runId, selected.artifact_id)} target="_blank" rel="noopener noreferrer">Open artifact</a>
          </div>
        )}
      </section>
    </div>
  );
}
