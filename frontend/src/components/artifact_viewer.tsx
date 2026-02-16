"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { Artifact } from "@/lib/types";

export function ArtifactViewer({ runId, artifacts }: { runId: string; artifacts: Artifact[] }) {
  const [selected, setSelected] = useState<Artifact | null>(artifacts[0] ?? null);
  return (
    <div className="grid grid-cols-3 gap-4">
      <div className="col-span-1 space-y-2">
        {artifacts.map((artifact) => (
          <button key={artifact.artifact_id} onClick={() => setSelected(artifact)} className="block w-full rounded border border-slate-700 p-2 text-left text-xs">
            {artifact.kind} · {artifact.path.split("/").pop()}
          </button>
        ))}
      </div>
      <div className="col-span-2 rounded border border-slate-700 p-3">
        {!selected ? "Select artifact" : selected.mime.startsWith("image/") ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={api.artifactUrl(runId, selected.artifact_id)} alt={selected.path} className="max-h-[28rem]" />
        ) : (
          <a className="underline" href={api.artifactUrl(runId, selected.artifact_id)} target="_blank">Open artifact</a>
        )}
      </div>
    </div>
  );
}
