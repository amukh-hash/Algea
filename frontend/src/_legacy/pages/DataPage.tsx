import { useEffect, useMemo, useState } from "react";
import { listRuns, RunSummary } from "../api";

export default function DataPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);

  useEffect(() => {
    listRuns({ limit: "200" }).then(setRuns);
  }, []);

  const dataRuns = useMemo(
    () => runs.filter((run) => (run.manifest.tags || []).some((tag: string) => ["ingest", "universe", "preproc"].includes(tag))),
    [runs]
  );

  return (
    <section>
      <h1>Data</h1>
      <table>
        <thead>
          <tr>
            <th>Run ID</th>
            <th>Tags</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {dataRuns.map((run) => (
            <tr key={run.manifest.run_id}>
              <td>{run.manifest.run_id}</td>
              <td>{(run.manifest.tags || []).join(", ")}</td>
              <td>{run.status?.status}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
