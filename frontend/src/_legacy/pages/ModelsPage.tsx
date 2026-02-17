import { useEffect, useMemo, useState } from "react";
import { listRuns, RunSummary } from "../api";

export default function ModelsPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);

  useEffect(() => {
    listRuns({ limit: "200" }).then(setRuns);
  }, []);

  const teacherRuns = useMemo(() => runs.filter((r) => r.manifest.pipeline_type?.includes("teacher")), [runs]);
  const selectorRuns = useMemo(() => runs.filter((r) => r.manifest.pipeline_type === "selector"), [runs]);

  return (
    <section>
      <h1>Models</h1>
      <div className="grid">
        <div className="card">
          <h2>Latest Teacher Runs</h2>
          <ul>
            {teacherRuns.slice(0, 5).map((run) => (
              <li key={run.manifest.run_id}>{run.manifest.run_id} - {run.status?.status}</li>
            ))}
          </ul>
        </div>
        <div className="card">
          <h2>Latest Selector Runs</h2>
          <ul>
            {selectorRuns.slice(0, 5).map((run) => (
              <li key={run.manifest.run_id}>{run.manifest.run_id} - {run.status?.status}</li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
