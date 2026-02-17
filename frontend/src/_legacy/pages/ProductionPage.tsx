import { useEffect, useMemo, useState } from "react";
import { listRuns, RunSummary } from "../api";

export default function ProductionPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);

  useEffect(() => {
    listRuns({ pipeline_type: "gate", status: "PASSED", limit: "20" }).then(setRuns);
  }, []);

  const latestGate = useMemo(() => runs[0], [runs]);

  return (
    <section>
      <h1>Production</h1>
      <div className="card">
        <h2>Latest Gate Passed Run</h2>
        {latestGate ? (
          <div>
            <p>Run ID: {latestGate.manifest.run_id}</p>
            <p>Started: {latestGate.manifest.start_time}</p>
          </div>
        ) : (
          <p>No gate passed runs found.</p>
        )}
      </div>
    </section>
  );
}
