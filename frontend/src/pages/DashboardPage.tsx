import { useEffect, useMemo, useState } from "react";
import { listRuns, RunSummary } from "../api";

const PIPELINE_TYPES = [
  "teacher_gold",
  "teacher_silver",
  "priors",
  "selector",
  "gate",
  "live",
  "full_pipeline",
];

export default function DashboardPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    try {
      const data = await listRuns({ limit: "200" });
      setRuns(data);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  useEffect(() => {
    load();
    const interval = setInterval(load, 5000);
    return () => clearInterval(interval);
  }, []);

  const latestByPipeline = useMemo(() => {
    const map: Record<string, RunSummary | undefined> = {};
    for (const type of PIPELINE_TYPES) {
      map[type] = runs.find((run) => run.manifest?.pipeline_type === type);
    }
    return map;
  }, [runs]);

  const runningRuns = runs.filter((run) => run.status?.status === "RUNNING");

  const failedLast24h = useMemo(() => {
    const cutoff = Date.now() - 24 * 60 * 60 * 1000;
    return runs.filter((run) => {
      const status = run.status?.status;
      const start = run.manifest?.start_time ? Date.parse(run.manifest.start_time) : 0;
      return status === "FAILED" && start >= cutoff;
    });
  }, [runs]);

  return (
    <section>
      <h1>Dashboard</h1>
      {error && <div className="error">{error}</div>}

      <div className="grid">
        <div className="card">
          <h2>Pipeline Ribbon</h2>
          <ul>
            {PIPELINE_TYPES.map((type) => {
              const run = latestByPipeline[type];
              return (
                <li key={type}>
                  <strong>{type}</strong>: {run?.status?.status || run?.manifest?.status || "N/A"} {run?.manifest?.run_id || ""}
                </li>
              );
            })}
          </ul>
        </div>
        <div className="card">
          <h2>Running Jobs</h2>
          {runningRuns.length === 0 ? (
            <p>None</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Run ID</th>
                  <th>Pipeline</th>
                  <th>Stage</th>
                </tr>
              </thead>
              <tbody>
                {runningRuns.map((run) => (
                  <tr key={run.manifest.run_id}>
                    <td>{run.manifest.run_id}</td>
                    <td>{run.manifest.pipeline_type}</td>
                    <td>{run.status?.stage || "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        <div className="card">
          <h2>Alerts</h2>
          <h3>Failed (last 24h)</h3>
          {failedLast24h.length === 0 ? (
            <p>None</p>
          ) : (
            <ul>
              {failedLast24h.map((run) => (
                <li key={run.manifest.run_id}>{run.manifest.run_id}</li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </section>
  );
}
