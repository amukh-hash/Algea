import { useEffect, useState } from "react";
import { getReport, listRuns, RunSummary } from "../api";

export default function ValidationsPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [report, setReport] = useState<any>(null);
  const [reportType, setReportType] = useState<string | null>(null);

  useEffect(() => {
    listRuns({ limit: "200" }).then(setRuns);
  }, []);

  const loadReport = async (runId: string, type: "preflight" | "gate") => {
    const data = await getReport(runId, type).catch(() => null);
    setReport(data);
    setReportType(`${type} (${runId})`);
  };

  return (
    <section>
      <h1>Validations & Gates</h1>
      <table>
        <thead>
          <tr>
            <th>Run ID</th>
            <th>Pipeline</th>
            <th>Preflight</th>
            <th>Gate</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.manifest.run_id}>
              <td>{run.manifest.run_id}</td>
              <td>{run.manifest.pipeline_type}</td>
              <td>
                <button type="button" onClick={() => loadReport(run.manifest.run_id, "preflight")}>
                  View
                </button>
              </td>
              <td>
                <button type="button" onClick={() => loadReport(run.manifest.run_id, "gate")}>
                  View
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="card">
        <h2>{reportType || "Report"}</h2>
        <pre>{report ? JSON.stringify(report, null, 2) : "Select a report"}</pre>
      </div>
    </section>
  );
}
