import { useEffect, useMemo, useState } from "react";
import {
  getArtifacts,
  getCheckpoints,
  getEvents,
  getMetrics,
  getReport,
  getRun,
  listRuns,
  RunSummary,
} from "../api";

const TABS = ["Overview", "Stages", "Metrics", "Checkpoints", "Validations", "Artifacts"] as const;

export default function RunsPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [statusFilter, setStatusFilter] = useState("");
  const [pipelineFilter, setPipelineFilter] = useState("");
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [tab, setTab] = useState<(typeof TABS)[number]>("Overview");
  const [details, setDetails] = useState<any>(null);
  const [events, setEvents] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<any[]>([]);
  const [artifacts, setArtifacts] = useState<any>(null);
  const [checkpoints, setCheckpoints] = useState<any>(null);
  const [preflight, setPreflight] = useState<any>(null);
  const [gate, setGate] = useState<any>(null);
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [compareData, setCompareData] = useState<any>(null);

  const loadRuns = async () => {
    const params: Record<string, string> = { limit: "200" };
    if (statusFilter) params.status = statusFilter;
    if (pipelineFilter) params.pipeline_type = pipelineFilter;
    if (fromDate) params.from = new Date(fromDate).toISOString();
    if (toDate) params.to = new Date(toDate).toISOString();
    const data = await listRuns(params);
    setRuns(data);
  };

  useEffect(() => {
    loadRuns();
  }, [statusFilter, pipelineFilter, fromDate, toDate]);

  useEffect(() => {
    if (!selectedRun) return;
    const loadDetails = async () => {
      const run = await getRun(selectedRun);
      setDetails(run);
      setEvents(await getEvents(selectedRun, 200));
      setMetrics(await getMetrics(selectedRun));
      setArtifacts(await getArtifacts(selectedRun));
      setCheckpoints(await getCheckpoints(selectedRun).catch(() => null));
      setPreflight(await getReport(selectedRun, "preflight").catch(() => null));
      setGate(await getReport(selectedRun, "gate").catch(() => null));
    };
    loadDetails();
  }, [selectedRun]);

  const toggleCompare = (runId: string) => {
    setCompareIds((prev) => {
      if (prev.includes(runId)) {
        return prev.filter((id) => id !== runId);
      }
      if (prev.length >= 2) {
        return [prev[1], runId];
      }
      return [...prev, runId];
    });
  };

  useEffect(() => {
    const loadCompare = async () => {
      if (compareIds.length !== 2) {
        setCompareData(null);
        return;
      }
      const [a, b] = compareIds;
      const [runA, runB, gateA, gateB, metricsA, metricsB] = await Promise.all([
        getRun(a),
        getRun(b),
        getReport(a, "gate").catch(() => null),
        getReport(b, "gate").catch(() => null),
        getMetrics(a).catch(() => []),
        getMetrics(b).catch(() => []),
      ]);
      setCompareData({ runA, runB, gateA, gateB, metricsA, metricsB });
    };
    loadCompare();
  }, [compareIds]);

  const stagesSummary = useMemo(() => {
    const grouped: Record<string, number> = {};
    for (const event of events) {
      grouped[event.stage] = (grouped[event.stage] || 0) + 1;
    }
    return grouped;
  }, [events]);

  return (
    <section>
      <h1>Runs</h1>
      <div className="filters">
        <input placeholder="Status" value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)} />
        <input placeholder="Pipeline type" value={pipelineFilter} onChange={(e) => setPipelineFilter(e.target.value)} />
        <input type="date" value={fromDate} onChange={(e) => setFromDate(e.target.value)} />
        <input type="date" value={toDate} onChange={(e) => setToDate(e.target.value)} />
      </div>
      <table>
        <thead>
          <tr>
            <th>Compare</th>
            <th>Start</th>
            <th>Run ID</th>
            <th>Pipeline</th>
            <th>Status</th>
            <th>Stage</th>
            <th>Step</th>
            <th>Commit</th>
            <th>Data Versions</th>
            <th>Tags</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.manifest.run_id} onClick={() => setSelectedRun(run.manifest.run_id)}>
              <td>
                <input
                  type="checkbox"
                  checked={compareIds.includes(run.manifest.run_id)}
                  onChange={(e) => {
                    e.stopPropagation();
                    toggleCompare(run.manifest.run_id);
                  }}
                />
              </td>
              <td>{run.manifest.start_time}</td>
              <td>{run.manifest.run_id}</td>
              <td>{run.manifest.pipeline_type}</td>
              <td>{run.status?.status || run.manifest.status}</td>
              <td>{run.status?.stage || run.manifest.stage || "-"}</td>
              <td>{run.status?.step || run.manifest.step || "-"}</td>
              <td>{run.manifest.code_version?.git_commit?.slice(0, 7)}</td>
              <td>
                {Object.entries(run.manifest.data_versions || {})
                  .map(([k, v]) => `${k}:${String(v).slice(0, 6)}`)
                  .join(" ")}
              </td>
              <td>{(run.manifest.tags || []).join(", ")}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {compareData && (
        <div className="card">
          <h2>Compare Runs</h2>
          <div className="compare-grid">
            {[compareData.runA, compareData.runB].map((run: any, idx: number) => (
              <div key={idx}>
                <h3>{run.manifest.run_id}</h3>
                <p>Config Hash: {run.manifest.config_hash}</p>
                <p>Code: {run.manifest.code_version?.git_commit?.slice(0, 7)}</p>
                <p>Data Versions: {JSON.stringify(run.manifest.data_versions)}</p>
              </div>
            ))}
          </div>
          <div className="compare-grid">
            {[compareData.gateA, compareData.gateB].map((gateReport: any, idx: number) => (
              <div key={idx}>
                <h4>Gate Metrics</h4>
                <pre>{gateReport ? JSON.stringify(gateReport.metrics, null, 2) : "N/A"}</pre>
              </div>
            ))}
          </div>
          <div className="compare-grid">
            {[compareData.metricsA, compareData.metricsB].map((metricSeries: any[], idx: number) => (
              <div key={idx}>
                <h4>Metrics Snapshot</h4>
                <pre>{JSON.stringify(metricSeries.slice(-5), null, 2)}</pre>
              </div>
            ))}
          </div>
        </div>
      )}

      {selectedRun && details && (
        <div className="detail">
          <h2>Run Detail: {selectedRun}</h2>
          <div className="tabs">
            {TABS.map((name) => (
              <button key={name} className={tab === name ? "tab active" : "tab"} onClick={() => setTab(name)}>
                {name}
              </button>
            ))}
          </div>

          {tab === "Overview" && (
            <div className="card">
              <h3>Status</h3>
              <pre>{JSON.stringify(details, null, 2)}</pre>
            </div>
          )}

          {tab === "Stages" && (
            <div className="card">
              <h3>Stage Summary</h3>
              <ul>
                {Object.entries(stagesSummary).map(([stage, count]) => (
                  <li key={stage}>{stage}: {count}</li>
                ))}
              </ul>
            </div>
          )}

          {tab === "Metrics" && (
            <div className="card">
              <h3>Metrics</h3>
              <pre>{JSON.stringify(metrics.slice(-50), null, 2)}</pre>
            </div>
          )}

          {tab === "Checkpoints" && (
            <div className="card">
              <h3>Checkpoints</h3>
              <pre>{JSON.stringify(checkpoints, null, 2)}</pre>
            </div>
          )}

          {tab === "Validations" && (
            <div className="card">
              <h3>Preflight</h3>
              <pre>{preflight ? JSON.stringify(preflight, null, 2) : "Not available"}</pre>
              <h3>Gate</h3>
              <pre>{gate ? JSON.stringify(gate, null, 2) : "Not available"}</pre>
            </div>
          )}

          {tab === "Artifacts" && (
            <div className="card">
              <h3>Artifacts</h3>
              <pre>{JSON.stringify(artifacts, null, 2)}</pre>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
