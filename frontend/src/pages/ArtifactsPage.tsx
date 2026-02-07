import { useEffect, useMemo, useState } from "react";
import { searchArtifacts } from "../api";

export default function ArtifactsPage() {
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState("");
  const [artifacts, setArtifacts] = useState<any[]>([]);

  useEffect(() => {
    const load = async () => {
      const data = await searchArtifacts(search);
      setArtifacts(data);
    };
    load();
  }, [search]);

  const filtered = useMemo(() => {
    if (!typeFilter) return artifacts;
    return artifacts.filter((a) => a.type === typeFilter);
  }, [artifacts, typeFilter]);

  const copyPath = (path: string) => {
    navigator.clipboard.writeText(path).catch(() => null);
  };

  return (
    <section>
      <h1>Artifacts</h1>
      <div className="filters">
        <input placeholder="Search" value={search} onChange={(e) => setSearch(e.target.value)} />
        <input placeholder="Type" value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)} />
      </div>
      <table>
        <thead>
          <tr>
            <th>Run</th>
            <th>Name</th>
            <th>Type</th>
            <th>Path</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map((artifact) => (
            <tr key={`${artifact.run_id}-${artifact.path}`}>
              <td>{artifact.run_id}</td>
              <td>{artifact.name}</td>
              <td>{artifact.type}</td>
              <td>{artifact.path}</td>
              <td>
                <button type="button" onClick={() => copyPath(artifact.path)}>Copy path</button>
                <a
                  className="link"
                  href={`/control-room/runs/${artifact.run_id}/file?path=${encodeURIComponent(artifact.path)}`}
                >
                  Download
                </a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
