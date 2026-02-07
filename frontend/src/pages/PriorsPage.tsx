import { useEffect, useMemo, useState } from "react";
import { searchArtifacts } from "../api";

export default function PriorsPage() {
  const [artifacts, setArtifacts] = useState<any[]>([]);

  useEffect(() => {
    searchArtifacts("priors").then(setArtifacts);
  }, []);

  const priors = useMemo(() => artifacts.filter((a) => a.name?.includes("priors")), [artifacts]);

  return (
    <section>
      <h1>Priors</h1>
      <table>
        <thead>
          <tr>
            <th>Run</th>
            <th>Name</th>
            <th>Path</th>
            <th>Meta</th>
          </tr>
        </thead>
        <tbody>
          {priors.map((artifact) => (
            <tr key={`${artifact.run_id}-${artifact.path}`}>
              <td>{artifact.run_id}</td>
              <td>{artifact.name}</td>
              <td>{artifact.path}</td>
              <td><pre>{JSON.stringify(artifact.meta, null, 2)}</pre></td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
