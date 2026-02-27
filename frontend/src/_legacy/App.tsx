import { useMemo, useState } from "react";
import DashboardPage from "./pages/DashboardPage";
import RunsPage from "./pages/RunsPage";
import ArtifactsPage from "./pages/ArtifactsPage";
import ValidationsPage from "./pages/ValidationsPage";
import ModelsPage from "./pages/ModelsPage";
import PriorsPage from "./pages/PriorsPage";
import ProductionPage from "./pages/ProductionPage";
import DataPage from "./pages/DataPage";

const NAV_ITEMS = [
  "Dashboard",
  "Runs",
  "Artifacts",
  "Validations & Gates",
  "Models",
  "Priors",
  "Production",
  "Data",
];

export default function App() {
  const [active, setActive] = useState("Dashboard");

  const page = useMemo(() => {
    switch (active) {
      case "Dashboard":
        return <DashboardPage />;
      case "Runs":
        return <RunsPage />;
      case "Artifacts":
        return <ArtifactsPage />;
      case "Validations & Gates":
        return <ValidationsPage />;
      case "Models":
        return <ModelsPage />;
      case "Priors":
        return <PriorsPage />;
      case "Production":
        return <ProductionPage />;
      case "Data":
        return <DataPage />;
      default:
        return null;
    }
  }, [active]);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">Algae 4.0 Control Room</div>
        <nav>
          {NAV_ITEMS.map((item) => (
            <button
              key={item}
              className={active === item ? "nav-item active" : "nav-item"}
              onClick={() => setActive(item)}
              type="button"
            >
              {item}
            </button>
          ))}
        </nav>
      </aside>
      <main className="main">
        {page}
      </main>
    </div>
  );
}
