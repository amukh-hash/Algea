"use client";

import { useQuery } from "@tanstack/react-query";
import { useOpsFilters } from "@/lib/ops_filters";
import { orchApi } from "@/lib/orch";

export function OpsHeader({ health }: { health?: "green" | "yellow" | "red" }) {
  const { asof, session, setFilter } = useOpsFilters();
  const dates = useQuery({ queryKey: ["ops-dates"], queryFn: orchApi.listDates });

  return (
    <div className="mb-3 flex flex-wrap items-end gap-3 rounded border border-border bg-surface-1 p-3 text-sm">
      <div>
        <div className="text-xs text-muted">As-of date</div>
        <select className="rounded border border-border bg-surface-2 px-2 py-1" value={asof} onChange={(e) => setFilter({ asof: e.target.value })}>
          <option value="">Latest</option>
          {(dates.data?.items ?? []).map((d) => <option key={d} value={d}>{d}</option>)}
        </select>
      </div>
      <div>
        <div className="text-xs text-muted">Session</div>
        <input className="rounded border border-border bg-surface-2 px-2 py-1" value={session} onChange={(e) => setFilter({ session: e.target.value })} placeholder="open/intraday" />
      </div>
      <div>
        <div className="text-xs text-muted">Health</div>
        <div className="font-medium">{(health ?? "yellow").toUpperCase()}</div>
      </div>
    </div>
  );
}
