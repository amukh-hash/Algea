"use client";

import { useMemo, useState } from "react";
import { TelemetryEvent } from "@/lib/types";
import { StatusBadge } from "./ui/primitives";

export function EventsTimeline({ events, filterLevel = "all", search = "", onSelectEvent }: { events: TelemetryEvent[]; filterLevel?: "all" | "info" | "warn" | "error"; search?: string; onSelectEvent?: (event: TelemetryEvent) => void }) {
  const [level, setLevel] = useState(filterLevel);
  const [query, setQuery] = useState(search);
  const [active, setActive] = useState(0);
  const filtered = useMemo(() => events.filter((evt) => (level === "all" || evt.level.toLowerCase() === level) && `${evt.type} ${evt.message}`.toLowerCase().includes(query.toLowerCase())), [events, level, query]);

  return (
    <section className="rounded-lg border border-border bg-surface-1 p-3">
      <div className="mb-2 flex flex-wrap gap-2"><select className="rounded border border-border bg-surface-2 p-1 text-xs" value={level} onChange={(e) => setLevel(e.target.value as typeof level)}><option value="all">All</option><option value="info">Info</option><option value="warn">Warn</option><option value="error">Error</option></select><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search events" className="rounded border border-border bg-surface-2 p-1 text-xs" /></div>
      <div role="list" className="max-h-72 space-y-2 overflow-auto" onKeyDown={(e) => {
        if (e.key === "ArrowDown") setActive((i) => Math.min(filtered.length - 1, i + 1));
        if (e.key === "ArrowUp") setActive((i) => Math.max(0, i - 1));
        if (e.key === "Enter" && filtered[active]) onSelectEvent?.(filtered[active]);
      }}>
        {filtered.slice(0, 200).map((event, idx) => (
          <button role="listitem" key={`${event.ts}-${idx}`} className={`w-full rounded border p-2 text-left text-xs ${idx === active ? "border-info" : "border-border-subtle"}`} onClick={() => onSelectEvent?.(event)}>
            <div className="mb-1 flex flex-wrap items-center gap-2"><span className="text-secondary">{new Date(event.ts).toLocaleString()}</span><StatusBadge status={event.level} /><span className="text-secondary">{event.type}</span></div>
            <p>{event.message}</p>
          </button>
        ))}
      </div>
    </section>
  );
}
