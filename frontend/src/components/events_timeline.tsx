"use client";

import { TelemetryEvent } from "@/lib/types";
import { useMemo, useState } from "react";

export function EventsTimeline({ events, filterLevel = "all", search = "", onSelectEvent }: { events: TelemetryEvent[]; filterLevel?: "all" | "info" | "warn" | "error"; search?: string; onSelectEvent?: (event: TelemetryEvent) => void; }) {
  const [selected, setSelected] = useState(0);
  const filtered = useMemo(() => events.filter((e) => (filterLevel === "all" || e.level === filterLevel) && `${e.type} ${e.message}`.toLowerCase().includes(search.toLowerCase())), [events, filterLevel, search]);

  return (
    <div>
      <div className="mb-2 text-xs text-secondary">Events ({filtered.length})</div>
      <ul role="list" className="max-h-72 space-y-2 overflow-auto" onKeyDown={(e) => {
        if (e.key === "ArrowDown") setSelected((v) => Math.min(v + 1, filtered.length - 1));
        if (e.key === "ArrowUp") setSelected((v) => Math.max(v - 1, 0));
        if (e.key === "Enter" && filtered[selected]) onSelectEvent?.(filtered[selected]);
      }}>
        {filtered.map((event, idx) => (
          <li key={`${event.ts}-${idx}`}>
            <button className={`w-full rounded border border-border bg-surface-1 p-2 text-left text-xs ${idx === selected ? "ring-1 ring-info" : ""}`} onClick={() => onSelectEvent?.(event)}>
              <div className="text-secondary">{new Date(event.ts).toLocaleString()} · {event.level} · {event.type}</div>
              <div className="text-primary">{event.message}</div>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
