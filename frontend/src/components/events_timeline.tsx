import { TelemetryEvent } from "@/lib/types";

export function EventsTimeline({ events }: { events: TelemetryEvent[] }) {
  return (
    <div className="space-y-2 max-h-72 overflow-auto">
      {events.map((event, idx) => (
        <div key={`${event.ts}-${idx}`} className="rounded border border-slate-800 bg-slate-900 p-2 text-xs">
          <div className="text-slate-400">{event.ts} · {event.level} · {event.type}</div>
          <div>{event.message}</div>
        </div>
      ))}
    </div>
  );
}
