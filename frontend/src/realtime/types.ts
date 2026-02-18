import { LWPoint, MetricPoint, TelemetryEvent } from "@/lib/types";

export type ConnectionState = "connecting" | "open" | "closed" | "error" | "reconnecting" | "rehydrating";

export type StreamEnvelope =
  | { id: number; type: "metric"; payload: MetricPoint }
  | { id: number; type: "event"; payload: TelemetryEvent }
  | { id: number; type: "status"; payload: { status: string } }
  | { id: number; type: "heartbeat"; payload: { ts: string } }
  | { id: number; type: "snapshot"; payload: { metrics: Record<string, LWPoint[]>; events: TelemetryEvent[] } };
