export type ConnectionState = "connecting" | "open" | "closed" | "error" | "reconnecting" | "rehydrating";

export type StreamEnvelope<T> = {
  id?: string;
  run_id?: string;
  data: T;
};
