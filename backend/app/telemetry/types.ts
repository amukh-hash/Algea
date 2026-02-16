export type RunType = "sleeve_live" | "sleeve_paper" | "backtest" | "train";
export type RunStatus = "starting" | "running" | "paused" | "stopped" | "completed" | "error";
export type EventLevel = "debug" | "info" | "warn" | "error";
export type EventType =
  | "DECISION_MADE"
  | "GATE_TRIPPED"
  | "ORDER_SUBMITTED"
  | "ORDER_FILLED"
  | "ORDER_REJECTED"
  | "RISK_LIMIT"
  | "ERROR"
  | "CHECKPOINT_SAVED"
  | "EVAL_COMPLETE"
  | "BACKTEST_COMPLETE";
export type ArtifactKind = "report" | "plot" | "table" | "checkpoint" | "config" | "log" | "other";

export interface Run {
  run_id: string;
  run_type: RunType;
  name: string;
  sleeve_name: string | null;
  status: RunStatus;
  started_at: string;
  ended_at: string | null;
  git_sha: string;
  config_hash: string;
  data_version: string;
  tags: string[];
  meta: Record<string, unknown>;
}

export interface MetricPoint {
  run_id: string;
  ts: string;
  key: string;
  value: number;
  labels: Record<string, string>;
}

export interface Event {
  run_id: string;
  ts: string;
  level: EventLevel;
  type: EventType;
  message: string;
  payload: Record<string, unknown>;
}

export interface Artifact {
  run_id: string;
  artifact_id: string;
  path: string;
  kind: ArtifactKind;
  mime: string;
  bytes: number;
  created_at: string;
  meta: Record<string, unknown>;
}
