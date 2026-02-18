export type RunType = "sleeve_live" | "sleeve_paper" | "backtest" | "train";
export type RunStatus = "starting" | "running" | "paused" | "stopped" | "completed" | "error";

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

export interface MetricPoint { run_id: string; ts: string; key: string; value: number; labels: Record<string, string> }
export interface TelemetryEvent { run_id: string; ts: string; level: string; type: string; message: string; payload: Record<string, unknown> }
export interface Artifact { run_id: string; artifact_id: string; path: string; kind: string; mime: string; bytes: number; created_at: string; meta: Record<string, unknown> }

export interface LWPoint { time: number; value: number }
