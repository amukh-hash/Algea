import type { RiskChecksReport } from "./orch";

export function normalizeRiskChecks(raw: RiskChecksReport): RiskChecksReport {
  if (!raw || !raw.schema_version) throw new Error("Risk report missing schema_version");
  if (typeof raw.status !== "string") throw new Error("Risk status must be a string");
  if (!Array.isArray(raw.missing_sleeves)) throw new Error("Risk missing_sleeves must be an array");
  if (!Array.isArray(raw.violations)) throw new Error("Risk report violations must be an array");
  for (const v of raw.violations) {
    if (!v || typeof v !== "object") throw new Error("Violation must be an object");
    if (typeof (v as any).code !== "string" || typeof (v as any).message !== "string") {
      throw new Error("Violation must include string code/message");
    }
  }
  return raw;
}
