"use client";

import { useSearchParams } from "next/navigation";

export type FlagKey = "shellV2" | "executionV2" | "researchV2" | "runDetailV2" | "compareV2";

const allFlags: FlagKey[] = ["shellV2", "executionV2", "researchV2", "runDetailV2", "compareV2"];

export function useFeatureFlags() {
  const params = useSearchParams();
  const ff = new Set((params.get("ff") ?? "").split(",").filter(Boolean));
  const env = process.env.NEXT_PUBLIC_ENV ?? "local";
  const prodDefault = env === "prod";
  return Object.fromEntries(allFlags.map((k) => [k, ff.has(k) || !prodDefault])) as Record<FlagKey, boolean>;
}
