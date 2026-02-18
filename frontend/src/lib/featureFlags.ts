"use client";

export type FlagKey = "shellV2" | "executionV2" | "researchV2" | "runDetailV2" | "compareV2";
const allFlags: FlagKey[] = ["shellV2", "executionV2", "researchV2", "runDetailV2", "compareV2"];

export function getFlags(search = ""): Record<FlagKey, boolean> {
  const envProd = process.env.NEXT_PUBLIC_ENV === "prod";
  const defaults = Object.fromEntries(allFlags.map((f) => [f, !envProd])) as Record<FlagKey, boolean>;
  const ff = new URLSearchParams(search).get("ff");
  if (!ff) return defaults;
  ff.split(",").forEach((k) => {
    if (allFlags.includes(k as FlagKey)) defaults[k as FlagKey] = true;
  });
  return defaults;
}
