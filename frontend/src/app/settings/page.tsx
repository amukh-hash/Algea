"use client";

import { getFlags } from "@/lib/featureFlags";

export default function SettingsPage() {
  const flags = getFlags(typeof window !== "undefined" ? window.location.search : "");
  return (
    <div className="space-y-3">
      <h1 className="text-xl font-semibold">Settings</h1>
      <div className="rounded border border-border bg-surface-1 p-3">
        <h2 className="mb-2 text-sm font-semibold">Feature flags</h2>
        <ul className="space-y-1 text-sm text-secondary">
          {Object.entries(flags).map(([k, v]) => <li key={k}>{k}: {String(v)}</li>)}
        </ul>
      </div>
    </div>
  );
}
