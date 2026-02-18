"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ReactNode, useEffect, useMemo, useState } from "react";
import { useConnectionSummary } from "@/realtime/useRunStream";

const nav = [
  { href: "/execution", label: "Execution" },
  { href: "/orchestrator", label: "Orchestrator" },
  { href: "/research", label: "Research" },
  { href: "/compare", label: "Compare" },
  { href: "/settings", label: "Settings" },
];

export function AppShell({ children }: { children: ReactNode }) {
  const path = usePathname();
  const [open, setOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const status = useConnectionSummary();
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const crumbs = useMemo(() => {
    if (path.startsWith("/runs/")) return ["Research", "Runs", path.split("/").at(-1) ?? ""];
    if (path.startsWith("/compare")) return ["Research", "Compare"];
    return [path === "/execution" ? "Execution" : "Research"];
  }, [path]);

  return (
    <div className="grid min-h-screen grid-cols-1 lg:grid-cols-[240px_1fr]" data-theme="dark">
      <aside className={`${open ? "block" : "hidden"} border-r border-border bg-surface-1 p-4 lg:block`}>
        <div className="mb-4 text-sm font-semibold">Trading Ops</div>
        <nav className="space-y-1">
          {nav.map((item) => (
            <Link key={item.href} href={item.href} className={`block rounded px-3 py-2 text-sm ${path.startsWith(item.href) ? "bg-surface-2 text-primary" : "text-secondary"}`}>
              {item.label}
            </Link>
          ))}
        </nav>
      </aside>
      <main className="h-screen overflow-auto">
        <header className="sticky top-0 z-20 flex items-center justify-between border-b border-border bg-surface-1 px-4 py-3">
          <div className="flex items-center gap-2">
            <button className="rounded border border-border px-2 lg:hidden" onClick={() => setOpen((v) => !v)} aria-label="Toggle navigation">☰</button>
            <span className="text-sm font-semibold">ALGai Ops</span>
            <span className="rounded bg-surface-2 px-2 text-xs">{(process.env.NEXT_PUBLIC_ENV ?? "local").toUpperCase()}</span>
          </div>
          <div className="flex items-center gap-3 text-xs text-secondary">
            <span>{status.state === "open" ? "Live" : status.state}</span>
            <span>Last update: {status.lastUpdate ? new Date(status.lastUpdate).toLocaleTimeString() : "—"}</span>
            <button className="rounded border border-border px-2 py-1" onClick={() => setPaletteOpen(true)}>⌘K</button>
          </div>
        </header>
        <div className="border-b border-border-subtle px-4 py-2 text-xs text-muted">{crumbs.join(" / ")}</div>
        <div className="mx-auto max-w-7xl p-4">{children}</div>
      </main>
      {paletteOpen && <CommandPalette onClose={() => setPaletteOpen(false)} />}
    </div>
  );
}

function CommandPalette({ onClose }: { onClose: () => void }) {
  const options = [
    { href: "/execution", label: "Go to Execution" },
    { href: "/research", label: "Go to Research" },
    { href: "/compare", label: "Go to Compare" },
  ];
  const [index, setIndex] = useState(0);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowDown") setIndex((i) => Math.min(i + 1, options.length - 1));
      if (e.key === "ArrowUp") setIndex((i) => Math.max(i - 1, 0));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, options.length]);

  return (
    <div className="fixed inset-0 z-50 bg-black/50 p-6" onClick={onClose}>
      <div className="mx-auto max-w-xl rounded-md border border-border bg-surface-2 p-2" onClick={(e) => e.stopPropagation()}>
        {options.map((opt, i) => (
          <Link key={opt.href} href={opt.href} className={`block rounded px-3 py-2 ${i === index ? "bg-surface-1" : ""}`}>{opt.label}</Link>
        ))}
      </div>
    </div>
  );
}
