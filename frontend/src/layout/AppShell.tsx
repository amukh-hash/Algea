"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { ReactNode, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/primitives";
import { useConnectionSummary } from "@/realtime/useRunStream";
import { CommandPalette } from "@/components/CommandPalette";
import { SystemHealthBar } from "@/components/SystemHealthBar";

const nav = [
  { href: "/overview", label: "Overview", key: "1" },
  { href: "/sleeves", label: "Sleeves", key: "2" },
  { href: "/performance", label: "Performance", key: "3" },
  { href: "/risk", label: "Risk", key: "4" },
  { href: "/inputs", label: "Inputs", key: "5" },
  { href: "/jobs", label: "Jobs", key: "6" },
  { href: "/artifacts", label: "Artifacts", key: "7" },
  { href: "/execution", label: "Execution", key: "8" },
  { href: "/orchestrator", label: "Orchestrator", key: "9" },
];

export function AppShell({ children }: { children: ReactNode }) {
  const path = usePathname();
  const [open, setOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const router = useRouter();
  const [appDisplay, setAppDisplay] = useState("Algae 4.0");

  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
    fetch(`${base}/meta`)
      .then((r) => (r.ok ? r.json() : null))
      .then((meta) => {
        if (meta?.display) setAppDisplay(String(meta.display));
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Don't fire shortcuts if typing in an input
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((v) => !v);
        return;
      }

      // Number keys for quick navigation (no modifier)
      if (!e.metaKey && !e.ctrlKey && !e.altKey) {
        const navItem = nav.find((n) => n.key === e.key);
        if (navItem) {
          e.preventDefault();
          router.push(navItem.href);
          return;
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [router]);

  const crumbs = useMemo(() => {
    if (path.startsWith("/runs/")) return ["Research", "Runs", path.split("/").at(-1) ?? "run"];
    if (path.startsWith("/overview") || path.startsWith("/ops")) return ["Overview / Ops Health"];
    if (path.startsWith("/sleeves")) return ["Sleeves Dashboard"];
    if (path.startsWith("/performance")) return ["Portfolio Performance"];
    if (path.startsWith("/risk")) return ["Risk & Constraints"];
    if (path.startsWith("/inputs")) return ["Data / Market Inputs"];
    if (path.startsWith("/jobs")) return ["Jobs / Orchestrator"];
    if (path.startsWith("/artifacts")) return ["Artifacts / Raw"];
    if (path === "/execution") return ["Execution"];
    if (path === "/orchestrator") return ["Orchestrator"];
    return ["Research"];
  }, [path]);

  if (path.startsWith("/tearoff")) {
    return <div className="min-h-screen p-4 bg-surface-1" data-theme="dark">{children}</div>;
  }

  return (
    <div className="grid min-h-screen grid-cols-1 lg:grid-cols-[240px_1fr]" data-theme="dark">
      <aside className={`${open ? "block" : "hidden"} border-r border-border bg-surface-1 p-3 lg:block`}>
        <div className="mb-4 text-lg font-semibold">{appDisplay} Ops</div>
        <nav className="space-y-1">
          {nav.map((item) => (
            <Link key={item.href} href={item.href} className={`block rounded-md px-3 py-2 text-sm ${path.startsWith(item.href) ? "bg-surface-2 text-primary" : "text-secondary hover:bg-surface-2"}`}>
              <span className="mr-2 text-xs text-muted font-mono">{item.key}</span>
              {item.label}
            </Link>
          ))}
        </nav>
        <div className="mt-6 border-t border-border pt-3">
          <div className="text-[0.6rem] text-muted uppercase tracking-widest mb-1">Shortcuts</div>
          <div className="text-[0.6rem] text-secondary space-y-0.5">
            <div>⌘K — Command Palette</div>
            <div>1-9 — Navigate sections</div>
          </div>
        </div>
      </aside>
      <main className="h-screen overflow-auto min-w-0">
        <header className="sticky top-0 z-20 flex items-center justify-between border-b border-border bg-surface-1 px-4 py-2">
          <div className="flex items-center gap-2">
            <Button onClick={() => setOpen((o) => !o)} className="lg:hidden">☰</Button>
            <span className="text-sm text-secondary">{(process.env.NEXT_PUBLIC_ENV ?? "local").toUpperCase()}</span>
          </div>
          <div className="flex-1 mx-4">
            <SystemHealthBar />
          </div>
          <div className="flex items-center gap-2">
            <Button onClick={() => setPaletteOpen(true)}>⌘K</Button>
          </div>
        </header>
        <div className="border-b border-border-subtle px-4 py-2 text-xs text-secondary">{crumbs.join(" / ")}</div>
        <div className="mx-auto max-w-7xl p-4">{children}</div>
        <footer className="border-t border-border-subtle px-4 py-2 text-xs text-muted">{appDisplay}</footer>
      </main>
      {paletteOpen && <CommandPalette onClose={() => setPaletteOpen(false)} />}
    </div>
  );
}
