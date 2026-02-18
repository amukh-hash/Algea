"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { ReactNode, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/primitives";
import { useFeatureFlags } from "@/lib/feature_flags";
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
  const flags = useFeatureFlags();
  const [open, setOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const status = useConnectionSummary();
  const router = useRouter();

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
    if (path.startsWith("/runs/")) return ["Research", "Runs", path.split("/").at(-1) ?? "run"];
    if (path.startsWith("/compare")) return ["Research", "Compare"];
    if (path === "/execution") return ["Execution"];
    if (path === "/orchestrator") return ["Orchestrator"];
    if (path === "/settings") return ["Settings"];
    return ["Research"];
  }, [path]);

  if (!flags.shellV2) return <div className="min-h-screen p-4">{children}</div>;

  return (
    <div className="grid min-h-screen grid-cols-1 lg:grid-cols-[240px_1fr]" data-theme="dark">
      <aside className={`${open ? "block" : "hidden"} border-r border-border bg-surface-1 p-3 lg:block`}>
        <div className="mb-4 text-lg font-semibold">Algai Ops</div>
        <nav className="space-y-1">
          {nav.map((item) => (
            <Link key={item.href} href={item.href} className={`block rounded-md px-3 py-2 text-sm ${path.startsWith(item.href) ? "bg-surface-2 text-primary" : "text-secondary hover:bg-surface-2"}`}>
              {item.label}
            </Link>
          ))}
        </nav>
      </aside>
      <main className="h-screen overflow-auto min-w-0">
        <header className="sticky top-0 z-20 flex items-center justify-between border-b border-border bg-surface-1 px-4 py-2">
          <div className="flex items-center gap-2">
            <Button onClick={() => setOpen((o) => !o)} className="lg:hidden">☰</Button>
            <span className="text-sm text-secondary">{(process.env.NEXT_PUBLIC_ENV ?? "local").toUpperCase()}</span>
          </div>
          <div className="flex items-center gap-3 text-xs text-secondary">
            <span>{status.state === "open" ? "Live" : status.state}</span>
            <span>Last update: {status.lastUpdate ? new Date(status.lastUpdate).toLocaleTimeString() : "—"}</span>
            <Button onClick={() => setPaletteOpen(true)}>⌘K</Button>
          </div>
        </header>
        <div className="border-b border-border-subtle px-4 py-2 text-xs text-secondary">{crumbs.join(" / ")}</div>
        <div className="mx-auto max-w-7xl p-4">{children}</div>
      </main>
      {paletteOpen && <CommandPalette onClose={() => setPaletteOpen(false)} router={router} />}
    </div>
  );
}

function CommandPalette({ onClose, router }: { onClose: () => void; router: ReturnType<typeof useRouter> }) {
  const options = nav.map(n => ({ href: n.href, label: `Go to ${n.label}` }));
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowDown") setIndex((i) => Math.min(i + 1, options.length - 1));
      if (e.key === "ArrowUp") setIndex((i) => Math.max(i - 1, 0));
      if (e.key === "Enter") {
        router.push(options[index].href);
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, options, index, router]);

  return (
    <div className="fixed inset-0 z-50 bg-black/50 p-6" onClick={onClose}>
      <div className="mx-auto max-w-xl rounded-md border border-border bg-surface-2 p-2" onClick={(e) => e.stopPropagation()}>
        <p className="mb-2 px-2 text-sm text-secondary">Command palette</p>
        <div className="space-y-1">
          {options.map((opt, i) => (
            <button
              key={opt.href}
              className={`block w-full rounded px-3 py-2 text-left text-sm ${i === index ? "bg-surface-1 text-primary" : "text-secondary"}`}
              onClick={() => { router.push(opt.href); onClose(); }}
              onMouseEnter={() => setIndex(i)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
