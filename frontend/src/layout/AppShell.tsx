"use client";

import Link from "next/link";
import { ReactNode, useEffect, useMemo, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { Button } from "@/components/ui/primitives";
import { useFeatureFlags } from "@/lib/feature_flags";

const nav = [
  { href: "/execution", label: "Execution" },
  { href: "/research", label: "Research" },
  { href: "/compare", label: "Compare" },
  { href: "/settings", label: "Settings" },
];

export function AppShell({ children }: { children: ReactNode }) {
  const path = usePathname();
  const flags = useFeatureFlags();
  const [open, setOpen] = useState(false);
  const [palette, setPalette] = useState(false);
  const router = useRouter();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPalette((p) => !p);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const crumbs = useMemo(() => {
    if (path.startsWith("/runs/")) return ["Research", "Runs", path.split("/").at(-1) ?? "run"];
    if (path.startsWith("/compare")) return ["Research", "Compare"];
    return [];
  }, [path]);

  if (!flags.shellV2) return <div className="min-h-screen p-4">{children}</div>;

  return (
    <div className="grid min-h-screen grid-cols-1 md:grid-cols-[240px_1fr]">
      <aside className={`border-r border-border bg-surface-1 p-3 ${open ? "block" : "hidden md:block"}`}>
        <div className="mb-4 text-lg font-semibold">Algai Ops</div>
        <nav className="space-y-1">
          {nav.map((item) => (
            <Link key={item.href} href={item.href} className={`block rounded-md px-3 py-2 text-sm ${path === item.href ? "bg-surface-2 text-primary" : "text-secondary hover:bg-surface-2"}`}>
              {item.label}
            </Link>
          ))}
        </nav>
      </aside>
      <div className="min-w-0">
        <header className="sticky top-0 z-20 flex items-center justify-between border-b border-border bg-app/95 px-4 py-2 backdrop-blur">
          <div className="flex items-center gap-2"><Button onClick={() => setOpen((o) => !o)} className="md:hidden">☰</Button><span className="text-sm text-secondary">{(process.env.NEXT_PUBLIC_ENV ?? "local").toUpperCase()}</span></div>
          <div className="flex items-center gap-3"><Button onClick={() => setPalette(true)}>⌘K</Button><span className="rounded-full border border-success/40 px-2 py-1 text-xs text-success">Live</span></div>
        </header>
        <main className="space-y-4 p-4">
          {crumbs.length > 0 && <div className="text-xs text-secondary">{crumbs.join(" / ")}</div>}
          {children}
        </main>
      </div>
      {palette && (
        <div className="fixed inset-0 z-50 bg-black/40 p-4" onClick={() => setPalette(false)}>
          <div className="mx-auto mt-20 max-w-lg rounded-lg border border-border bg-surface-1 p-3" onClick={(e) => e.stopPropagation()}>
            <p className="mb-2 text-sm text-secondary">Command palette</p>
            <div className="space-y-2">
              {nav.map((item) => <button key={item.href} className="block w-full rounded p-2 text-left hover:bg-surface-2" onClick={() => { router.push(item.href); setPalette(false); }}>{item.label}</button>)}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
