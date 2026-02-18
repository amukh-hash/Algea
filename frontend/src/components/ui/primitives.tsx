"use client";

import { HTMLAttributes, ReactNode, useMemo, useState } from "react";

export function Button({ className = "", variant = "secondary", ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "ghost" | "destructive" }) {
  const style = variant === "primary" ? "bg-info text-black" : variant === "destructive" ? "bg-error text-white" : variant === "ghost" ? "bg-transparent" : "bg-surface-2";
  return <button {...props} className={`rounded-md border border-border px-3 py-2 text-sm ${style} ${className}`} />;
}

export function Card({ className = "", ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div {...props} className={`rounded-md border border-border bg-surface-1 p-4 ${className}`} />;
}

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: string; actions?: ReactNode }) {
  return <div className="mb-4 flex items-start justify-between"><div><h1 className="text-2xl font-semibold">{title}</h1>{subtitle && <p className="text-sm text-secondary">{subtitle}</p>}</div><div className="flex gap-2">{actions}</div></div>;
}

export function StatusBadge({ status }: { status: string }) {
  const map: Record<string, string> = { running: "text-info", completed: "text-success", ok: "text-success", warning: "text-warning", error: "text-error", paused: "text-warning" };
  return <span className={`inline-flex items-center gap-1 rounded bg-surface-2 px-2 py-1 text-xs ${map[status] ?? "text-secondary"}`}>● {status}</span>;
}

export function EmptyState({ title, message, cta }: { title: string; message: string; cta?: ReactNode }) {
  return <Card className="text-center"><h3 className="text-lg">{title}</h3><p className="mt-1 text-sm text-secondary">{message}</p>{cta && <div className="mt-3">{cta}</div>}</Card>;
}

export function Skeleton({ className = "h-20" }: { className?: string }) {
  return <div className={`animate-pulse rounded-md bg-surface-2 ${className}`} />;
}

export function Tabs({ items, active, onChange }: { items: { id: string; label: string; panel: ReactNode }[]; active: string; onChange: (id: string) => void }) {
  const activeIndex = useMemo(() => Math.max(0, items.findIndex((i) => i.id === active)), [items, active]);
  const [focusIndex, setFocusIndex] = useState(activeIndex);
  return (
    <div>
      <div role="tablist" aria-label="Run detail tabs" className="flex gap-2 border-b border-border pb-2">
        {items.map((tab, i) => (
          <button
            key={tab.id}
            id={`tab-${tab.id}`}
            role="tab"
            aria-selected={tab.id === active}
            aria-controls={`panel-${tab.id}`}
            tabIndex={i === focusIndex ? 0 : -1}
            onKeyDown={(e) => {
              if (e.key === "ArrowRight") setFocusIndex(Math.min(i + 1, items.length - 1));
              if (e.key === "ArrowLeft") setFocusIndex(Math.max(i - 1, 0));
              if (e.key === " " || e.key === "Enter") onChange(tab.id);
            }}
            onClick={() => onChange(tab.id)}
            className={`rounded px-3 py-1.5 text-sm ${tab.id === active ? "bg-surface-2" : "text-secondary"}`}
          >{tab.label}</button>
        ))}
      </div>
      {items.map((tab) => tab.id === active ? <div key={tab.id} id={`panel-${tab.id}`} role="tabpanel" aria-labelledby={`tab-${tab.id}`} className="pt-3">{tab.panel}</div> : null)}
    </div>
  );
}

export function SearchInput(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={`rounded-md border border-border bg-surface-2 px-3 py-2 text-sm ${props.className ?? ""}`} />;
}
