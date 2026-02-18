"use client";

import { ButtonHTMLAttributes, PropsWithChildren, ReactNode, useMemo } from "react";

export function cn(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

export function Button({ className, variant = "secondary", ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "ghost" | "destructive" }) {
  const styles = {
    primary: "bg-info text-app hover:opacity-90",
    secondary: "bg-surface-2 text-primary border border-border",
    ghost: "text-secondary hover:bg-surface-2",
    destructive: "bg-danger text-white",
  }[variant];
  return <button className={cn("rounded-md px-3 py-2 text-sm transition", styles, className)} {...props} />;
}

export function IconButton({ "aria-label": ariaLabel, className, ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button aria-label={ariaLabel} className={cn("rounded-md p-2 hover:bg-surface-2", className)} {...props} />;
}

export function Card({ children, className }: PropsWithChildren<{ className?: string }>) {
  return <section className={cn("rounded-lg border border-border bg-surface-1 p-4", className)}>{children}</section>;
}

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: string; actions?: ReactNode }) {
  return (
    <header className="flex flex-wrap items-center justify-between gap-3">
      <div>
        <h1 className="text-2xl font-semibold">{title}</h1>
        {subtitle && <p className="text-sm text-secondary">{subtitle}</p>}
      </div>
      {actions}
    </header>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const tone = useMemo(() => {
    if (["running", "ok", "completed"].includes(status)) return "text-success border-success/40";
    if (["error", "inputs_missing"].includes(status)) return "text-danger border-danger/50";
    if (["warning", "paused"].includes(status)) return "text-warning border-warning/40";
    return "text-secondary border-border";
  }, [status]);
  return <span className={cn("inline-flex items-center gap-2 rounded-full border px-2 py-1 text-xs", tone)}><span aria-hidden>●</span><span>{status}</span></span>;
}

export function EmptyState({ title, message, cta }: { title: string; message: string; cta?: ReactNode }) {
  return <Card className="text-center"><p className="font-medium">{title}</p><p className="mt-1 text-sm text-secondary">{message}</p>{cta && <div className="mt-3">{cta}</div>}</Card>;
}

export function Skeleton({ className = "h-12" }: { className?: string }) {
  return <div className={cn("animate-pulse rounded-md bg-surface-2", className)} />;
}
