"use client";

import { createContext, ReactNode, useContext, useMemo, useState } from "react";

type Toast = { id: number; message: string; actionLabel?: string; onAction?: () => void };
const ToastCtx = createContext<{ push: (toast: Omit<Toast, "id"> | string) => void } | null>(null);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [queue, setQueue] = useState<Toast[]>([]);
  const current = queue[0];
  const value = useMemo(
    () => ({
      push: (toast: Omit<Toast, "id"> | string) => {
        const payload = typeof toast === "string" ? { message: toast } : toast;
        setQueue((q) => [...q, { ...payload, id: Date.now() }]);
      }
    }),
    []
  );

  return (
    <ToastCtx.Provider value={value}>
      {children}
      {current && (
        <div className="fixed bottom-4 right-4 z-50 rounded-md border border-border bg-surface-2 p-3 text-sm text-primary shadow">
          <div>{current.message}</div>
          {current.actionLabel && <button className="mt-2 text-info" onClick={current.onAction}>{current.actionLabel}</button>}
          <button className="ml-3 text-muted" onClick={() => setQueue((q) => q.slice(1))}>Dismiss</button>
        </div>
      )}
    </ToastCtx.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastCtx);
  if (!ctx) throw new Error("useToast must be used inside ToastProvider");
  return ctx;
}
