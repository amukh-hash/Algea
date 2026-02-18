"use client";

import { createContext, ReactNode, useContext, useMemo, useState } from "react";

type Toast = { id: number; message: string };

const Ctx = createContext<{ push: (message: string) => void } | null>(null);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toast, setToast] = useState<Toast | null>(null);
  const value = useMemo(() => ({ push: (message: string) => setToast({ id: Date.now(), message }) }), []);
  return (
    <Ctx.Provider value={value}>
      {children}
      {toast && (
        <div className="fixed bottom-4 right-4 z-50 rounded-md border border-border bg-surface-1 px-3 py-2 text-sm">
          {toast.message}
          <button className="ml-3 text-info" onClick={() => setToast(null)}>Dismiss</button>
        </div>
      )}
    </Ctx.Provider>
  );
}

export function useToast() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useToast must be used inside ToastProvider");
  return ctx;
}
