"use client";

import { QueryClientProvider } from "@tanstack/react-query";
import { ReactNode, useEffect, useState } from "react";
import { createQueryClient } from "@/query/client";
import { ToastProvider } from "@/components/ui/toast";
import { resolveRuntimeConfig } from "@/runtime/config";

export function AppProviders({ children }: { children: ReactNode }) {
  const [client] = useState(() => createQueryClient());
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    resolveRuntimeConfig()
      .then(() => setReady(true))
      .catch((e) => {
        setError(e instanceof Error ? e.message : String(e));
        setReady(true);
      });
  }, []);

  if (!ready) {
    return <div className="flex min-h-screen items-center justify-center text-secondary">Starting local engine…</div>;
  }

  if (error) {
    return <div className="m-8 rounded border border-danger p-4">Runtime init failed: {error}</div>;
  }

  return (
    <QueryClientProvider client={client}>
      <ToastProvider>{children}</ToastProvider>
    </QueryClientProvider>
  );
}
