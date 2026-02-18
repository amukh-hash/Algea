import { QueryClient } from "@tanstack/react-query";

export function createQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        refetchOnWindowFocus: false,
        refetchOnReconnect: true,
        staleTime: 30_000,
        gcTime: 5 * 60_000,
        retry: 2,
        retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 15_000),
      },
    },
  });
}
