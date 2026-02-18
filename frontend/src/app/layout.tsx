import "./globals.css";
import { ReactNode } from "react";
import { ErrorBoundary } from "@/components/error_boundary";
import { AppProviders } from "@/components/providers";
import { AppShell } from "@/layout/AppShell";

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" data-theme="dark">
      <body>
        <ErrorBoundary>
          <AppProviders>
            <AppShell>{children}</AppShell>
          </AppProviders>
        </ErrorBoundary>
      </body>
    </html>
  );
}
