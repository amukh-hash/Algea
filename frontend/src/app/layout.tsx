import "./globals.css";
import { ReactNode } from "react";
import { Metadata } from "next";
import { ErrorBoundary } from "@/components/error_boundary";
import { AppProviders } from "@/components/providers";
import { AppShell } from "@/layout/AppShell";

export const metadata: Metadata = {
  title: "Algae 4.0",
  description: "Algae 4.0 operations dashboard",
};

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
