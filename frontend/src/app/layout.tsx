import "./globals.css";
import { ReactNode } from "react";
import { ErrorBoundary } from "@/components/error_boundary";
import { AppProviders } from "@/components/providers";

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <ErrorBoundary>
          <AppProviders>
            <div className="min-h-screen p-4">{children}</div>
          </AppProviders>
        </ErrorBoundary>
      </body>
    </html>
  );
}
