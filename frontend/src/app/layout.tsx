import "./globals.css";
import { ReactNode } from "react";
import { AppProviders } from "@/components/providers";

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <AppProviders>
          <div className="min-h-screen p-4">{children}</div>
        </AppProviders>
      </body>
    </html>
  );
}
