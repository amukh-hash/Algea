"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";

interface Props { children?: ReactNode }
interface State { hasError: boolean; error?: Error }

export class ErrorBoundary extends Component<Props, State> {
  public state: State = { hasError: false };

  public static getDerivedStateFromError(error: Error): State { return { hasError: true, error }; }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) { console.error("Uncaught error:", error, errorInfo); }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="flex min-h-screen items-center justify-center bg-app p-8 text-primary">
          <div className="max-w-xl rounded-lg border border-danger/40 bg-surface-1 p-6">
            <h1 className="text-xl font-semibold text-danger">Something went wrong</h1>
            <p className="mt-2 text-sm text-secondary">A runtime error interrupted this page.</p>
            <pre className="mt-3 overflow-auto rounded bg-surface-2 p-2 text-xs">{this.state.error?.toString()}</pre>
            <div className="mt-4 flex gap-2">
              <button className="rounded bg-info px-3 py-2 text-sm text-app" onClick={() => window.location.reload()}>Reload</button>
              <button className="rounded border border-border px-3 py-2 text-sm" onClick={() => navigator.clipboard.writeText(this.state.error?.stack ?? this.state.error?.message ?? "")}>Copy error details</button>
            </div>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
