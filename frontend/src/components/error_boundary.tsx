"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";

interface Props { children?: ReactNode; }
interface State { hasError: boolean; error?: Error; stack?: string | null; }

export class ErrorBoundary extends Component<Props, State> {
  public state: State = { hasError: false };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({ stack: errorInfo.componentStack });
    console.error("Uncaught error:", error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-app p-8 text-primary">
          <h1 className="text-2xl font-bold">Something went wrong</h1>
          <p className="mt-2 text-secondary">The UI crashed unexpectedly.</p>
          <pre className="mt-4 overflow-auto rounded border border-border bg-surface-1 p-3 text-xs">{this.state.error?.toString()}</pre>
          <div className="mt-3 flex gap-2">
            <button className="rounded border border-border px-3 py-2" onClick={() => window.location.reload()}>Reload</button>
            <button className="rounded border border-border px-3 py-2" onClick={() => navigator.clipboard.writeText(`${this.state.error}\n${this.state.stack ?? ""}`)}>Copy error details</button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
