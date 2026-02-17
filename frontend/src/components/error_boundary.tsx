"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";

interface Props {
  children?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error:", error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen p-8 bg-zinc-950 text-red-500 font-mono">
          <h1 className="text-2xl font-bold mb-4">Application Crash</h1>
          <div className="p-4 border border-red-900 bg-red-950/30 rounded">
            <p className="font-semibold">{this.state.error?.toString()}</p>
            <p className="mt-2 text-sm text-zinc-500">Check console for details.</p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
