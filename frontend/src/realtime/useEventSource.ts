"use client";
import { useEffect, useRef, useState } from "react";
import type { ConnectionState } from "./types";

export function useEventSource(url: string | null, handlers: Record<string, (event: MessageEvent) => void>) {
  const [state, setState] = useState<ConnectionState>("connecting");
  const [lastMessageAt, setLastMessageAt] = useState<number | null>(null);
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!url) return;
    setState("connecting");
    const source = new EventSource(url);
    sourceRef.current = source;
    source.onopen = () => setState("open");
    source.onerror = () => setState("reconnecting");
    Object.entries(handlers).forEach(([name, cb]) => {
      source.addEventListener(name, (evt) => {
        setLastMessageAt(Date.now());
        cb(evt as MessageEvent);
      });
    });
    return () => {
      source.close();
      setState("closed");
    };
  }, [url]);

  return { state, lastMessageAt };
}
