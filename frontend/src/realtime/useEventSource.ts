"use client";

import { useEffect, useState } from "react";
import { ConnectionState } from "./types";

export function useEventSource(url: string | null, onEvent: (event: MessageEvent) => void) {
  const [state, setState] = useState<ConnectionState>("connecting");
  const [lastMessageAt, setLastMessageAt] = useState<number | null>(null);

  useEffect(() => {
    if (!url) return;
    const source = new EventSource(url);
    source.onopen = () => setState("open");
    source.onerror = () => setState("reconnecting");
    source.onmessage = (evt) => {
      setLastMessageAt(Date.now());
      onEvent(evt);
    };
    const named = (evt: MessageEvent) => {
      setLastMessageAt(Date.now());
      onEvent(evt);
    };
    source.addEventListener("metric", named as EventListener);
    source.addEventListener("event", named as EventListener);
    source.addEventListener("status", named as EventListener);
    source.addEventListener("heartbeat", named as EventListener);
    return () => {
      setState("closed");
      source.close();
    };
  }, [url, onEvent]);

  return { state, lastMessageAt };
}
