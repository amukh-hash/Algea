"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import type { ConnectionState } from "./types";

const BACKOFF_BASE_MS = 1_000;
const BACKOFF_MAX_MS = 30_000;
const MAX_RETRIES = 20;

/**
 * Hardened EventSource hook with exponential backoff reconnect,
 * retry counting, and rehydrating state.
 */
export function useEventSource(
  url: string | null,
  handlers: Record<string, (event: MessageEvent) => void>,
) {
  const [state, setState] = useState<ConnectionState>("connecting");
  const [lastMessageAt, setLastMessageAt] = useState<number | null>(null);
  const [reconnectCount, setReconnectCount] = useState(0);
  const sourceRef = useRef<EventSource | null>(null);
  const retriesRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const handlersRef = useRef(handlers);
  handlersRef.current = handlers;

  const connect = useCallback(() => {
    if (!url) return;

    setState(retriesRef.current === 0 ? "connecting" : "reconnecting");
    const source = new EventSource(url);
    sourceRef.current = source;

    source.onopen = () => {
      setState("open");
      retriesRef.current = 0;
    };

    source.onerror = () => {
      source.close();
      sourceRef.current = null;

      if (retriesRef.current >= MAX_RETRIES) {
        setState("error");
        return;
      }

      const delay = Math.min(
        BACKOFF_BASE_MS * 2 ** retriesRef.current,
        BACKOFF_MAX_MS,
      );
      retriesRef.current += 1;
      setReconnectCount((c) => c + 1);
      setState("reconnecting");

      timerRef.current = setTimeout(() => {
        connect();
      }, delay);
    };

    Object.entries(handlersRef.current).forEach(([name, cb]) => {
      source.addEventListener(name, (evt) => {
        setLastMessageAt(Date.now());
        cb(evt as MessageEvent);
      });
    });
  }, [url]);

  useEffect(() => {
    retriesRef.current = 0;
    connect();

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      if (sourceRef.current) sourceRef.current.close();
      setState("closed");
    };
  }, [connect]);

  return { state, lastMessageAt, reconnectCount };
}
