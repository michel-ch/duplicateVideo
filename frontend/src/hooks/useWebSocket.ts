// WebSocket hook for real-time scan progress

import { useEffect, useRef, useState, useCallback } from 'react';
import type { ScanProgressMessage, ScanErrorLogEntry } from '../types';

// Match the page's scheme so the WS works behind HTTPS too.
const WS_BASE = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/api`;

// Cap the in-memory error history per scan to bound memory + render cost.
const MAX_ERRORS = 200;

export function useWebSocket(scanId: number | null) {
  const [progress, setProgress] = useState<ScanProgressMessage | null>(null);
  const [errors, setErrors] = useState<ScanErrorLogEntry[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Ref so the onclose handler always sees the latest progress without a
  // stale closure (connect is memoised on scanId only).
  const progressRef = useRef<ScanProgressMessage | null>(null);
  // Track which scan the current errors belong to so we reset cleanly
  // when the parent component switches to a new active scan.
  const errorsScanIdRef = useRef<number | null>(null);

  const connect = useCallback(() => {
    if (!scanId) return;

    // New scan → drop the previous scan's error history.  The backend
    // replays its in-memory log on connect, so anything still relevant
    // for the new scan will arrive shortly.
    if (errorsScanIdRef.current !== scanId) {
      errorsScanIdRef.current = scanId;
      setErrors([]);
    }

    const ws = new WebSocket(`${WS_BASE}/scan/${scanId}/ws`);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data?.type === 'error_log') {
          const entry = data as ScanErrorLogEntry;
          setErrors(prev => {
            const next = [...prev, entry];
            return next.length > MAX_ERRORS ? next.slice(next.length - MAX_ERRORS) : next;
          });
          return;
        }
        if (data?.type === 'pong') return;
        const msg = data as ScanProgressMessage;
        progressRef.current = msg;
        setProgress(msg);
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      // Reconnect after a short delay if scan is still running.
      // Use the ref (not the closure variable) to always see the latest status.
      const last = progressRef.current;
      if (last && last.status !== 'completed' && last.status !== 'failed' && last.status !== 'stopped') {
        reconnectRef.current = setTimeout(() => connect(), 2000);
      }
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [scanId]);

  useEffect(() => {
    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectRef.current) {
        clearTimeout(reconnectRef.current);
      }
    };
  }, [connect]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }
  }, []);

  const clearErrors = useCallback(() => setErrors([]), []);

  return { progress, errors, isConnected, disconnect, clearErrors };
}
