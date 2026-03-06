// WebSocket hook for real-time scan progress

import { useEffect, useRef, useState, useCallback } from 'react';
import type { ScanProgressMessage } from '../types';

const WS_BASE = `ws://${window.location.host}/api`;

export function useWebSocket(scanId: number | null) {
  const [progress, setProgress] = useState<ScanProgressMessage | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Ref so the onclose handler always sees the latest progress without a
  // stale closure (connect is memoised on scanId only).
  const progressRef = useRef<ScanProgressMessage | null>(null);

  const connect = useCallback(() => {
    if (!scanId) return;

    const ws = new WebSocket(`${WS_BASE}/scan/${scanId}/ws`);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as ScanProgressMessage;
        progressRef.current = data;
        setProgress(data);
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

  return { progress, isConnected, disconnect };
}
