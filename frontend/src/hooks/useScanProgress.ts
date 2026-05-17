// Scan progress hook combining WebSocket and polling

import { useState, useEffect, useCallback } from 'react';
import { useWebSocket } from './useWebSocket';
import type { ScanStatus } from '../types';
import api from '../services/api';

export function useScanProgress(scanId: number | null) {
  const { progress: wsProgress, errors, isConnected, clearErrors } = useWebSocket(scanId);
  const [scanStatus, setScanStatus] = useState<ScanStatus | null>(null);
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fallback: poll status if WebSocket is not connected
  useEffect(() => {
    if (!scanId || isConnected) return;

    const interval = setInterval(async () => {
      try {
        const status = await api.getScanStatus(scanId);
        setScanStatus(status);
        if (status.status === 'completed' || status.status === 'failed' || status.status === 'stopped') {
          setIsComplete(true);
          if (status.status === 'failed') {
            setError(status.error_message);
          }
          clearInterval(interval);
        }
      } catch (e: any) {
        console.error('Polling error:', e);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [scanId, isConnected]);

  // Update from WebSocket
  useEffect(() => {
    if (!wsProgress) return;

    setScanStatus({
      id: wsProgress.scan_id,
      root_path: '',
      status: wsProgress.status,
      total_files: wsProgress.total_files,
      scanned_files: wsProgress.scanned_files,
      current_file: wsProgress.current_file,
      current_stage: wsProgress.current_stage,
      progress_percent: wsProgress.progress_percent,
      duplicate_groups_found: wsProgress.duplicate_groups_found || 0,
      recoverable_space: wsProgress.recoverable_space || 0,
      started_at: null,
      completed_at: null,
      error_message: null,
    });

    if (wsProgress.type === 'complete') {
      setIsComplete(true);
    }
    if (wsProgress.status === 'stopped') {
      setIsComplete(true);
    }
    if (wsProgress.type === 'error') {
      setIsComplete(true);
      setError(wsProgress.message || 'Scan failed');
    }
  }, [wsProgress]);

  const reset = useCallback(() => {
    setScanStatus(null);
    setIsComplete(false);
    setError(null);
  }, []);

  return {
    scanStatus,
    wsProgress,
    errors,
    clearErrors,
    isConnected,
    isComplete,
    error,
    reset,
  };
}
