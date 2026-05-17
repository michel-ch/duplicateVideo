import React, { useState, useEffect, useCallback, useRef } from 'react';
import api from '../services/api';
import { useScanProgress } from '../hooks/useScanProgress';
import ProgressTracker from '../components/ProgressTracker';
import FolderBrowser from '../components/FolderBrowser';
import type { Stats, GPUStatus, ScanStatus } from '../types';
import { useNavigate } from 'react-router-dom';

function formatBytes(bytes: number): string {
  if (!bytes) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

const TERMINAL = ['completed', 'failed', 'stopped'];
const ACTIVE = ['pending', 'scanning', 'metadata', 'hashing', 'comparing', 'paused'];

export default function Dashboard() {
  const navigate = useNavigate();
  const [path, setPath] = useState('');
  const [stats, setStats] = useState<Stats | null>(null);
  const [gpuStatus, setGpuStatus] = useState<GPUStatus | null>(null);
  const [error, setError] = useState('');
  const [threshold, setThreshold] = useState(70);
  const [showOptions, setShowOptions] = useState(false);
  const [showBrowser, setShowBrowser] = useState(false);

  // Scan list
  const [scans, setScans] = useState<ScanStatus[]>([]);
  // Distinguish "haven't loaded yet" from "loaded an empty list" so the
  // first render after F5 shows a placeholder instead of just an empty page.
  const [scansLoaded, setScansLoaded] = useState(false);

  // Active scan tracking
  const activeScan = scans.find(s => ACTIVE.includes(s.status)) || null;
  const activeScanId = activeScan?.id ?? null;
  const isScanning = activeScanId !== null;

  const {
    scanStatus,
    wsProgress,
    errors: scanErrors,
    clearErrors,
    isComplete,
    error: scanError,
    reset,
  } = useScanProgress(activeScanId);

  // ── Load scans and stats ──────────────────────────────────────────────────
  // Tracks the prior scan list so we can detect ACTIVE→TERMINAL transitions
  // and refresh stats immediately — without waiting for the next poll tick.
  const prevScansRef = useRef<ScanStatus[]>([]);
  // Monotonically increasing request ID so out-of-order responses (a slow
  // /api/scans landing AFTER a faster one started later) can be discarded
  // — otherwise the stale response overwrites fresh data and rows appear
  // to "rewind" or vanish.
  const loadScansSeqRef = useRef(0);

  const loadScans = useCallback(async () => {
    const seq = ++loadScansSeqRef.current;
    try {
      const data = await api.listScans();
      // Drop late-arriving responses
      if (seq !== loadScansSeqRef.current) return;
      const newScans = data as ScanStatus[];
      const transitioned = newScans.some(ns => {
        const old = prevScansRef.current.find(p => p.id === ns.id);
        return !!old && ACTIVE.includes(old.status) && TERMINAL.includes(ns.status);
      });
      prevScansRef.current = newScans;
      setScans(newScans);
      setScansLoaded(true);
      if (transitioned) {
        api.getStats().then(setStats).catch((e) => console.error('getStats after transition:', e));
      }
    } catch (e) {
      // Don't blank out the UI on transient errors — keep the last-known list.
      // Just surface the failure so we can see it instead of hiding behind {}.
      console.error('loadScans failed:', e);
    }
  }, []);

  useEffect(() => {
    api.getStats().then(setStats).catch((e) => console.error('getStats:', e));
    api.getGpuStatus().then(setGpuStatus).catch((e) => console.error('getGpuStatus:', e));
    loadScans();

    // If the very first /api/scans is slow (backend cold-start), retry a
    // few times with backoff so the page doesn't sit empty for 1.5 s+
    // waiting for the normal poll tick.
    const t1 = setTimeout(() => { if (!scansLoaded) loadScans(); }, 250);
    const t2 = setTimeout(() => { if (!scansLoaded) loadScans(); }, 750);

    // Poll at 1.5 s so structural changes (new scans, queued→pending,
    // cancellations) appear quickly.  In-scan PROGRESS doesn't need the
    // poll — it streams in over the WebSocket and patches the local
    // scans array via the effect below.
    const interval = setInterval(() => {
      api.getStats().then(setStats).catch((e) => console.error('getStats:', e));
      loadScans();
    }, 1500);
    return () => {
      clearInterval(interval);
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, [loadScans]); // scansLoaded intentionally omitted — the timeouts check it themselves

  // ── Patch the local scans array from WebSocket progress ───────────────────
  // Without this, the row for the active scan in the scan list stays stale
  // (status badge, progress %, scanned/total counters) until the next 1.5 s
  // poll tick fires, even though the WebSocket is already streaming updates.
  useEffect(() => {
    if (!wsProgress?.scan_id) return;
    setScans(prev => {
      let changed = false;
      const next = prev.map(s => {
        if (s.id !== wsProgress.scan_id) return s;
        changed = true;
        return {
          ...s,
          status: wsProgress.status ?? s.status,
          total_files: wsProgress.total_files ?? s.total_files,
          scanned_files: wsProgress.scanned_files ?? s.scanned_files,
          current_file: wsProgress.current_file ?? s.current_file,
          current_stage: wsProgress.current_stage ?? s.current_stage,
          progress_percent: wsProgress.progress_percent ?? s.progress_percent,
          duplicate_groups_found: wsProgress.duplicate_groups_found ?? s.duplicate_groups_found,
          recoverable_space: wsProgress.recoverable_space ?? s.recoverable_space,
        };
      });
      return changed ? next : prev;
    });
  }, [wsProgress]);

  // ── Handle completion ─────────────────────────────────────────────────────
  // Refresh scans + stats BEFORE clearing the progress tracker.  Otherwise
  // reset() runs first → ProgressTracker vanishes → the user briefly sees
  // an empty page with stale stats until the next poll tick fills it back.
  useEffect(() => {
    if (!isComplete && !scanError) return;
    let cancelled = false;
    Promise.all([
      loadScans(),
      api.getStats().then(setStats).catch(() => {}),
    ]).finally(() => {
      if (!cancelled) reset();
    });
    return () => { cancelled = true; };
  }, [isComplete, scanError, loadScans, reset]);

  // ── Start a new scan ──────────────────────────────────────────────────────
  const handleStartScan = async () => {
    if (!path.trim()) {
      setError('Please enter a directory path');
      return;
    }
    setError('');

    try {
      await api.startScan(path.trim(), {
        similarity_threshold: threshold,
        duration_tolerance: 2.0,
        key_frames_count: 8,
        hash_threshold: 10,
      });
      setPath('');
      loadScans();
    } catch (e: any) {
      setError(e.message);
    }
  };

  // ── Pause / Resume / Stop ─────────────────────────────────────────────────
  const handlePause = async () => {
    if (!activeScanId) return;
    try {
      await api.pauseScan(activeScanId);
      loadScans();
    } catch (e: any) {
      console.error('Pause failed:', e);
    }
  };

  const handleResume = async () => {
    if (!activeScanId) return;
    try {
      await api.resumeScan(activeScanId);
      loadScans();
    } catch (e: any) {
      console.error('Resume failed:', e);
    }
  };

  const handleStop = async () => {
    if (!activeScanId) return;
    try {
      await api.stopScan(activeScanId);
      reset();
      loadScans();
    } catch (e: any) {
      console.error('Stop failed:', e);
    }
  };

  const handleCancel = async (scanId: number) => {
    try {
      await api.cancelScan(scanId);
      loadScans();
    } catch (e: any) {
      console.error('Cancel failed:', e);
    }
  };

  const handleDeleteScan = async (scanId: number) => {
    if (!window.confirm(`Delete scan #${scanId} from history?\n\nThis removes the scan and its duplicate groups.\nThe file cache (used to speed up future scans) is kept.`)) return;
    try {
      await api.deleteScan(scanId);
      await Promise.all([loadScans(), api.getStats().then(setStats).catch(() => {})]);
    } catch (e: any) {
      console.error('Delete scan failed:', e);
      setError(e.message || 'Failed to delete scan');
    }
  };

  const handleDeleteAllScans = async () => {
    if (!window.confirm('Delete ALL non-active scans from history?\n\nThis removes every queued and finished scan along with its duplicate groups.\nActive scans are left running. The file cache is kept.')) return;
    try {
      const res = await api.deleteAllScans();
      await Promise.all([loadScans(), api.getStats().then(setStats).catch(() => {})]);
      if (res.deleted_count === 0) {
        setError('No scans to delete.');
      }
    } catch (e: any) {
      console.error('Delete all scans failed:', e);
      setError(e.message || 'Failed to delete scans');
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const items = e.dataTransfer.items;
    if (items.length > 0) {
      const item = items[0];
      if (item.kind === 'file') {
        const file = item.getAsFile();
        if (file) {
          setPath((file as any).path || file.name);
        }
      }
    }
  }, []);

  // Split scans into categories
  const queuedScans = scans.filter(s => s.status === 'queued');
  const completedScans = scans.filter(s => TERMINAL.includes(s.status));

  return (
    <div>
      <div className="page-header">
        <h2>⚡ Mission Control</h2>
        <p>
          Scan directories to detect and eliminate duplicate videos
          {gpuStatus?.acceleration_active && (
            <span className="gpu-header-badge" title={`${gpuStatus.gpu_name} • ${gpuStatus.vram_total_mb} MB VRAM • ${gpuStatus.cuvid_decoders.length} CUVID decoders`}>
              ⚡ GPU Accelerated
            </span>
          )}
        </p>
      </div>

      {/* Stats */}
      {stats && (
        <div className="stats-grid">
          <div className="stat-card cyan">
            <div className="stat-label">Videos Indexed</div>
            <div className="stat-value cyan">{stats.total_videos.toLocaleString()}</div>
          </div>
          <div className="stat-card yellow">
            <div className="stat-label">Duplicate Groups</div>
            <div className="stat-value yellow">{stats.duplicate_groups.toLocaleString()}</div>
          </div>
          <div className="stat-card red">
            <div className="stat-label">Recoverable Space</div>
            <div className="stat-value red">{formatBytes(stats.recoverable_space)}</div>
          </div>
          <div className="stat-card green">
            <div className="stat-label">Space Recovered</div>
            <div className="stat-value green">{formatBytes(stats.space_recovered)}</div>
          </div>
          <div className="stat-card purple">
            <div className="stat-label">Total Scans</div>
            <div className="stat-value purple">{stats.total_scans}</div>
          </div>
          {gpuStatus?.acceleration_active && (
            <div className="stat-card gpu">
              <div className="stat-label">GPU Engine</div>
              <div className="stat-value gpu" style={{ fontSize: '1rem' }}>{gpuStatus.gpu_name}</div>
              <div className="gpu-detail">
                {gpuStatus.vram_total_mb} MB VRAM • {gpuStatus.cuvid_decoders.length} decoders
              </div>
            </div>
          )}
        </div>
      )}

      {/* Scan Input */}
      <div
        className="scan-section"
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        <div className="scan-title">
          <span className="icon">🔍</span>
          {isScanning ? 'Queue Another Scan' : 'Start New Scan'}
        </div>

        <div className="input-row">
          <div className="input-group" style={{ flex: 1 }}>
            <label className="input-label">Directory Path</label>
            <input
              type="text"
              className="input-field"
              placeholder="Enter or drop a directory path (e.g., C:\Users\Videos)"
              value={path}
              onChange={(e) => setPath(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleStartScan()}
            />
          </div>
          <button
            className="btn btn-secondary btn-lg"
            onClick={() => setShowBrowser(true)}
            title="Browse folders"
          >
            📂 Browse
          </button>
          <button
            className="btn btn-primary btn-lg"
            onClick={handleStartScan}
            disabled={!path.trim()}
          >
            {isScanning ? '📋 Add to Queue' : '🚀 Start Scan'}
          </button>
        </div>

        {/* Options toggle */}
        <div style={{ marginTop: 'var(--space-md)' }}>
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => setShowOptions(!showOptions)}
          >
            ⚙️ {showOptions ? 'Hide' : 'Show'} Options
          </button>
        </div>

        {showOptions && (
          <div className="scan-options-row">
            <div className="input-group">
              <label className="input-label">Similarity Threshold ({threshold}%)</label>
              <input
                type="range"
                className="range-slider"
                min={50}
                max={100}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
              />
            </div>
          </div>
        )}

        {error && (
          <div style={{
            marginTop: 'var(--space-md)',
            padding: 'var(--space-md)',
            background: 'var(--accent-red-dim)',
            border: '1px solid rgba(255,61,87,0.3)',
            borderRadius: 'var(--radius-sm)',
            color: 'var(--accent-red)',
            fontSize: '0.88rem',
          }}>
            ⚠️ {error}
          </div>
        )}
      </div>

      {/* Active Scan Progress */}
      {activeScan && (
        <ProgressTracker
          status={scanStatus}
          wsProgress={wsProgress}
          onPause={handlePause}
          onResume={handleResume}
          onStop={handleStop}
        />
      )}

      {/* First-load placeholder so a hard refresh doesn't show an empty
          page while /api/scans is still in flight (backend cold-start,
          slow DB, etc.).  Replaced by the real lists as soon as the
          first response lands. */}
      {!scansLoaded && (
        <div
          className="card"
          style={{
            marginTop: 'var(--space-lg)',
            padding: 'var(--space-md) var(--space-lg)',
            display: 'flex',
            alignItems: 'center',
            gap: 'var(--space-sm)',
            color: 'var(--text-secondary)',
          }}
        >
          <span className="spinner" />
          Loading scans…
        </div>
      )}

      {/* Backend error stream — per-file failures that previously only
          went to stdout (frame-extract timeouts, codec errors, etc.).
          Persists briefly after a scan ends so the user can still see what
          went wrong on the just-finished run. */}
      {scanErrors.length > 0 && (
        <div className="card" style={{ marginTop: 'var(--space-lg)', padding: 'var(--space-md) var(--space-lg)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 'var(--space-sm)' }}>
            <h3 style={{ fontFamily: 'var(--font-mono)', margin: 0, color: 'var(--accent-red)' }}>
              ⚠️ Scan Errors ({scanErrors.length})
            </h3>
            <button
              className="btn btn-ghost btn-sm"
              onClick={clearErrors}
              title="Clear the error list (does not affect the scan)"
            >
              Clear
            </button>
          </div>
          <div
            style={{
              maxHeight: 280,
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: 4,
              fontFamily: 'var(--font-mono)',
              fontSize: '0.82rem',
            }}
          >
            {scanErrors.slice().reverse().map((e, i) => {
              const time = (() => {
                try { return new Date(e.timestamp).toLocaleTimeString(); }
                catch { return ''; }
              })();
              const stageBadgeClass =
                e.stage === 'metadata' ? 'badge-yellow' :
                e.stage === 'hashing' ? 'badge-red' :
                e.stage === 'audio_fp' ? 'badge-cyan' :
                e.stage === 'pipeline' ? 'badge-red' :
                'badge-muted';
              return (
                <div
                  key={`${e.timestamp}-${i}`}
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 'var(--space-sm)',
                    padding: '4px 6px',
                    borderLeft: '2px solid var(--accent-red-dim, rgba(255,61,87,0.3))',
                    background: 'rgba(255,61,87,0.04)',
                  }}
                >
                  <span style={{ color: 'var(--text-secondary)', minWidth: 64 }}>{time}</span>
                  <span className={`badge ${stageBadgeClass}`} style={{ minWidth: 80, textAlign: 'center' }}>
                    {e.stage}
                  </span>
                  <span style={{ flex: 1, minWidth: 0 }}>
                    {e.file_path && (
                      <span
                        className="truncate"
                        style={{ display: 'block', color: 'var(--text-primary)' }}
                        title={e.file_path}
                      >
                        {e.file_path}
                      </span>
                    )}
                    <span style={{ color: 'var(--accent-red)' }}>{e.message}</span>
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Queued Scans */}
      {queuedScans.length > 0 && (
        <div style={{ marginTop: 'var(--space-lg)' }}>
          <h3 style={{ fontFamily: 'var(--font-mono)', marginBottom: 'var(--space-md)', color: 'var(--text-secondary)' }}>
            📋 Queued ({queuedScans.length})
          </h3>
          <div className="flex-col gap-sm">
            {queuedScans.map((scan) => (
              <div key={scan.id} className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: 'var(--space-md) var(--space-lg)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)', flex: 1, minWidth: 0 }}>
                  <span className="badge badge-muted">#{scan.id}</span>
                  <span className="text-mono text-sm truncate" style={{ flex: 1 }} title={scan.root_path}>
                    {scan.root_path}
                  </span>
                  <span className="badge badge-yellow">Queued</span>
                </div>
                <button
                  className="btn btn-danger btn-sm"
                  onClick={() => handleCancel(scan.id)}
                  style={{ marginLeft: 'var(--space-md)' }}
                >
                  ✕ Cancel
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Completed Scans */}
      {completedScans.length > 0 && (
        <div style={{ marginTop: 'var(--space-lg)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 'var(--space-md)' }}>
            <h3 style={{ fontFamily: 'var(--font-mono)', margin: 0, color: 'var(--text-secondary)' }}>
              📜 Recent Scans ({completedScans.length})
            </h3>
            <button
              className="btn btn-danger btn-sm"
              onClick={handleDeleteAllScans}
              title="Delete every non-active scan from history"
            >
              🗑 Clear All
            </button>
          </div>
          <div className="flex-col gap-sm">
            {completedScans.slice(0, 10).map((scan) => (
              <div key={scan.id} className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: 'var(--space-md) var(--space-lg)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)', flex: 1, minWidth: 0 }}>
                  <span className="badge badge-muted">#{scan.id}</span>
                  <span className="text-mono text-sm truncate" style={{ flex: 1 }} title={scan.root_path}>
                    {scan.root_path}
                  </span>
                  <span className={`badge ${
                    scan.status === 'completed' ? 'badge-green' :
                    scan.status === 'failed' ? 'badge-red' :
                    'badge-yellow'
                  }`}>
                    {scan.status === 'completed' ? '✓ Completed' :
                     scan.status === 'failed' ? '✗ Failed' :
                     '⏹ Stopped'}
                  </span>
                  {scan.status === 'completed' && scan.duplicate_groups_found > 0 && (
                    <span className="badge badge-cyan">
                      {scan.duplicate_groups_found} groups
                    </span>
                  )}
                  {scan.status === 'completed' && scan.recoverable_space > 0 && (
                    <span className="badge badge-red">
                      {formatBytes(scan.recoverable_space)}
                    </span>
                  )}
                </div>
                <div style={{ display: 'flex', gap: 'var(--space-sm)', marginLeft: 'var(--space-md)' }}>
                  {scan.status === 'completed' && scan.duplicate_groups_found > 0 && (
                    <button
                      className="btn btn-primary btn-sm"
                      onClick={() => navigate('/duplicates')}
                    >
                      Review →
                    </button>
                  )}
                  <button
                    className="btn btn-danger btn-sm"
                    onClick={() => handleDeleteScan(scan.id)}
                    title="Delete this scan from history"
                  >
                    ✕
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <FolderBrowser
        isOpen={showBrowser}
        onClose={() => setShowBrowser(false)}
        onSelect={(selected) => setPath(selected)}
        initialPath={path || undefined}
      />
    </div>
  );
}
