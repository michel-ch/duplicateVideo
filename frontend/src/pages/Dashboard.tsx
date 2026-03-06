import React, { useState, useEffect, useCallback } from 'react';
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
  const [threshold, setThreshold] = useState(85);
  const [showOptions, setShowOptions] = useState(false);
  const [showBrowser, setShowBrowser] = useState(false);

  // Scan list
  const [scans, setScans] = useState<ScanStatus[]>([]);

  // Active scan tracking
  const activeScan = scans.find(s => ACTIVE.includes(s.status)) || null;
  const activeScanId = activeScan?.id ?? null;
  const isScanning = activeScanId !== null;

  const { scanStatus, wsProgress, isComplete, error: scanError, reset } = useScanProgress(activeScanId);

  // ── Load scans and stats ──────────────────────────────────────────────────
  const loadScans = useCallback(async () => {
    try {
      const data = await api.listScans();
      setScans(data as ScanStatus[]);
    } catch {}
  }, []);

  useEffect(() => {
    api.getStats().then(setStats).catch(() => {});
    api.getGpuStatus().then(setGpuStatus).catch(() => {});
    loadScans();

    const interval = setInterval(() => {
      api.getStats().then(setStats).catch(() => {});
      loadScans();
    }, 3000);
    return () => clearInterval(interval);
  }, [loadScans]);

  // ── Handle completion ─────────────────────────────────────────────────────
  useEffect(() => {
    if (isComplete || scanError) {
      reset();
      loadScans();
    }
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
          <h3 style={{ fontFamily: 'var(--font-mono)', marginBottom: 'var(--space-md)', color: 'var(--text-secondary)' }}>
            📜 Recent Scans
          </h3>
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
                {scan.status === 'completed' && scan.duplicate_groups_found > 0 && (
                  <button
                    className="btn btn-primary btn-sm"
                    onClick={() => navigate('/duplicates')}
                    style={{ marginLeft: 'var(--space-md)' }}
                  >
                    Review →
                  </button>
                )}
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
