import type { ScanProgressMessage, ScanStatus } from '../types';

interface Props {
  status: ScanStatus | null;
  wsProgress: ScanProgressMessage | null;
  onPause?: () => void;
  onResume?: () => void;
  onStop?: () => void;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

export default function ProgressTracker({ status, wsProgress, onPause, onResume, onStop }: Props) {
  const data = wsProgress || status;

  if (!data) {
    return (
      <div className="scan-progress-section">
        <div className="scan-progress-header">
          <div className="scan-stage">
            <span className="spinner" />
            Initializing scan...
          </div>
        </div>
        <div className="progress-container">
          <div className="progress-bar" style={{ width: '0%' }} />
        </div>
      </div>
    );
  }

  const percent = data.progress_percent || 0;
  const stage = ('current_stage' in data ? data.current_stage : '') || 'Initializing...';
  const currentFile = ('current_file' in data ? data.current_file : '') || '';
  const scanned = data.scanned_files || 0;
  const total = data.total_files || 0;
  const message = wsProgress?.message || '';
  const gpuActive = (wsProgress as any)?.gpu_active || false;
  const gpuName = (wsProgress as any)?.gpu_name || '';

  const scanStatus = data.status;
  const isPaused = scanStatus === 'paused';
  const isStopped = scanStatus === 'stopped';
  const isFinished = scanStatus === 'completed' || scanStatus === 'failed' || isStopped;
  const isRunning = !isFinished && !isPaused;

  return (
    <div className="scan-progress-section">
      <div className="scan-progress-header">
        <div className="scan-stage">
          {isRunning && <span className="spinner" />}
          {isPaused && <span className="pause-icon">⏸</span>}
          {scanStatus === 'completed' ? '✓ ' : scanStatus === 'failed' ? '✗ ' : isStopped ? '⏹ ' : ''}
          {stage}
        </div>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          {gpuActive && (
            <span className="gpu-badge" title={gpuName}>
              <span className="gpu-badge-icon">⚡</span>
              GPU
            </span>
          )}
          <span className={`badge ${
            isPaused ? 'badge-yellow' :
            isStopped ? 'badge-red' :
            scanStatus === 'completed' ? 'badge-green' :
            scanStatus === 'failed' ? 'badge-red' :
            'badge-cyan'
          }`}>
            {scanStatus?.toUpperCase()}
          </span>
        </div>
      </div>

      <div className="progress-container">
        <div
          className={`progress-bar ${gpuActive ? 'progress-bar-gpu' : ''} ${isPaused ? 'progress-bar-paused' : ''}`}
          style={{ width: `${percent}%` }}
        />
      </div>

      <div className="progress-label">
        <span className="percent">{percent.toFixed(1)}%</span>
        <span className="detail">{scanned} / {total} files</span>
      </div>

      {currentFile && (
        <div className="scan-current-file" title={currentFile}>
          📄 {currentFile}
        </div>
      )}

      {message && (
        <div className="scan-current-file" style={{ marginTop: '4px', color: 'var(--text-secondary)' }}>
          {message}
        </div>
      )}

      {/* ── Scan Controls: Pause / Resume / Stop ── */}
      {!isFinished && (
        <div className="scan-controls">
          {isPaused ? (
            <button
              className="btn btn-scan-control btn-resume"
              onClick={onResume}
              title="Resume scan"
            >
              <span className="control-icon">▶</span>
              Resume
            </button>
          ) : (
            <button
              className="btn btn-scan-control btn-pause"
              onClick={onPause}
              title="Pause scan"
            >
              <span className="control-icon">⏸</span>
              Pause
            </button>
          )}
          <button
            className="btn btn-scan-control btn-stop"
            onClick={onStop}
            title="Stop scan"
          >
            <span className="control-icon">⏹</span>
            Stop
          </button>
        </div>
      )}

      {(data as any).duplicate_groups_found > 0 && (
        <div style={{ marginTop: 'var(--space-md)', display: 'flex', gap: 'var(--space-lg)' }}>
          <span className="badge badge-yellow">
            {(data as any).duplicate_groups_found} duplicate groups
          </span>
          {(data as any).recoverable_space > 0 && (
            <span className="badge badge-green">
              {formatBytes((data as any).recoverable_space)} recoverable
            </span>
          )}
        </div>
      )}
    </div>
  );
}
