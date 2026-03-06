import type { VideoFile } from '../types';
import QualityBadge from './QualityBadge';

interface Props {
  video: VideoFile;
  onKeep?: () => void;
  onDelete?: () => void;
  isSelected?: boolean;
  selectionMode?: 'keep' | 'delete' | null;
}

function formatBytes(bytes: number): string {
  if (!bytes) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDuration(seconds: number | null): string {
  if (!seconds) return '--:--';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function getResLabel(w: number | null, h: number | null): string {
  if (!w || !h) return 'Unknown';
  if (h >= 2160) return '4K';
  if (h >= 1440) return '1440p';
  if (h >= 1080) return '1080p';
  if (h >= 720) return '720p';
  if (h >= 480) return '480p';
  return `${w}×${h}`;
}

export default function VideoCard({ video, onKeep, onDelete, selectionMode }: Props) {
  const cardClass = [
    'video-card',
    video.is_best_quality ? 'best' : '',
    selectionMode === 'delete' ? 'selected-delete' : '',
  ].filter(Boolean).join(' ');

  return (
    <div className={cardClass}>
      <div className="video-thumb">
        {video.thumbnail_path ? (
          <img
            src={video.thumbnail_path}
            alt={video.file_name}
            loading="lazy"
          />
        ) : (
          <span className="placeholder">🎬</span>
        )}
        {video.is_best_quality && (
          <span className="best-badge">★ BEST</span>
        )}
        <span className="resolution-badge">
          {getResLabel(video.width, video.height)}
        </span>
      </div>

      <div className="video-info">
        <div className="filename" title={video.file_name}>
          {video.file_name}
        </div>

        <div className="meta-row">
          <span>Duration</span>
          <span>{formatDuration(video.duration)}</span>
        </div>
        <div className="meta-row">
          <span>Size</span>
          <span>{formatBytes(video.file_size)}</span>
        </div>
        <div className="meta-row">
          <span>Codec</span>
          <span>{video.video_codec?.toUpperCase() || 'N/A'}</span>
        </div>
        <div className="meta-row">
          <span>Bitrate</span>
          <span>{video.bitrate ? `${(video.bitrate / 1_000_000).toFixed(1)} Mbps` : 'N/A'}</span>
        </div>
        <div className="meta-row">
          <span>FPS</span>
          <span>{video.fps?.toFixed(1) || 'N/A'}</span>
        </div>
        <div className="meta-row">
          <span>Quality</span>
          <QualityBadge score={video.quality_score} />
        </div>
      </div>

      {(onKeep || onDelete) && (
        <div className="video-actions">
          {onKeep && (
            <button
              className="btn btn-success btn-sm"
              onClick={onKeep}
              style={{ flex: 1 }}
            >
              ✓ Keep
            </button>
          )}
          {onDelete && (
            <button
              className="btn btn-danger btn-sm"
              onClick={onDelete}
              style={{ flex: 1 }}
            >
              ✗ Delete
            </button>
          )}
        </div>
      )}
    </div>
  );
}
