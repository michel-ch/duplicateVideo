import type { VideoFile } from '../types';

interface Props {
  video: VideoFile;
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

export default function MetadataTable({ video }: Props) {
  const rows = [
    { label: 'File Path', value: video.file_path },
    { label: 'File Name', value: video.file_name },
    { label: 'File Size', value: formatBytes(video.file_size) },
    { label: 'Duration', value: formatDuration(video.duration) },
    { label: 'Resolution', value: video.width && video.height ? `${video.width} × ${video.height}` : 'N/A' },
    { label: 'Video Codec', value: video.video_codec?.toUpperCase() || 'N/A' },
    { label: 'Audio Codec', value: video.audio_codec?.toUpperCase() || 'N/A' },
    { label: 'Bitrate', value: video.bitrate ? `${(video.bitrate / 1_000_000).toFixed(2)} Mbps` : 'N/A' },
    { label: 'Frame Rate', value: video.fps ? `${video.fps.toFixed(2)} FPS` : 'N/A' },
    { label: 'Audio Ch.', value: video.audio_channels ? `${video.audio_channels} channels` : 'N/A' },
    { label: 'Sample Rate', value: video.audio_sample_rate ? `${video.audio_sample_rate} Hz` : 'N/A' },
    { label: 'Quality Score', value: video.quality_score?.toFixed(2) || 'N/A' },
  ];

  return (
    <table className="meta-table">
      <tbody>
        {rows.map((row) => (
          <tr key={row.label}>
            <td>{row.label}</td>
            <td>{row.value}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
