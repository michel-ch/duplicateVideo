// TypeScript interfaces for the application

export interface ScanOptions {
  similarity_threshold: number;
  duration_tolerance: number;
  key_frames_count: number;
  hash_threshold: number;
  max_concurrent: number;
}

export interface ScanStatus {
  id: number;
  root_path: string;
  status: string;
  total_files: number;
  scanned_files: number;
  current_file: string | null;
  current_stage: string | null;
  progress_percent: number;
  duplicate_groups_found: number;
  recoverable_space: number;
  started_at: string | null;
  completed_at: string | null;
  error_message: string | null;
}

export interface VideoFile {
  id: number;
  file_path: string;
  file_name: string;
  file_size: number;
  duration: number | null;
  width: number | null;
  height: number | null;
  bitrate: number | null;
  video_codec: string | null;
  audio_codec: string | null;
  fps: number | null;
  audio_channels: number | null;
  audio_sample_rate: number | null;
  quality_score: number | null;
  is_best_quality: boolean;
  thumbnail_path: string | null;
  is_deleted: boolean;
}

export interface DuplicateGroup {
  id: number;
  similarity_score: number;
  total_wasted_space: number;
  file_count: number;
  status: string;
  best_file_id: number | null;
  videos: VideoFile[];
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
}

export interface Stats {
  total_videos: number;
  total_scans: number;
  duplicate_groups: number;
  total_duplicates: number;
  recoverable_space: number;
  space_recovered: number;
  last_scan_date: string | null;
}

export interface AppSettings {
  similarity_threshold: number;
  duration_tolerance: number;
  key_frames_count: number;
  hash_threshold: number;
  max_concurrent: number;
  resolution_weight: number;
  bitrate_weight: number;
  codec_weight: number;
  file_size_weight: number;
  fps_weight: number;
  default_trash_mode: boolean;
  video_extensions: string[];
  protected_paths: string[];
}

export interface DeletionLog {
  id: number;
  original_path: string;
  trash_path: string | null;
  file_size: number;
  deletion_mode: string;
  deleted_at: string | null;
  is_undone: boolean;
}

export interface ScanProgressMessage {
  type: string;
  scan_id: number;
  status: string;
  total_files: number;
  scanned_files: number;
  current_file: string | null;
  current_stage: string | null;
  progress_percent: number;
  duplicate_groups_found?: number;
  recoverable_space?: number;
  message?: string;
  gpu_active?: boolean;
  gpu_name?: string | null;
}

export interface ScanErrorLogEntry {
  type: 'error_log';
  scan_id: number;
  stage: string;       // "metadata" | "hashing" | "audio_fp" | "cache_sweep" | "pipeline"
  level: string;       // "error" | "warning"
  message: string;
  file_path: string | null;
  timestamp: string;   // ISO 8601 UTC
}

export interface BrowseEntry {
  name: string;
  path: string;
  is_dir: boolean;
}

export interface BrowseResult {
  current_path: string;
  parent_path: string | null;
  entries: BrowseEntry[];
}

export interface GPUStatus {
  gpu_available: boolean;
  gpu_name: string;
  driver_version: string;
  vram_total_mb: number;
  vram_free_mb: number;
  hwaccel_supported: boolean;
  cuvid_decoders: string[];
  cuda_filters: string[];
  nvenc_encoders: string[];
  acceleration_active: boolean;
}

