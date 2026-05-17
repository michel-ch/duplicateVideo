# API Reference

All routes are prefixed with `/api`. The interactive Swagger UI is at http://localhost:9000/docs when the backend is running.

Routers are split across [`backend/api/scan.py`](../backend/api/scan.py), [`backend/api/duplicates.py`](../backend/api/duplicates.py), and [`backend/api/actions.py`](../backend/api/actions.py).

## Scan

### `POST /api/scan` — start a scan

```json
{
  "path": "D:\\Movies",
  "options": {
    "similarity_threshold": 70.0,
    "duration_tolerance": 2.0,
    "key_frames_count": 8,
    "hash_threshold": 10,
    "max_concurrent": 4
  }
}
```

Returns:

```json
{ "id": 42, "status": "queued" | "pending", "message": "..." }
```

If a scan is already active, the new scan goes to `queued` and is launched automatically when the active one ends. Otherwise it starts immediately as a background task.

### `GET /api/scan/{id}/status`

Returns a `ScanStatusResponse` (see [`models/schemas.py`](../backend/models/schemas.py)). Includes `status`, `progress_percent`, `current_stage`, `current_file`, counts.

### `GET /api/scans`

Returns all scan jobs, newest first.

### `POST /api/scan/{id}/pause`

Sets `status=paused` and clears the resume event. The pipeline blocks at the next `_pipeline_check`. If the scan is already terminal (`completed`/`failed`/`stopped`), returns 400.

### `POST /api/scan/{id}/resume`

Sets the resume event so the pipeline unblocks. Status returns to `scanning`.

### `POST /api/scan/{id}/stop`

Sets the stop event. The pipeline raises `_ScanStopped` at the next `_pipeline_check`. Status becomes `stopped` immediately and any in-flight work is discarded.

### `DELETE /api/scan/{id}` — cancel queued or delete from history

Works for both `queued` scans (cancels them before they start) **and** terminal scans (`completed` / `failed` / `stopped`, removing them from history). Rejects active scans (`pending` / `scanning` / `metadata` / `hashing` / `comparing` / `paused`) with a 400 — stop those first via `/stop`.

Cascades to `video_files` and `duplicate_groups` for that scan, in dependency order. The cross-scan `file_cache` is intentionally left intact so re-scans of the same paths still skip work.

```json
{ "scan_id": 42, "message": "Scan deleted" }
```

### `DELETE /api/scans` — clear all scan history

Wipes every non-active scan (queued + terminal) in one shot. Active scans are left alone. Uses raw `DELETE` statements rather than ORM cascade, so clearing 100+ scans is one query per table (`video_files` → `duplicate_groups` → `scan_jobs`).

```json
{ "deleted_count": 37, "message": "Deleted 37 scan(s) from history" }
```

### `WebSocket /api/scan/{id}/ws`

Sends JSON messages on every progress update:

```json
{
  "type": "progress",
  "scan_id": 42,
  "status": "hashing",
  "current_stage": "Computing perceptual hashes...",
  "current_file": "movie.mp4",
  "progress_percent": 62.5,
  "total_files": 1200,
  "scanned_files": 850,
  "message": "Hashed 850/1200 files",
  "gpu_active": true,
  "gpu_name": "NVIDIA GeForce RTX 3060 Ti"
}
```

Progress messages are throttled — only emitted when `progress_percent` advances by ≥ 0.5 % (or hits 100).

Final message has `"type": "complete"` and includes `duplicate_groups_found`, `recoverable_space`. Server pings can be replied to with `"ping"` (server replies `{"type": "pong"}`).

#### Per-file error stream

Per-file failures during the scan (frame-extract timeouts, ffprobe errors, audio decode failures, etc.) are broadcast as separate messages with `"type": "error_log"`:

```json
{
  "type": "error_log",
  "scan_id": 42,
  "stage": "hashing",
  "level": "error",
  "message": "no frames extracted (timeout or codec issue)",
  "file_path": "\\\\synology\\media\\long_movie.mkv",
  "timestamp": "2026-05-16T14:23:17.451983+00:00"
}
```

`stage` is one of `metadata`, `hashing`, `audio_fp`, `cache_sweep`, `pipeline` (the last one is the catastrophic-failure case that also flips the scan to `failed`). On WS connect, the endpoint replays the in-memory backlog of error_log entries (up to 200 per scan, kept ~30 s after scan end) so a late subscriber catches up.

Metadata-stage errors surface the actual `ffprobe` failure reason. The opaque `skipped (metadata unavailable)` placeholder was replaced with `ffprobe: <truncated stderr>` (or `ffprobe: ffprobe timed out`) so the cause of a refused file is visible in the UI panel.

The frontend `useWebSocket` reconnects automatically (up to 2s after close) **unless** the last status was terminal. The WS scheme is `wss:` when the page is served over HTTPS, `ws:` otherwise.

### `GET /api/browse?path=...`

Filesystem browser for the directory picker UI. With no path, returns drive letters on Windows or `/` on Unix. With a path, returns subdirectories only (files are filtered out). Raises 403 on PermissionError, 400 on non-directory paths.

```json
{
  "current_path": "C:\\Users\\me",
  "parent_path": "C:\\Users",
  "entries": [
    { "name": "Documents", "path": "C:\\Users\\me\\Documents", "is_dir": true }
  ]
}
```

## Duplicates

### `GET /api/duplicates`

```
?page=1&per_page=20&sort_by=wasted_space&min_similarity=80&status=pending&scan_id=42
```

`sort_by` ∈ {`wasted_space`, `similarity`, `file_count`, `date`}.

Returns a paginated list of duplicate groups, each with **non-deleted** videos inlined.

### `GET /api/duplicates/{group_id}`

Single group with **all** videos (including soft-deleted ones, for history).

### `POST /api/duplicates/{group_id}/resolve`

```json
{
  "keep_file_ids": [11, 12],
  "delete_file_ids": [13, 14],
  "move_to_trash": true
}
```

Deletes the `delete_file_ids`, marks them `is_deleted=true`, writes a `DeletionLog` per file. Sets `group.status="resolved"`. Errors are returned per-file in the response.

The `keep_file_ids` are documented for clarity but not used by the server — they have no effect on what's deleted. The user just needs to send the explicit delete list.

## Actions

### `POST /api/delete`

Same shape as `/duplicates/{id}/resolve` but works on arbitrary `file_ids` not bound to a group.

```json
{ "file_ids": [13, 14, 15], "move_to_trash": true }
```

### `POST /api/auto-clean`

```json
{ "move_to_trash": true, "confirm": false }
```

When `confirm=false` (default), returns a **preview** of files that would be deleted (every video in a non-`resolved` group with `is_best_quality=false` and `is_deleted=false`):

```json
{
  "preview": true,
  "files_to_delete": [...],
  "total_files": 47,
  "total_space": 23456789012,
  "message": "Will delete 47 files, freeing 21.85 GB"
}
```

When `confirm=true`, actually deletes them and sets the affected groups to `status=resolved`. Calling auto-clean a second time is a no-op: the `is_deleted` flag prevents redundant deletes, so the preview will simply return zero files.

### `GET /api/stats`

Dashboard summary: total videos, total scans, duplicate groups, total duplicates, recoverable space, space recovered, last scan date.

### `GET /api/settings` / `PUT /api/settings`

Read/write the live `Settings` instance. **Changes are not persisted** to disk — they reset on restart unless you edit `config.py` or set `.env` values.

```json
{
  "similarity_threshold": 70.0,
  "duration_tolerance": 2.0,
  "key_frames_count": 12,
  "hash_threshold": 14,
  "max_concurrent": 8,
  "resolution_weight": 0.40,
  "bitrate_weight":   0.25,
  "codec_weight":     0.15,
  "file_size_weight": 0.10,
  "fps_weight":       0.10,
  "default_trash_mode": true,
  "video_extensions": [".mp4", ".mkv", ...],
  "protected_paths": []
}
```

### `GET /api/history`

Paginated `DeletionLog` rows, newest first.

### `DELETE /api/history`

Clears all `DeletionLog` rows. Does **not** delete the trashed files themselves.

### `POST /api/history/{log_id}/undo`

Restores a file from trash to its original location:

1. Errors if `deletion_mode == "permanent"` or `is_undone == true`.
2. Calls `file_manager.undo_deletion(original_path, trash_path)`.
3. Marks the log `is_undone=true, undone_at=now`.
4. Finds the matching `VideoFile` (by `file_path == original_path`) and resets `is_deleted, deleted_at, trash_path`.

### `GET /api/gpu-status`

Returns the cached `GPUInfo` from `services/gpu_detector.py:get_gpu_info()`. Polled by the Dashboard on load. The data is static after startup.

```json
{
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3060 Ti",
  "driver_version": "555.42.02",
  "vram_total_mb": 8192,
  "vram_free_mb": 7400,
  "hwaccel_supported": true,
  "cuvid_decoders": ["h264_cuvid", "hevc_cuvid", ...],
  "cuda_filters": ["scale_cuda", "hwupload_cuda", ...],
  "nvenc_encoders": ["h264_nvenc", "hevc_nvenc", ...],
  "acceleration_active": true
}
```

## Static

- `GET /thumbnails/<file>` — served from `backend/thumbnails/`. Generated during stage 2 with names `scan{scan_id}_thumb_{idx}.jpg`.
- `GET /assets/*` and `GET /` — present only if `frontend/dist` exists. Serves the SPA with a fallback that returns `index.html` for any unknown path so client-side routes work on hard refresh.

## Error responses

FastAPI's standard envelope:

```json
{ "detail": "Scan not found" }
```

The frontend's `services/api.ts:request` extracts `.detail` and re-throws as a JS Error with that message.

## CORS

Configured in `main.py`:

```python
allow_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
]
```

Not used in production (frontend served from same origin), but allows dev with the Vite dev server.

## Filtered access logs

The uvicorn access logger has a `_QuietPollFilter` attached that drops `/api/stats` and `/api/gpu-status` lines. The Dashboard polls these every 3s and the noise was overwhelming.
