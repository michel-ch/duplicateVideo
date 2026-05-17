# Configuration

All settings are in [`backend/config.py`](../backend/config.py) as a single Pydantic `BaseSettings` class. They can be overridden in three ways:

1. **Edit `config.py`** directly — changes survive restarts.
2. **`.env` file** in `backend/` — loaded by `pydantic_settings`. Same key names as the class attributes (uppercase).
3. **`PUT /api/settings`** — modifies the live `settings` instance in memory. **Not persisted.** Resets on restart.

Use #3 for experimentation, #1 or #2 for permanent changes.

## All settings

### Database

| Key | Default | Effect |
|---|---|---|
| `DATABASE_URL` | `sqlite+aiosqlite:///./duplicate_detector.db` | SQLAlchemy URL. Must be an async-compatible driver. |

### Scanning

| Key | Default | Effect |
|---|---|---|
| `VIDEO_EXTENSIONS` | `.mp4 .mkv .avi .mov .wmv .flv .webm .m4v .ts .3gp` | File extensions considered videos. Lowercased before matching. |
| `MAX_CONCURRENT_FFMPEG` | `8` | Max parallel ffmpeg/ffprobe subprocesses when running CPU-only. |
| `KEY_FRAMES_COUNT` | `12` | Number of frames extracted per video for hashing. More = more robust to per-frame noise but slower. |

### GPU

| Key | Default | Effect |
|---|---|---|
| `GPU_ENABLED` | `True` | Master switch. Set False to force CPU even when an NVIDIA GPU is present. |
| `GPU_MAX_CONCURRENT` | `12` | Replaces `MAX_CONCURRENT_FFMPEG` when GPU is active. Higher because GPU decode has lower CPU cost per task. |

See [gpu-acceleration.md](gpu-acceleration.md) for what GPU mode actually accelerates.

### Duplicate detection

| Key | Default | Effect |
|---|---|---|
| `DURATION_TOLERANCE_SECONDS` | `3.0` | Absolute duration tolerance for stage-1 grouping. Combined with 5% relative tolerance, whichever is larger. |
| `HASH_SIMILARITY_THRESHOLD` | `14` | Hamming distance threshold (0–256 for a 16×16 pHash). **Higher = more lenient.** Default is calibrated for ~95% precision. |
| `SIMILARITY_THRESHOLD_PERCENT` | `70.0` | Default for the Dashboard slider + `ScanOptions.similarity_threshold` + the auto-clean/settings UI. The comparator itself uses Hamming distance (`HASH_SIMILARITY_THRESHOLD`), not this percentage. Lowered from 85% so the default catches more near-duplicates out of the box. |

The audio threshold (80%) is **hard-coded** in [`comparator.py`](../backend/services/comparator.py:144) and not exposed.

See [duplicate-detection.md](duplicate-detection.md) for the algorithms.

### Quality scoring weights

| Key | Default | Notes |
|---|---|---|
| `RESOLUTION_WEIGHT` | `0.40` | |
| `BITRATE_WEIGHT`    | `0.25` | |
| `CODEC_WEIGHT`      | `0.15` | |
| `FILE_SIZE_WEIGHT`  | `0.10` | |
| `FPS_WEIGHT`        | `0.10` | |

**Weights must sum to 1.0.** Nothing checks this — if you change one, adjust the others.

### Codec scores

| Key | Default | Notes |
|---|---|---|
| `CODEC_SCORES` | `{ hevc: 1.0, h265: 1.0, h264: 0.8, avc: 0.8, vp9: 0.85, av1: 1.0 }` | Looked up by substring match. Anything not found = 0.5. |

See [quality-scoring.md](quality-scoring.md).

### Deletion

| Key | Default | Effect |
|---|---|---|
| `DEFAULT_TRASH_MODE` | `True` | When True, deletions move files to `.duplicate_trash/` (recoverable). When False, deletes are permanent unless the API request says otherwise. |
| `TRASH_FOLDER_NAME` | `.duplicate_trash` | Per-scan-root subdirectory used as trash. Mirrored relative paths preserved inside. |

### Protected paths

| Key | Default | Effect |
|---|---|---|
| `PROTECTED_PATHS` | `[]` | Files inside any of these absolute paths cannot be deleted. The check is in `file_manager.py:is_protected_path` using `Path.resolve()` prefix matching. |

### Tools

| Key | Default | Effect |
|---|---|---|
| `FFPROBE_PATH` | `ffprobe` | Override if ffprobe isn't on PATH. ffmpeg's path is **not** configurable — it must be `ffmpeg` on PATH. |
| `THUMBNAILS_DIR` | `thumbnails` | Relative to backend cwd. Mounted at `/thumbnails` for static serving. |

## `.env` example

```ini
# backend/.env
GPU_ENABLED=true
HASH_SIMILARITY_THRESHOLD=12
DURATION_TOLERANCE_SECONDS=5.0
PROTECTED_PATHS=["D:\\Important", "E:\\Originals"]
```

Booleans, lists, and dicts are parsed by Pydantic. Lists must be valid JSON syntax.

## Settings API quirks

`PUT /api/settings` accepts lower_case keys (matching the response shape) and maps them to the upper-case attribute names internally. See `actions.py:update_settings` for the field map. Most attributes are spelled differently between the API and the class, e.g. `similarity_threshold` ↔ `SIMILARITY_THRESHOLD_PERCENT`.

## What changes take effect when

- **`HASH_SIMILARITY_THRESHOLD`, `DURATION_TOLERANCE_SECONDS`, `KEY_FRAMES_COUNT`** — only affect **new scans**. Existing duplicate groups don't recompute.
- **Quality weights and codec scores** — only affect **new scans**. Existing `quality_score` columns keep their old values.
- **`PROTECTED_PATHS`** — checked at deletion time; takes effect immediately.
- **`VIDEO_EXTENSIONS`** — checked during stage 1 of each scan.
- **`GPU_ENABLED`** — read at startup. **Restart required** for changes to take effect.
- **`MAX_CONCURRENT_FFMPEG` / `GPU_MAX_CONCURRENT`** — read at the start of each scan.

## Per-scan overrides

The `ScanRequest.options` field accepts:

```json
{
  "similarity_threshold": 70.0,
  "duration_tolerance":   2.0,
  "key_frames_count":     8,
  "hash_threshold":       10,
  "max_concurrent":       4
}
```

These override the corresponding settings **for that one scan only**. Useful for testing different thresholds against the same dataset.
