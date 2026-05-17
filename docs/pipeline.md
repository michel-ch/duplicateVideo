# The Scan Pipeline

A scan transforms a directory path into a set of `DuplicateGroup` rows in SQLite. The work happens inside a single async coroutine — `run_scan_pipeline()` in [`backend/api/scan.py`](../backend/api/scan.py).

## Overview

| Stage | Status name | Function | Output |
|---|---|---|---|
| 1 | `scanning` | `services.scanner.discover_videos` | List of video file paths |
| 1.5 | `scanning` | inline cache lookup by `(path, size, mtime_ns)`, version-gated by `PHASH_VERSION` | `hits` + `misses` partitions; new `FileCache` stubs; backfill of `head_tail_xxh3` for legacy hits |
| 2 | `metadata` | `services.metadata.extract_metadata` + `services.hasher.extract_thumbnail` + `services.scanner.compute_head_tail_hash` | `VideoFile` rows + thumbnails + blake2b head/tail hash (misses only; hits built from cache) |
| 2.5 | `metadata` | inline byte-identical clustering by `(file_size, head_tail_xxh3)` | `phash_follower_to_rep` map — followers skip stages 3 and 4b, inherit rep's outputs after |
| 3 | `hashing` | `services.hasher.extract_and_hash` + `_strip_letterbox` + `compute_aggregate_hash` | Perceptual hash JSON + 256-bit aggregate hash (pre-screened candidates only; followers inherit; hits use cached) |
| 4 | `comparing` (sub-stage A) | `services.comparator.group_by_duration` | Set of "candidate" file paths (already computed earlier for the stage-3 pre-screen, recomputed here for the audio gate) |
| 4 | `comparing` (sub-stage B) | `services.audio_fingerprint.audio_fingerprint` (middle 60 s) | RMS energy profile (only candidates without cached FP at `cache_version >= AUDIO_FP_VERSION`, followers excluded) |
| 5 | `comparing` (sub-stage C) | `services.comparator.run_duplicate_pipeline` (FAISS-shortlisted for groups ≥ 16) | Duplicate groups |
| 6 | `comparing` (sub-stage D) | `services.quality_scorer.rank_group` | Ranked groups + best file |
| 7 | (post-complete) | inline `DELETE FROM file_cache WHERE …` | Stale cache rows under root removed |

Stages 2, 3, and 4b are gated by the cache: a hit on `(file_path, file_size, mtime_ns)` with the correct `cache_version` skips that stage's work and reads from `file_cache` instead. Stages emit progress over WebSocket via `manager.send_progress()` (throttled to ≥ 0.5% advance) and persist progress to the `scan_jobs` row. Per-file failures fan out as `error_log` messages.

## Stage 1 — discover

`discover_videos(root_path)` walks the tree with `os.walk`, filtering by `settings.VIDEO_EXTENSIONS`. Hidden directories (`startswith('.')`) and the trash folder (`.duplicate_trash`) are skipped. Returns absolute paths.

Cost: O(files), no per-file processing. For 100k files this completes in seconds.

If zero files are found, the scan short-circuits to `completed`.

## Stage 1.5 — cache lookup + partition

Implemented inline in `run_scan_pipeline`. For every discovered path:

1. `get_file_info(path)` returns `(file_path, file_size, mtime_ns)` from a single `stat()` call.
2. The pipeline bulk-loads `FileCache` rows for these paths (chunked at 500 paths per `IN` clause to stay under SQLite's parameter limit).
3. Each file is partitioned:
   - **Hit**: a cache row exists *and* has `perceptual_hashes`. A `VideoFile` is built directly from the cache (no ffprobe, no frame extraction). `cache_hit = True`.
   - **Miss**: no cache row, *or* row exists but lacks `perceptual_hashes`. A new `FileCache` stub is `db.add()`ed (or the existing partial row is reused). The miss flows through stages 2–4b normally.
4. After partitioning, a single `db.commit()` flushes the new stubs so they have IDs to be referenced as `VideoFile.file_cache_id`.

The cache lookup itself is essentially free — no I/O beyond what stage 1 already does. Stages 2, 3, and 4b only iterate misses (and, for 4b, only candidates among those misses). On a fully-cached re-scan, the work for unchanged files collapses into "build VideoFile from cache row".

See [`research/caching-incremental.md`](research/caching-incremental.md) for the design rationale and savings estimates.

## Stage 2 — metadata + thumbnail + head/tail hash (misses only)

Runs only on cache misses. Three operations are co-located here because they all need to touch the file once:

1. `extract_metadata(file_path)` runs ffprobe with `-show_format -show_streams -show_entries stream_side_data=rotation`. Parses the JSON into duration, dimensions, bitrate, codecs, FPS, audio channels, sample rate, **SAR** (sample aspect ratio), and **rotation**.
2. `extract_thumbnail(file_path, output, duration, codec)` extracts a single JPEG from the middle of the video using GPU decoding when available.
3. `compute_head_tail_hash(file_path, file_size)` reads the first 64 KiB and last 64 KiB and returns an 8-byte blake2b digest. Runs off-event-loop via `asyncio.to_thread`. ~1 ms per file on SSD; used by stage 2.5 for the byte-identical fast-path.

All results are written back to the corresponding `FileCache` row so future scans skip this stage entirely.

**Cross-stage optimisation:** the SAR/rotation/duration/codec from stage 2 are stashed on the `VideoFile` ORM object as `_meta_video_info`. This dict is then passed to `extract_and_hash()` in stage 3 to skip three redundant ffprobe calls per video.

**Backfill for legacy cache hits**: hits whose `FileCache.head_tail_xxh3` is null (rows created before the column existed) get the value computed in a small concurrent batch right after partitioning, before stage 2.5 runs.

### Batching and concurrency

```python
sem = asyncio.Semaphore(max_concurrent)
BATCH = max_concurrent * 4

for batch_start in range(0, total_files, BATCH):
    await _pipeline_check(...)              # pause/stop check
    batch = video_paths[batch_start : batch_start + BATCH]
    tasks = [_process_one_meta(i, p) for i, p in enumerate(batch)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    await _pipeline_check(...)              # post-batch check
    # persist + send WS update
```

`max_concurrent` is `GPU_MAX_CONCURRENT` (12) when GPU is available, else `MAX_CONCURRENT_FFMPEG` (8). Each task acquires the semaphore before spawning ffmpeg/ffprobe.

Progress percentage during this stage spans **5% → 45%**.

## Stage 2.5 — byte-identical fast-path clustering

After stage 2 completes, every video has a `head_tail_xxh3`. Videos sharing `(file_size, head_tail_xxh3)` are clustered (a hash collision plus an identical size is astronomically unlikely — these files are virtually certainly byte-identical). Each cluster picks its lexicographically-first path as the **representative**; the remaining videos are **followers**.

Followers skip stages 3 and 4b. After each of those stages, the representative's freshly-computed (or cached) `perceptual_hashes` and `audio_fp` are propagated into the followers' `VideoFile` objects AND persisted to their `FileCache` rows (with `cache_version` bumped). This means a byte-identical copy still participates in transitive pHash matching — e.g. if there's also a re-encode with the same content, the follower links to it via the shared hash.

The cost saved per cluster of N is `(N - 1)` skipped stage-3 + stage-4b decodes.

## Stage 3 — perceptual hashing (candidates only)

Runs only on `VideoFile` rows that satisfy ALL of:

- No `perceptual_hashes` already populated (cache miss or rejected by `cache_version < PHASH_VERSION`).
- File is in a duration group of ≥ 2 (the pre-screen — videos with unique durations can never match anyone, so hashing them is pure waste).
- Not a byte-identical follower (will inherit the representative's hashes right after this stage).

`extract_and_hash(file_path, num_frames, duration, codec, video_info)` branches on `duration`:

**Short videos (< 60 s)** use the original single-pass GPU path:
1. Compute target fps so exactly `num_frames` frames are extracted: `target_fps = num_frames / duration`.
2. Build an FFmpeg command that decodes (optionally on GPU via `*_cuvid`), then on the CPU side: SAR (`scale=iw*sar:ih,setsar=1`), portrait → landscape (`transpose=1`), scale to `320:-2`.
3. ffmpeg timeout is scaled by duration: `max(60, min(900, int(duration * 0.3)))`.

**Longer videos (≥ 60 s)** use `_extract_frames_seek_sync` — N parallel ffmpeg subprocesses, each with `-ss <timestamp>` **before** `-i` for container-level fast seek. Wall-clock per video is roughly constant in `num_frames` instead of `duration`, which fixes the timeouts previously suffered by HEVC files over network shares: the `fps=N/duration` filter on the single-pass path forces a full sequential decode, which on SMB blows past the 60 s timeout floor for files as short as ~2 min. CPU decode only (per-call CUDA context init isn't worth it for ~1 s of bitstream per subprocess). Inner pool of 4 workers; per-frame timeout 60 s.

If the short-video GPU extraction yields zero frames, it retries on CPU automatically. If the seek path yields zero frames, falls back to the short-video single-pass path.

Each extracted JPEG is then run through `_strip_letterbox` (Python/numpy bbox detection at threshold 24/255; only crops when ≥ 5 % of a dimension is dark border) before `imagehash.phash(img, hash_size=16)` (256-bit hash). This closes the letterbox false-negative class — two encodes that differ only in black-bar padding now produce matching hashes.

Hashes are stored as a JSON array on `VideoFile.perceptual_hashes` AND written through to `FileCache.perceptual_hashes` with `cache_version = max(old, PHASH_VERSION)`.

Right after this stage, `compute_aggregate_hash` reduces each video's 12 frame hashes to a single 256-bit per-bit-majority hash, cached as `FileCache.aggregate_hash`. Used by the FAISS prescreen in stage 5.

Progress: **45% → 75%**.

## Stage 4a — duration pre-grouping (audio candidate selection)

Before paying for audio fingerprinting, the pipeline calls `group_by_duration()` once with all videos and collects the union of all candidate paths. **Videos with unique durations cannot match anything**, so audio-fingerprinting them is wasted work. This optimisation typically eliminates 50–95% of audio FP work depending on dataset uniformity.

```python
_pre_groups = group_by_duration(_pre_video_data, duration_tolerance)
_candidate_paths = {vd["file_path"] for g in _pre_groups for vd in g}
```

## Stage 4b — audio fingerprinting (candidates without cached FP)

Three-tier filter:

1. The candidate set comes from stage 4a (only files in a duration group).
2. Byte-identical followers are excluded — they'll inherit the rep's FP right after this stage.
3. Candidates with a cached `FileCache.audio_fp` whose `cache_version >= AUDIO_FP_VERSION` skip fingerprinting entirely; older cached FPs are rejected and re-extracted (the sampling rule changed in version 2 — see below).

For each file that does need fingerprinting, `audio_fingerprint(file_path, duration)`:

1. If `duration is not None and duration > 60`: ffmpeg decodes a **centred 60-second window** (`-ss (duration - 60) / 2 -t 60`). Otherwise decodes from the start with `-t 60` (so short clips get whatever they have). Output: **8 kHz mono 16-bit PCM** piped to stdout. Decoding the full track on long videos used to be the single biggest stage-4b cost; this saves 80-95% of decode time on movies.
2. Splits the sample stream into 64 equal segments.
3. Computes the RMS energy of each segment.
4. Normalises to `[0, 1]` by dividing by the peak.

The output is a list of 64 floats. Writes to `FileCache.audio_fp` and bumps `cache_version` to `AUDIO_FP_VERSION`.

**Separate CPU semaphore**: audio decode is CPU-bound and used to share the GPU-tuned `max_concurrent` (often 12 on GPU machines), which over-saturated the CPU. Now uses `cpu_concurrent = min(MAX_CONCURRENT_FFMPEG, cpu_count())`, distinct from the stage-3 GPU pool.

After extraction, the rep's `audio_fp` is propagated into every follower in its byte-identical cluster (and persisted to their cache rows).

Progress: stays around **76%** with WS messages reflecting candidate counts.

## Stage 5 — comparison

`run_duplicate_pipeline(videos, duration_tolerance, hash_threshold)`:

1. **`group_by_duration`** — sort by duration, expand a "current group" while the next video's duration is within `max(tolerance, 0.05 * anchor)` of the anchor. Groups of size 1 are dropped.
2. **`_faiss_phash_candidates`** (optional shortcut) — for duration groups of ≥ 16 videos where `faiss-cpu` is installed AND most videos have a cached `aggregate_hash`, builds an in-memory `faiss.IndexBinaryFlat` and runs `range_search` at radius `1.5 × hash_threshold`. The result is a set of candidate `(i, j)` pairs that *might* match. Falls back to all-pairs (current behaviour) on small groups, missing FAISS, missing aggregates, or any FAISS error (logged via `print` so a programming bug doesn't silently regress).
3. **`find_duplicates_in_group`** — Union-Find over the group:
   - Skip pairs where file sizes differ by >20× (sanity check).
   - **pHash check** — gated on the FAISS shortlist when available; runs `compare_hash_sets` against the verifier (12×12 best-match) only for shortlisted pairs. When FAISS isn't used, every pair is checked (no behaviour change). If average best-match Hamming ≤ `hash_threshold`, the pair matches as `"video"`.
   - **Audio fallback** — if no pHash match, `compare_audio_fingerprints` returns a normalised cross-correlation 0–100. ≥ 80 matches as `"audio"`. Always runs for every duration-group pair (FAISS doesn't gate this).
   - On match: `union(i, j)`, record the similarity score and method.
4. **`calculate_group_similarity`** averages pairwise similarities for each connected component.

Result: a list of `{"videos": [...], "similarity_score": float}` dicts.

See [duplicate-detection.md](duplicate-detection.md) for algorithmic detail.

## Stage 6 — quality ranking + persistence

For each duplicate group:

1. `rank_group(videos)` computes a quality score per video (see [quality-scoring.md](quality-scoring.md)) and sorts descending. The top-ranked is marked `is_best_quality=True`.
2. `calculate_wasted_space(ranked)` sums the file sizes of all but the largest. (Note: largest by **size**, not best by **quality** — so the wasted-space figure is conservative.)
3. A `DuplicateGroup` row is inserted with `best_file_id`, `similarity_score`, `total_wasted_space`, `file_count`.
4. Each `VideoFile` row in the group is updated with `duplicate_group_id`, `quality_score`, `is_best_quality`.

Progress: **90% → 100%**.

## Stage 7 — scan-end cache sweep

After the scan completes (status `completed`, final WebSocket message sent), the pipeline runs:

```sql
DELETE FROM file_cache
WHERE last_seen_at < scan_started_at
  AND file_path LIKE '<resolved_root>\%'
```

Rationale: any cache row whose `file_path` falls under the just-scanned root but wasn't touched (`last_seen_at` not bumped) during this scan is stale — the file was moved or deleted between scans. Pruning here keeps the cache from growing forever.

The sweep is wrapped in a try/except — a sweep failure does not fail the scan. It does **not** run on `_ScanStopped` or other error paths, so a partial scan won't accidentally prune entries it didn't have a chance to touch.

## Pause / stop / queue

[`services/scan_control.py`](../backend/services/scan_control.py) holds two `asyncio.Event` per scan:

- `resume_event` — when **clear**, the pipeline blocks. Initialised set.
- `stop_event` — when **set**, the pipeline aborts (raises `_ScanStopped`).

`_pipeline_check()` is called between every batch and after every `gather()`. It:

1. Checks `stop_event` → if set, raises `_ScanStopped`.
2. Checks `resume_event` → if clear, awaits resume **or** stop, blocking the coroutine. While blocked, the DB row is `status=paused`.
3. After resuming, restores the previous stage status.

The `finally` block calls `scan_control.unregister(scan_id)` and `_start_next_queued()`, which scans for the oldest `queued` ScanJob and launches it as a new background task. This is what makes "queue multiple scans" work.

## WebSocket progress throttling

Sending a WS message for every batch would flood the frontend on small files. The pipeline maintains `_last_ws_pct` and only sends when the new percentage is ≥ **0.5** above the last (or hits 100). The threshold is small enough that every batch-end emission survives but suppresses no-change duplicates. DB updates happen every batch regardless — only the wire-level fan-out is throttled.

## Crash recovery / mid-scan persistence

`FileCache` writes commit batch-by-batch so a hard kill (Ctrl-C, OOM, power loss) loses at most one batch of work:

| Stage | Commit cadence |
|---|---|
| 2 (metadata + thumbnail + head/tail hash for misses) | per batch |
| 2 (head/tail backfill for cache hits) | per 64-file chunk |
| 3 (perceptual hashes) | per batch |
| 3 (aggregate hash compute) | every 256 rows (the loop is pure-Python with no `await`s) |
| 4b (audio fingerprint) | per batch |

On the next startup, `main.py:_recover_orphaned_scans()` (runs in the FastAPI lifespan after `init_db()`) finds every `ScanJob` still in an active status (`pending` / `scanning` / `metadata` / `hashing` / `comparing` / `paused`) — those are scans whose pipeline died with the process — and flips them to `stopped`. Without this, the UI shows a phantom spinner forever and the queue stays blocked because `_has_active_scan()` would always return True.

Because the cross-scan `file_cache` is keyed by `(file_path, file_size, mtime_ns)` and outlives scans, re-running the killed scan on the same root picks up exactly where it left off — stage 1.5 partitions every committed cache row as a hit and skips stages 2–4b for those files.

## Per-scan error log

`services/error_log.py` is an in-memory ring buffer (200 entries) per scan_id. Every per-file failure in the pipeline (`metadata extraction failed`, `no frames extracted`, `audio fingerprint produced no samples`, etc.) is recorded by `_log_err(stage, message, file_path)`, which both appends to the buffer AND broadcasts an `error_log` WebSocket message via `manager.send_error_log()`. On WS connect, the endpoint replays the buffer so a late subscriber catches up.

The buffer is registered in pipeline init and unregistered ~30 s after scan end (deferred via `asyncio.create_task` so a UI client viewing the just-finished scan still gets the backlog).

## Error handling

The whole pipeline is wrapped in:

```python
try:
    ...
except _ScanStopped:
    # mark stopped, send completion message
except Exception as e:
    # mark failed, store error_message, send error WS message,
    # also log to error_log as stage="pipeline" so the UI panel shows it
finally:
    scan_control.unregister(scan_id)
    # error_log.unregister(scan_id) is scheduled with a 30s delay
    await _start_next_queued()
```

Per-file errors inside `gather(..., return_exceptions=True)` are logged via `_log_err` (visible in the UI) AND never abort the scan.
