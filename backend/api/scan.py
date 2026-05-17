"""Scan API endpoints with pause / stop support."""
from __future__ import annotations

import asyncio
import json
import os
import platform
import string
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete

from models.database import get_db, ScanJob, VideoFile, DuplicateGroup, FileCache
from models.schemas import ScanRequest, ScanStatusResponse
from services.scanner import discover_videos, get_file_info, compute_head_tail_hash
from services.metadata import extract_metadata
from services.hasher import (
    extract_and_hash,
    extract_thumbnail,
    compute_aggregate_hash,
    PHASH_VERSION,
)
from services.audio_fingerprint import audio_fingerprint, AUDIO_FP_VERSION
from services.comparator import run_duplicate_pipeline
from services.quality_scorer import rank_group, calculate_wasted_space
from services import scan_control, error_log
from api.websocket import manager
from config import settings

router = APIRouter()


# ── helpers ────────────────────────────────────────────────────────────────────

async def _send_status_ws(scan_id: int, payload: dict, gpu_active: bool = False, gpu_name: str | None = None):
    """Convenience wrapper that always injects gpu fields."""
    payload.setdefault("gpu_active", gpu_active)
    payload.setdefault("gpu_name", gpu_name)
    await manager.send_progress(scan_id, payload)


async def _mark_stopped(db, scan: ScanJob, scan_id: int):
    """Mark a scan as stopped and notify clients."""
    scan.status = "stopped"
    scan.completed_at = datetime.now(timezone.utc)
    scan.current_stage = "Scan stopped by user"
    scan.current_file = None
    await db.commit()
    await manager.send_complete(scan_id, {
        "scan_id": scan_id,
        "status": "stopped",
        "duplicate_groups_found": scan.duplicate_groups_found or 0,
        "recoverable_space": scan.recoverable_space or 0,
        "total_files": scan.total_files or 0,
        "message": "Scan stopped by user.",
    })


async def _mark_paused(db, scan: ScanJob, scan_id: int, gpu_active: bool, gpu_name: str | None):
    scan.status = "paused"
    scan.current_stage = "Paused — waiting to resume…"
    await db.commit()
    await _send_status_ws(scan_id, {
        "type": "progress", "scan_id": scan_id,
        "status": "paused",
        "current_stage": "Paused — waiting to resume…",
        "current_file": scan.current_file,
        "progress_percent": scan.progress_percent or 0,
        "total_files": scan.total_files or 0,
        "scanned_files": scan.scanned_files or 0,
        "message": "Scan paused by user.",
    }, gpu_active, gpu_name)


# ── Pipeline ───────────────────────────────────────────────────────────────────

class _ScanStopped(Exception):
    """Raised inside the pipeline when the user requests stop."""


async def run_scan_pipeline(scan_id: int, root_path: str, options: dict):
    """Background task: full scan pipeline (GPU-accelerated, pause/stop aware)."""
    from models.database import async_session
    from services.gpu_detector import get_gpu_info

    # Register this scan for control signals and error collection
    scan_control.register(scan_id)
    error_log.register(scan_id)

    async def _log_err(stage: str, message: str, file_path: str | None = None) -> None:
        """Record a per-file error and broadcast it over the scan WebSocket.

        Failure here must never crash the pipeline — a logging glitch should
        not abort an otherwise-healthy scan.
        """
        try:
            entry = error_log.log(scan_id, stage, message, file_path=file_path)
            await manager.send_error_log(scan_id, entry)
        except Exception:
            pass

    async with async_session() as db:
        try:
            gpu = get_gpu_info()
            gpu_active = gpu.available and gpu.hwaccel_supported
            gpu_name = gpu.gpu_name if gpu_active else None

            # ── Signal helper (used throughout the pipeline) ──────────
            async def _pipeline_check(stage_status: str, stage_label: str):
                """Check pause/stop.  Blocks when paused, raises on stop.

                Call this between batches and after each gather().
                """
                nonlocal scan

                # Stop?
                signal = await scan_control.check_signals(scan_id)
                if signal == "stopped":
                    raise _ScanStopped()

                # Pause requested while batch was running?
                if scan_control.is_paused(scan_id):
                    await _mark_paused(db, scan, scan_id, gpu_active, gpu_name)
                    # This blocks until resumed or stopped
                    signal = await scan_control.check_signals(scan_id)
                    if signal == "stopped":
                        raise _ScanStopped()
                    # Resumed — restore status
                    scan.status = stage_status
                    scan.current_stage = stage_label
                    await db.commit()
                    await _send_status_ws(scan_id, {
                        "type": "progress", "scan_id": scan_id,
                        "status": stage_status,
                        "current_stage": stage_label,
                        "progress_percent": scan.progress_percent or 0,
                        "total_files": scan.total_files or 0,
                        "scanned_files": scan.scanned_files or 0,
                        "message": f"Resumed — {stage_label}",
                    }, gpu_active, gpu_name)

            # ── Step 0: init ─────────────────────────────────────────
            scan = await db.get(ScanJob, scan_id)
            scan.status = "scanning"
            scan.current_stage = "Discovering video files..."
            await db.commit()

            accel_label = f"⚡ GPU: {gpu.gpu_name}" if gpu_active else "CPU mode"
            await _send_status_ws(scan_id, {
                "type": "progress", "scan_id": scan_id,
                "status": "scanning",
                "current_stage": "Discovering video files...",
                "progress_percent": 0, "total_files": 0, "scanned_files": 0,
                "message": f"Searching for video files... ({accel_label})",
            }, gpu_active, gpu_name)

            # ── Step 1: discover ─────────────────────────────────────
            video_paths = discover_videos(root_path)
            total_files = len(video_paths)
            scan.total_files = total_files
            await db.commit()

            if total_files == 0:
                scan.status = "completed"
                scan.completed_at = datetime.now(timezone.utc)
                scan.progress_percent = 100
                scan.current_stage = "No video files found"
                await db.commit()
                await manager.send_complete(scan_id, {
                    "scan_id": scan_id, "status": "completed",
                    "message": "No video files found in the specified directory.",
                })
                return

            await _send_status_ws(scan_id, {
                "type": "progress", "scan_id": scan_id,
                "status": "scanning",
                "current_stage": "Extracting metadata...",
                "progress_percent": 5, "total_files": total_files, "scanned_files": 0,
                "message": f"Found {total_files} video files. Extracting metadata...",
            }, gpu_active, gpu_name)

            # ── Step 1.5: cache lookup + partition (NEW) ─────────────
            scan_started_at = scan.started_at or datetime.now(timezone.utc)
            now = datetime.now(timezone.utc)

            all_file_infos = [get_file_info(p) for p in video_paths]

            # Bulk-load any cache rows for these paths (chunked to keep
            # SQLite IN clauses sane).  Fetched-by-path; filtered-by-key.
            cache_by_key: dict = {}
            unique_paths = list({fi["file_path"] for fi in all_file_infos})
            PATH_CHUNK = 500
            for chunk_start in range(0, len(unique_paths), PATH_CHUNK):
                chunk = unique_paths[chunk_start:chunk_start + PATH_CHUNK]
                result = await db.execute(
                    select(FileCache).where(FileCache.file_path.in_(chunk))
                )
                for c in result.scalars().all():
                    cache_by_key[(c.file_path, c.file_size, c.mtime_ns)] = c

            # Partition into hits (full pipeline output cached) and misses.
            # A cache row only counts as a hit if its perceptual_hashes were
            # written by code at least as new as PHASH_VERSION — otherwise the
            # pHash format / preprocessing changed and the cached value is no
            # longer comparable to fresh extractions.
            hits: list = []      # (file_info, cache)
            misses: list = []    # (file_info, vpath, cache)
            for fi, vpath in zip(all_file_infos, video_paths):
                key = (fi["file_path"], fi["file_size"], fi["mtime_ns"])
                cache = cache_by_key.get(key)
                phash_fresh = (
                    cache is not None
                    and cache.perceptual_hashes
                    and (cache.cache_version or 1) >= PHASH_VERSION
                )
                if phash_fresh:
                    cache.last_seen_at = now
                    hits.append((fi, cache))
                else:
                    if cache is None:
                        cache = FileCache(
                            file_path=key[0], file_size=key[1], mtime_ns=key[2],
                            first_seen_at=now, last_seen_at=now,
                        )
                        db.add(cache)
                    else:
                        # Stale pHashes — clear so they don't leak into the
                        # comparison stage if extraction later fails.
                        cache.last_seen_at = now
                        cache.perceptual_hashes = None
                    misses.append((fi, vpath, cache))

            # Flush so new cache rows have ids before we link VideoFile to them
            await db.commit()

            num_hits = len(hits)
            num_misses = len(misses)

            # Backfill head_tail_xxh3 for cache hits that predate the column.
            # Without this, the byte-identical fast-path can't reach hits,
            # which would mean it only catches duplicates among NEW files.
            hits_needing_htx = [
                (fi, cache) for fi, cache in hits if cache.head_tail_xxh3 is None
            ]
            if hits_needing_htx:
                BACKFILL_BATCH = 64

                async def _backfill_one(fi, cache):
                    cache.head_tail_xxh3 = await asyncio.to_thread(
                        compute_head_tail_hash,
                        fi["file_path"], fi.get("file_size"),
                    )

                for chunk_start in range(0, len(hits_needing_htx), BACKFILL_BATCH):
                    # Honor pause/stop between chunks so a long backfill on a
                    # cache-upgrade rescan can be interrupted by the user.
                    await _pipeline_check("scanning", "Backfilling cache content hashes...")
                    chunk = hits_needing_htx[chunk_start:chunk_start + BACKFILL_BATCH]
                    await asyncio.gather(
                        *(_backfill_one(fi, c) for fi, c in chunk),
                        return_exceptions=True,
                    )
                    # Commit per-chunk so a hard kill mid-backfill doesn't
                    # discard the chunks that already finished.
                    await db.commit()

            # ── Step 2: metadata + thumbnails (BATCH CONCURRENT, misses only) ──
            scan.current_stage = "Extracting metadata..."
            scan.status = "metadata"
            await db.commit()

            thumbnails_dir = os.path.join(os.getcwd(), settings.THUMBNAILS_DIR)
            os.makedirs(thumbnails_dir, exist_ok=True)

            video_records = []

            # Build VideoFile rows directly from cache for hits — no work needed
            for fi, cache in hits:
                video = VideoFile(
                    scan_job_id=scan_id,
                    file_path=fi["file_path"],
                    file_name=fi["file_name"],
                    file_size=fi["file_size"],
                    created_at=fi.get("created_at"),
                    modified_at=fi.get("modified_at"),
                    duration=cache.duration,
                    width=cache.width,
                    height=cache.height,
                    bitrate=cache.bitrate,
                    video_codec=cache.video_codec,
                    audio_codec=cache.audio_codec,
                    fps=cache.fps,
                    audio_channels=cache.audio_channels,
                    audio_sample_rate=cache.audio_sample_rate,
                    thumbnail_path=cache.thumbnail_path,
                    perceptual_hashes=cache.perceptual_hashes,
                    hash_computed=True,
                    file_cache_id=cache.id,
                    cache_hit=True,
                )
                video._meta_video_info = {
                    "width": cache.width or 0,
                    "height": cache.height or 0,
                    "sar_num": cache.sar_num or 1,
                    "sar_den": cache.sar_den or 1,
                    "rotation": cache.rotation or 0,
                }
                video._cache_row = cache
                db.add(video)
                video_records.append(video)

            if gpu_active:
                max_concurrent = options.get("max_concurrent", settings.GPU_MAX_CONCURRENT)
            else:
                max_concurrent = options.get("max_concurrent", settings.MAX_CONCURRENT_FFMPEG)
            sem = asyncio.Semaphore(max_concurrent)

            # Audio fingerprinting is CPU-bound — sharing the GPU-tuned semaphore
            # (often 12) over-saturates the CPU with ffmpeg audio decodes that
            # the GPU can't help with.  Cap at the number of CPU cores.
            cpu_concurrent = options.get(
                "cpu_concurrent",
                min(settings.MAX_CONCURRENT_FFMPEG, os.cpu_count() or 4),
            )

            # Throttle WS updates: 0.5% is small enough that every batch-end
            # progress emission survives (BATCH = max_concurrent*4 ≈ 48
            # files, ~0.4% per batch in stage 2/3), but still drops the
            # occasional duplicate when the per-batch advance happens to
            # round-trip the same value twice.
            _last_ws_pct = -1.0

            def _should_send_ws(new_pct: float) -> bool:
                nonlocal _last_ws_pct
                if new_pct - _last_ws_pct >= 0.5 or new_pct >= 100.0:
                    _last_ws_pct = new_pct
                    return True
                return False

            async def _process_one_miss(idx: int, fi: dict, cache: FileCache):
                """Process metadata + thumbnail for a cache miss; populate cache row."""
                if scan_control.is_stopped(scan_id):
                    return None
                async with sem:
                    if scan_control.is_stopped(scan_id):
                        return None
                    meta = await extract_metadata(fi["file_path"])

                if "error" in meta:
                    return {"_meta_error": meta["error"]}

                # Head+tail content hash for the byte-identical fast-path.
                # ~1 ms per file on SSD; runs off the event loop so it
                # doesn't stall the metadata batch.
                if cache.head_tail_xxh3 is None:
                    cache.head_tail_xxh3 = await asyncio.to_thread(
                        compute_head_tail_hash,
                        fi["file_path"], fi.get("file_size"),
                    )

                if scan_control.is_stopped(scan_id):
                    return None

                thumb_name = f"scan{scan_id}_thumb_{idx}.jpg"
                thumb_path = os.path.join(thumbnails_dir, thumb_name)
                await extract_thumbnail(
                    fi["file_path"], thumb_path,
                    duration=meta.get("duration"),
                    codec=meta.get("video_codec"),
                )
                thumb_url = f"/thumbnails/{thumb_name}" if os.path.exists(thumb_path) else None

                # Persist metadata to the cache row so future scans can skip stage 2
                cache.duration = meta.get("duration")
                cache.width = meta.get("width")
                cache.height = meta.get("height")
                cache.bitrate = meta.get("bitrate")
                cache.video_codec = meta.get("video_codec")
                cache.audio_codec = meta.get("audio_codec")
                cache.fps = meta.get("fps")
                cache.audio_channels = meta.get("audio_channels")
                cache.audio_sample_rate = meta.get("audio_sample_rate")
                cache.sar_num = meta.get("sar_num", 1)
                cache.sar_den = meta.get("sar_den", 1)
                cache.rotation = meta.get("rotation", 0)
                cache.thumbnail_path = thumb_url

                video = VideoFile(
                    scan_job_id=scan_id,
                    file_path=fi["file_path"],
                    file_name=fi["file_name"],
                    file_size=fi["file_size"],
                    created_at=fi.get("created_at"),
                    modified_at=fi.get("modified_at"),
                    duration=meta.get("duration"),
                    width=meta.get("width"),
                    height=meta.get("height"),
                    bitrate=meta.get("bitrate"),
                    video_codec=meta.get("video_codec"),
                    audio_codec=meta.get("audio_codec"),
                    fps=meta.get("fps"),
                    audio_channels=meta.get("audio_channels"),
                    audio_sample_rate=meta.get("audio_sample_rate"),
                    thumbnail_path=thumb_url,
                    file_cache_id=cache.id,
                    cache_hit=False,
                )
                video._meta_video_info = {
                    "width": meta.get("width") or 0,
                    "height": meta.get("height") or 0,
                    "sar_num": meta.get("sar_num", 1),
                    "sar_den": meta.get("sar_den", 1),
                    "rotation": meta.get("rotation", 0),
                }
                video._cache_row = cache
                return video

            # Process misses in batches; cache hits are essentially free
            BATCH = max_concurrent * 4
            for batch_start in range(0, num_misses, BATCH):
                await _pipeline_check("metadata", "Extracting metadata...")

                batch_end = min(batch_start + BATCH, num_misses)
                batch = misses[batch_start:batch_end]

                tasks = [
                    _process_one_miss(num_hits + batch_start + i, fi, cache)
                    for i, (fi, _vpath, cache) in enumerate(batch)
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                await _pipeline_check("metadata", "Extracting metadata...")

                for r_idx, r in enumerate(results):
                    if isinstance(r, Exception):
                        failed_path = batch[r_idx][0]["file_path"] if r_idx < len(batch) else None
                        await _log_err("metadata", f"metadata extraction failed: {r}", failed_path)
                        continue
                    if isinstance(r, dict) and "_meta_error" in r:
                        failed_path = batch[r_idx][0]["file_path"] if r_idx < len(batch) else None
                        if failed_path and not scan_control.is_stopped(scan_id):
                            await _log_err("metadata", f"ffprobe: {r['_meta_error']}", failed_path)
                        continue
                    if r is not None:
                        db.add(r)
                        video_records.append(r)
                    else:
                        # stopped mid-way
                        failed_path = batch[r_idx][0]["file_path"] if r_idx < len(batch) else None
                        if failed_path and not scan_control.is_stopped(scan_id):
                            await _log_err("metadata", "skipped (stop requested)", failed_path)

                processed = num_hits + batch_end
                scan.scanned_files = processed
                miss_progress = batch_end / num_misses if num_misses > 0 else 1.0
                scan.progress_percent = round(5 + miss_progress * 40, 1)
                scan.current_file = batch[-1][0]["file_name"]
                await db.commit()

                if _should_send_ws(scan.progress_percent):
                    await _send_status_ws(scan_id, {
                        "type": "progress", "scan_id": scan_id,
                        "status": "metadata",
                        "current_stage": "Extracting metadata...",
                        "current_file": scan.current_file,
                        "progress_percent": scan.progress_percent,
                        "total_files": total_files,
                        "scanned_files": processed,
                        "message": f"Processed {processed}/{total_files} files ({num_hits} from cache)",
                    }, gpu_active, gpu_name)

            if num_misses == 0:
                # All cache hits — jump straight to end of stage 2
                scan.scanned_files = num_hits
                scan.progress_percent = 45.0

            await db.commit()

            # ── Step 2.5: byte-identical fast-path clustering ─────────
            # Files sharing (file_size, head_tail_xxh3) are almost certainly
            # byte-identical.  Within each such cluster we pick one
            # representative video to do the heavy work (pHash + audio FP);
            # the rest are "followers" that inherit the representative's
            # outputs after stage 3/4b.  This preserves transitive matching
            # (a follower can still link to a transcode via the shared hash)
            # while skipping ~half the decode work per byte-identical cluster.
            from collections import defaultdict as _dd
            htx_buckets = _dd(list)
            for v in video_records:
                cache_row = getattr(v, "_cache_row", None)
                htx = cache_row.head_tail_xxh3 if cache_row is not None else None
                if htx and v.file_size:
                    htx_buckets[(int(v.file_size), htx)].append(v)

            phash_follower_to_rep: dict = {}  # file_path → rep VideoFile
            byte_identical_clusters: list = []  # list of [VideoFile, ...]
            for vids in htx_buckets.values():
                if len(vids) < 2:
                    continue
                vids_sorted = sorted(vids, key=lambda v: v.file_path)
                rep = vids_sorted[0]
                byte_identical_clusters.append(vids_sorted)
                for follower in vids_sorted[1:]:
                    phash_follower_to_rep[follower.file_path] = rep

            num_byte_identical_followers = len(phash_follower_to_rep)

            # ── Step 3: perceptual hashes (BATCH CONCURRENT, misses only) ─
            scan.current_stage = "Computing perceptual hashes..."
            scan.status = "hashing"
            await db.commit()
            await _send_status_ws(scan_id, {
                "type": "progress", "scan_id": scan_id,
                "status": "hashing",
                "current_stage": "Computing perceptual hashes...",
                "progress_percent": 45,
                "total_files": total_files,
                "scanned_files": len(video_records),
                "message": "Extracting frames and computing hashes...",
            }, gpu_active, gpu_name)

            num_frames = options.get("key_frames_count", settings.KEY_FRAMES_COUNT)

            # Pre-screen by duration: a video with a unique duration can never
            # match another video, so extracting its pHashes is pure waste.
            # Only hash videos that share a duration bucket with at least one
            # other video (same gating logic already used for audio FP below).
            from services.comparator import group_by_duration as _pre_group
            _hash_tol = options.get(
                "duration_tolerance", settings.DURATION_TOLERANCE_SECONDS,
            )
            _hash_pre = [
                {"file_path": v.file_path, "duration": v.duration}
                for v in video_records
            ]
            _hash_candidate_paths = {
                vd["file_path"]
                for g in _pre_group(_hash_pre, _hash_tol)
                for vd in g
            }

            # Hash only videos that (a) don't already have hashes (cache miss)
            # AND (b) are in a duration candidate group AND (c) are not a
            # byte-identical follower (will inherit the rep's hashes below).
            videos_to_hash = [
                v for v in video_records
                if not v.perceptual_hashes
                and v.file_path in _hash_candidate_paths
                and v.file_path not in phash_follower_to_rep
            ]
            num_to_hash = len(videos_to_hash)
            num_cached_hashes = len(video_records) - num_to_hash

            async def _hash_one(video):
                """Hash a single video (runs concurrently under semaphore)."""
                if scan_control.is_stopped(scan_id):
                    return None
                async with sem:
                    if scan_control.is_stopped(scan_id):
                        return None
                    vinfo = getattr(video, "_meta_video_info", None)
                    return await extract_and_hash(
                        video.file_path,
                        num_frames,
                        duration=video.duration,
                        codec=video.video_codec,
                        video_info=vinfo,
                    )

            HASH_BATCH = max_concurrent * 4
            hash_done = 0
            for batch_start in range(0, num_to_hash, HASH_BATCH):
                await _pipeline_check("hashing", "Computing perceptual hashes...")

                batch_end = min(batch_start + HASH_BATCH, num_to_hash)
                batch_videos = videos_to_hash[batch_start:batch_end]

                tasks = [_hash_one(v) for v in batch_videos]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                await _pipeline_check("hashing", "Computing perceptual hashes...")

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        await _log_err(
                            "hashing", f"hash computation crashed: {result}",
                            batch_videos[i].file_path,
                        )
                        continue
                    if result and result.get("hashes"):
                        hashes_json = json.dumps(result["hashes"])
                        batch_videos[i].perceptual_hashes = hashes_json
                        batch_videos[i].hash_computed = True
                        # Persist to cache row so future scans skip stage 3
                        cache_row = getattr(batch_videos[i], "_cache_row", None)
                        if cache_row is not None:
                            cache_row.perceptual_hashes = hashes_json
                            cache_row.cache_version = max(
                                cache_row.cache_version or 1, PHASH_VERSION,
                            )
                    else:
                        # Frame extraction yielded zero usable hashes — most
                        # often a timeout on long network files or a codec
                        # ffmpeg can't decode.  Capture the in-helper detail
                        # if present so the UI shows a useful message.
                        err_msg = (result or {}).get("error") or "no frames extracted (timeout or codec issue)"
                        await _log_err("hashing", err_msg, batch_videos[i].file_path)

                hash_done = batch_end
                miss_progress = hash_done / num_to_hash if num_to_hash > 0 else 1.0
                scan.progress_percent = round(45 + miss_progress * 30, 1)
                scan.current_file = batch_videos[-1].file_name
                await db.commit()

                if _should_send_ws(scan.progress_percent):
                    await _send_status_ws(scan_id, {
                        "type": "progress", "scan_id": scan_id,
                        "status": "hashing",
                        "current_stage": "Computing perceptual hashes...",
                        "current_file": scan.current_file,
                        "progress_percent": scan.progress_percent,
                        "total_files": total_files,
                        "scanned_files": num_cached_hashes + hash_done,
                        "message": f"Hashed {hash_done}/{num_to_hash} files ({num_cached_hashes} from cache)",
                    }, gpu_active, gpu_name)

            if num_to_hash == 0:
                scan.progress_percent = 75.0

            # Propagate the byte-identical-cluster representative's pHashes
            # to every follower in the cluster.  Followers stay un-hashed
            # by ffmpeg but participate in pHash comparison as if they
            # were hashed (because they ARE the same bytes).
            if phash_follower_to_rep:
                video_by_path = {v.file_path: v for v in video_records}
                for follower_path, rep in phash_follower_to_rep.items():
                    follower = video_by_path.get(follower_path)
                    if follower is None or not rep.perceptual_hashes:
                        continue
                    follower.perceptual_hashes = rep.perceptual_hashes
                    follower.hash_computed = True
                    cache_row = getattr(follower, "_cache_row", None)
                    if cache_row is not None:
                        cache_row.perceptual_hashes = rep.perceptual_hashes
                        cache_row.cache_version = max(
                            cache_row.cache_version or 1, PHASH_VERSION,
                        )

            # Compute the aggregate (per-bit majority) hash for every video
            # that has perceptual_hashes.  This becomes the screening key
            # for the FAISS binary index in the comparator.  Cached values
            # are accepted if their cache_version is at least PHASH_VERSION
            # (same gate as perceptual_hashes themselves).
            _agg_dirty = 0
            for v in video_records:
                if not v.perceptual_hashes:
                    continue
                cache_row = getattr(v, "_cache_row", None)
                cached_agg = cache_row.aggregate_hash if cache_row is not None else None
                cached_ver = (cache_row.cache_version or 1) if cache_row is not None else 1
                if cached_agg and cached_ver >= PHASH_VERSION:
                    v._aggregate_hash = cached_agg
                    continue
                try:
                    hashes_list = json.loads(v.perceptual_hashes)
                except (ValueError, TypeError):
                    continue
                agg = compute_aggregate_hash(hashes_list)
                if agg is None:
                    continue
                v._aggregate_hash = agg
                if cache_row is not None:
                    cache_row.aggregate_hash = agg
                    # Defense in depth: this writer always runs in the same
                    # scan as the pHash writer (which already bumped the
                    # version), so the bump is redundant today.  Make it
                    # explicit so a future refactor that recomputes
                    # aggregates from cached hashes can't leave a row with
                    # a fresh aggregate but a stale cache_version.
                    cache_row.cache_version = max(
                        cache_row.cache_version or 1, PHASH_VERSION,
                    )
                    _agg_dirty += 1
                    # Periodic flush so a kill mid-compute persists the
                    # aggregates already calculated.  256 rows ≈ one
                    # SQLite UPDATE batch — cheap.
                    if _agg_dirty >= 256:
                        await db.commit()
                        _agg_dirty = 0

            await db.commit()

            # ── Final stop/pause check before heavy comparison ──
            await _pipeline_check("comparing", "Comparing videos for duplicates...")

            # ── Step 4: compare and find duplicates ──────────────────
            scan.current_stage = "Comparing videos..."
            scan.status = "comparing"
            scan.progress_percent = 75
            await db.commit()
            await _send_status_ws(scan_id, {
                "type": "progress", "scan_id": scan_id,
                "status": "comparing",
                "current_stage": "Comparing videos for duplicates...",
                "progress_percent": 75,
                "total_files": total_files,
                "scanned_files": len(video_records),
                "message": "Running duplicate detection pipeline...",
            }, gpu_active, gpu_name)

            # ── Step 3.5: audio fingerprints (BATCH, only for candidates) ──
            # Only fingerprint videos that are in a duration group (potential
            # duplicates).  Videos with unique durations can never match, so
            # fingerprinting them is wasted work.
            from services.comparator import group_by_duration as _pre_group
            duration_tolerance = options.get("duration_tolerance", settings.DURATION_TOLERANCE_SECONDS)

            _pre_video_data = [
                {"file_path": v.file_path, "duration": v.duration}
                for v in video_records
            ]
            _pre_groups = _pre_group(_pre_video_data, duration_tolerance)
            _candidate_paths: set = set()
            for g in _pre_groups:
                for vd in g:
                    _candidate_paths.add(vd["file_path"])

            scan.current_stage = "Computing audio fingerprints..."
            await db.commit()
            await _send_status_ws(scan_id, {
                "type": "progress", "scan_id": scan_id,
                "status": "comparing",
                "current_stage": "Computing audio fingerprints...",
                "progress_percent": 76,
                "total_files": total_files,
                "scanned_files": len(video_records),
                "message": f"Extracting audio fingerprints for {len(_candidate_paths)} candidate files...",
            }, gpu_active, gpu_name)

            audio_fps: dict = {}   # file_path → List[float]
            audio_sem = asyncio.Semaphore(cpu_concurrent)

            candidate_list = [v for v in video_records if v.file_path in _candidate_paths]

            # Prefill audio_fps from the cache, falling through to fingerprinting
            # for any candidate without a usable cached value.  Reject cached
            # fingerprints whose cache_version predates AUDIO_FP_VERSION (the
            # sampling rule changed, so old full-track 64-point profiles can't
            # be correlated against new middle-60s 64-point profiles).
            # Byte-identical followers are skipped — they inherit the rep's
            # fingerprint after the extraction batch completes.
            files_needing_fp: list = []
            for v in candidate_list:
                if v.file_path in phash_follower_to_rep:
                    continue
                cache_row = getattr(v, "_cache_row", None)
                cached_fp = cache_row.audio_fp if cache_row is not None else None
                cached_ver = (cache_row.cache_version or 1) if cache_row is not None else 1
                if cached_fp and cached_ver >= AUDIO_FP_VERSION:
                    try:
                        audio_fps[v.file_path] = json.loads(cached_fp)
                        continue
                    except (ValueError, TypeError):
                        pass
                files_needing_fp.append(v)

            num_cached_fps = len(candidate_list) - len(files_needing_fp)

            async def _fp_one_v(video):
                if scan_control.is_stopped(scan_id):
                    return video, []
                async with audio_sem:
                    if scan_control.is_stopped(scan_id):
                        return video, []
                    fp = await audio_fingerprint(
                        video.file_path, duration=video.duration,
                    )
                    return video, fp

            FP_BATCH = cpu_concurrent * 4
            for batch_start in range(0, len(files_needing_fp), FP_BATCH):
                await _pipeline_check("comparing", "Computing audio fingerprints...")

                batch_end = min(batch_start + FP_BATCH, len(files_needing_fp))
                tasks = [_fp_one_v(v) for v in files_needing_fp[batch_start:batch_end]]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                await _pipeline_check("comparing", "Computing audio fingerprints...")

                for r in results:
                    if isinstance(r, Exception):
                        await _log_err("audio_fp", f"audio fingerprint crashed: {r}")
                        continue
                    video, fp = r
                    if fp:
                        audio_fps[video.file_path] = fp
                        cache_row = getattr(video, "_cache_row", None)
                        if cache_row is not None:
                            cache_row.audio_fp = json.dumps(fp)
                            cache_row.cache_version = max(
                                cache_row.cache_version or 1, AUDIO_FP_VERSION,
                            )
                    elif not scan_control.is_stopped(scan_id):
                        await _log_err(
                            "audio_fp", "audio fingerprint produced no samples",
                            video.file_path,
                        )

                # Commit per-batch so a hard kill mid-stage doesn't discard
                # the audio fingerprints already extracted this batch — they
                # are the slow part (60 s middle-of-file PCM decode each).
                await db.commit()

            # Propagate audio FP from byte-identical representatives to
            # their followers, persisting to each follower's cache row.
            if phash_follower_to_rep:
                video_by_path = {v.file_path: v for v in video_records}
                for follower_path, rep in phash_follower_to_rep.items():
                    rep_fp = audio_fps.get(rep.file_path)
                    if not rep_fp:
                        continue
                    audio_fps[follower_path] = rep_fp
                    follower = video_by_path.get(follower_path)
                    if follower is None:
                        continue
                    cache_row = getattr(follower, "_cache_row", None)
                    if cache_row is not None:
                        cache_row.audio_fp = json.dumps(rep_fp)
                        cache_row.cache_version = max(
                            cache_row.cache_version or 1, AUDIO_FP_VERSION,
                        )

            video_data = []
            video_id_map = {}
            for v in video_records:
                hashes = json.loads(v.perceptual_hashes) if v.perceptual_hashes else []
                vd = {
                    "id": v.id,
                    "file_path": v.file_path,
                    "duration": v.duration,
                    "hashes": hashes,
                    "aggregate_hash": getattr(v, "_aggregate_hash", None),
                    "audio_fp": audio_fps.get(v.file_path, []),
                    "width": v.width,
                    "height": v.height,
                    "bitrate": v.bitrate,
                    "video_codec": v.video_codec,
                    "file_size": v.file_size,
                    "fps": v.fps,
                }
                video_data.append(vd)
                video_id_map[v.file_path] = v

            hash_threshold = options.get("hash_threshold", settings.HASH_SIMILARITY_THRESHOLD)
            duration_tolerance = options.get("duration_tolerance", settings.DURATION_TOLERANCE_SECONDS)

            duplicate_groups = run_duplicate_pipeline(
                video_data,
                duration_tolerance=duration_tolerance,
                hash_threshold=hash_threshold,
            )

            # ── Step 5: save groups + quality scores ─────────────────
            scan.progress_percent = 90
            scan.current_stage = "Scoring quality and ranking..."
            await db.commit()

            total_recoverable = 0

            for dg_data in duplicate_groups:
                ranked = rank_group(dg_data["videos"])
                wasted = calculate_wasted_space(ranked)
                total_recoverable += wasted

                group = DuplicateGroup(
                    scan_job_id=scan_id,
                    similarity_score=dg_data["similarity_score"],
                    total_wasted_space=wasted,
                    file_count=len(ranked),
                    best_file_id=video_id_map.get(ranked[0]["file_path"], None)
                                and video_id_map[ranked[0]["file_path"]].id,
                )
                db.add(group)
                await db.flush()

                for vid_data in ranked:
                    vid_path = vid_data["file_path"]
                    if vid_path in video_id_map:
                        v = video_id_map[vid_path]
                        v.duplicate_group_id = group.id
                        v.quality_score = vid_data["quality_score"]
                        v.is_best_quality = vid_data["is_best_quality"]

            await db.commit()

            # ── Complete ─────────────────────────────────────────────
            scan.status = "completed"
            scan.completed_at = datetime.now(timezone.utc)
            scan.progress_percent = 100
            scan.duplicate_groups_found = len(duplicate_groups)
            scan.recoverable_space = total_recoverable
            scan.current_stage = "Scan complete!"
            scan.current_file = None
            await db.commit()

            await manager.send_complete(scan_id, {
                "scan_id": scan_id,
                "status": "completed",
                "duplicate_groups_found": len(duplicate_groups),
                "recoverable_space": total_recoverable,
                "total_files": total_files,
                "message": f"Scan complete! Found {len(duplicate_groups)} duplicate groups.",
            })

            # Sweep stale cache rows under this root that we didn't see
            # this scan (file moved or deleted between runs).
            try:
                normalized_root = str(Path(root_path).resolve())
                root_prefix = normalized_root + os.sep
                await db.execute(
                    delete(FileCache).where(
                        FileCache.last_seen_at < scan_started_at,
                        FileCache.file_path.like(root_prefix + "%"),
                    )
                )
                await db.commit()
            except Exception as e:
                await _log_err("cache_sweep", f"error pruning stale entries: {e}")

        except _ScanStopped:
            # Clean stop requested by user — mark as stopped
            # (may already be marked by the stop endpoint; re-fetch to check)
            scan = await db.get(ScanJob, scan_id)
            if scan and scan.status != "stopped":
                await _mark_stopped(db, scan, scan_id)

        except Exception as e:
            scan = await db.get(ScanJob, scan_id)
            if scan:
                scan.status = "failed"
                scan.error_message = str(e)
                scan.completed_at = datetime.now(timezone.utc)
                await db.commit()

            # Also surface this fatal failure into the per-scan error log
            # so it sits alongside the per-file errors in the UI's panel.
            await _log_err("pipeline", f"scan crashed: {e}")

            await manager.send_progress(scan_id, {
                "type": "error", "scan_id": scan_id,
                "status": "failed",
                "message": f"Scan failed: {str(e)}",
            })

        finally:
            scan_control.unregister(scan_id)
            # Defer error_log cleanup briefly — a UI client may still want
            # to fetch the backlog after the scan-complete WS arrives.
            # Schedule the unregister as a fire-and-forget task with a
            # short delay so the buffer survives the immediate reconnect.
            async def _delayed_error_unregister() -> None:
                await asyncio.sleep(30)
                error_log.unregister(scan_id)
            asyncio.create_task(_delayed_error_unregister())
            # ── Auto-start next queued scan ──────────────────────────
            await _start_next_queued()


TERMINAL_STATUSES = ("completed", "failed", "stopped")
ACTIVE_STATUSES = ("pending", "scanning", "metadata", "hashing", "comparing", "paused")


async def _has_active_scan(db: AsyncSession) -> bool:
    """Check if any scan is currently running (not queued, not terminal)."""
    result = await db.execute(
        select(func.count(ScanJob.id)).where(ScanJob.status.in_(ACTIVE_STATUSES))
    )
    return (result.scalar() or 0) > 0


async def _start_next_queued():
    """Pick the oldest queued scan and launch it."""
    from models.database import async_session
    async with async_session() as db:
        # Only start if nothing else is active
        if await _has_active_scan(db):
            return
        result = await db.execute(
            select(ScanJob)
            .where(ScanJob.status == "queued")
            .order_by(ScanJob.started_at.asc())
            .limit(1)
        )
        next_scan = result.scalar_one_or_none()
        if not next_scan:
            return
        next_scan.status = "pending"
        await db.commit()
        options = json.loads(next_scan.options) if next_scan.options else {}
        asyncio.create_task(run_scan_pipeline(next_scan.id, next_scan.root_path, options))


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/scan")
async def start_scan(
    request: ScanRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Start a new duplicate video scan (or queue it if one is already running)."""
    if not os.path.isdir(request.path):
        raise HTTPException(status_code=400, detail=f"Invalid directory: {request.path}")

    active = await _has_active_scan(db)
    initial_status = "queued" if active else "pending"

    scan = ScanJob(
        root_path=request.path,
        status=initial_status,
        options=json.dumps(request.options.model_dump() if request.options else {}),
    )
    db.add(scan)
    await db.commit()
    await db.refresh(scan)

    if not active:
        options = request.options.model_dump() if request.options else {}
        background_tasks.add_task(run_scan_pipeline, scan.id, request.path, options)

    return {"id": scan.id, "status": initial_status, "message": "Scan queued" if active else "Scan started"}


@router.get("/scan/{scan_id}/status")
async def get_scan_status(scan_id: int, db: AsyncSession = Depends(get_db)):
    """Get current scan status."""
    scan = await db.get(ScanJob, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return ScanStatusResponse.model_validate(scan)


# ── Pause / Resume / Stop ─────────────────────────────────────────────────────

@router.post("/scan/{scan_id}/pause")
async def pause_scan(scan_id: int, db: AsyncSession = Depends(get_db)):
    """Pause a running scan."""
    scan = await db.get(ScanJob, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    if scan.status in ("completed", "failed", "stopped"):
        raise HTTPException(status_code=400, detail=f"Cannot pause a {scan.status} scan")

    if scan.status == "paused":
        return {"scan_id": scan_id, "status": "paused", "message": "Scan is already paused"}

    ok = scan_control.pause(scan_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Scan is not currently running")

    # Immediately update DB + push WS so the frontend sees "paused" right away
    # (the pipeline will also call _mark_paused when it next checks signals,
    #  but that can take seconds while a batch is in-flight)
    scan.status = "paused"
    scan.current_stage = "Paused — waiting to resume…"
    await db.commit()
    await manager.send_progress(scan_id, {
        "type": "progress",
        "scan_id": scan_id,
        "status": "paused",
        "current_stage": "Paused — waiting to resume…",
        "current_file": scan.current_file,
        "progress_percent": scan.progress_percent or 0,
        "total_files": scan.total_files or 0,
        "scanned_files": scan.scanned_files or 0,
        "message": "Scan paused by user.",
    })

    return {"scan_id": scan_id, "status": "paused", "message": "Scan paused"}


@router.post("/scan/{scan_id}/resume")
async def resume_scan(scan_id: int, db: AsyncSession = Depends(get_db)):
    """Resume a paused scan."""
    scan = await db.get(ScanJob, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    if scan.status != "paused":
        raise HTTPException(status_code=400, detail=f"Scan is not paused (status: {scan.status})")

    ok = scan_control.resume(scan_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Scan signal not found")

    # Immediately update DB + push WS so the frontend sees "running" right away
    scan.status = "scanning"
    scan.current_stage = "Resuming…"
    await db.commit()
    await manager.send_progress(scan_id, {
        "type": "progress",
        "scan_id": scan_id,
        "status": "scanning",
        "current_stage": "Resuming…",
        "current_file": scan.current_file,
        "progress_percent": scan.progress_percent or 0,
        "total_files": scan.total_files or 0,
        "scanned_files": scan.scanned_files or 0,
        "message": "Scan resumed.",
    })

    return {"scan_id": scan_id, "status": "scanning", "message": "Scan resumed"}


@router.post("/scan/{scan_id}/stop")
async def stop_scan(scan_id: int, db: AsyncSession = Depends(get_db)):
    """Stop a running or paused scan."""
    scan = await db.get(ScanJob, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    if scan.status in ("completed", "failed", "stopped"):
        raise HTTPException(status_code=400, detail=f"Scan is already {scan.status}")

    ok = scan_control.stop(scan_id)

    # Immediately update DB + push WS so the frontend sees "stopped" right away
    scan.status = "stopped"
    scan.completed_at = datetime.now(timezone.utc)
    scan.current_stage = "Scan stopped by user"
    scan.current_file = None
    await db.commit()
    await manager.send_complete(scan_id, {
        "scan_id": scan_id,
        "status": "stopped",
        "duplicate_groups_found": scan.duplicate_groups_found or 0,
        "recoverable_space": scan.recoverable_space or 0,
        "total_files": scan.total_files or 0,
        "message": "Scan stopped by user.",
    })

    return {"scan_id": scan_id, "status": "stopped", "message": "Scan stopped"}


_ACTIVE_STATUSES = ("pending", "scanning", "metadata", "hashing", "comparing", "paused")


async def _delete_scan_rows(db: AsyncSession, scan_ids: list[int]) -> None:
    """Bulk-delete a scan and its dependents (video_files, duplicate_groups).

    Uses raw DELETE statements rather than ORM cascade so it scales when
    wiping many scans in one shot.  Order matters: video_files FK both
    duplicate_groups and scan_jobs, so they go first.
    """
    if not scan_ids:
        return
    await db.execute(delete(VideoFile).where(VideoFile.scan_job_id.in_(scan_ids)))
    await db.execute(delete(DuplicateGroup).where(DuplicateGroup.scan_job_id.in_(scan_ids)))
    await db.execute(delete(ScanJob).where(ScanJob.id.in_(scan_ids)))


@router.delete("/scan/{scan_id}")
async def delete_scan(scan_id: int, db: AsyncSession = Depends(get_db)):
    """Cancel a queued scan OR delete a finished scan from history.

    Rejects scans that are currently running/paused — use POST /scan/{id}/stop
    first, then DELETE.  Cascades to video_files and duplicate_groups for
    that scan; the cross-scan file_cache is left intact.
    """
    scan = await db.get(ScanJob, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    if scan.status in _ACTIVE_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete an active scan (status: {scan.status}). Stop it first.",
        )

    was_queued = scan.status == "queued"
    await _delete_scan_rows(db, [scan_id])
    await db.commit()

    # Defensive: if the scan was somehow still registered for signals
    # (shouldn't be for terminal/queued), unregister so we don't leak.
    scan_control.unregister(scan_id)

    return {
        "scan_id": scan_id,
        "message": "Queued scan cancelled" if was_queued else "Scan deleted",
    }


@router.delete("/scans")
async def delete_all_scans(db: AsyncSession = Depends(get_db)):
    """Delete every non-active scan (queued + terminal) from history.

    Active scans (pending/scanning/metadata/hashing/comparing/paused) are
    left alone — the caller must stop them first.  Returns the count of
    scans actually removed.
    """
    result = await db.execute(
        select(ScanJob.id).where(~ScanJob.status.in_(_ACTIVE_STATUSES))
    )
    scan_ids = [row[0] for row in result.all()]

    if not scan_ids:
        return {"deleted_count": 0, "message": "No scans to delete"}

    await _delete_scan_rows(db, scan_ids)
    await db.commit()

    for sid in scan_ids:
        scan_control.unregister(sid)

    return {
        "deleted_count": len(scan_ids),
        "message": f"Deleted {len(scan_ids)} scan(s) from history",
    }


# ── WebSocket ──────────────────────────────────────────────────────────────────

@router.websocket("/scan/{scan_id}/ws")
async def scan_websocket(websocket: WebSocket, scan_id: int):
    """WebSocket endpoint for real-time scan progress."""
    await manager.connect(websocket, scan_id)
    try:
        from models.database import async_session
        async with async_session() as db:
            scan = await db.get(ScanJob, scan_id)
            if scan:
                msg_type = "complete" if scan.status in ("completed", "failed", "stopped") else "progress"
                await websocket.send_json({
                    "type": msg_type,
                    "scan_id": scan_id,
                    "status": scan.status,
                    "current_stage": scan.current_stage or "",
                    "current_file": scan.current_file,
                    "progress_percent": scan.progress_percent or 0,
                    "total_files": scan.total_files or 0,
                    "scanned_files": scan.scanned_files or 0,
                    "duplicate_groups_found": scan.duplicate_groups_found or 0,
                    "recoverable_space": scan.recoverable_space or 0,
                    "message": "",
                })

        # Replay the in-memory error log so a late subscriber can see what
        # has already gone wrong this scan — otherwise the user only sees
        # errors that fire strictly after they open the page.
        for entry in error_log.get_recent(scan_id):
            try:
                await websocket.send_json({"type": "error_log", **entry})
            except Exception:
                break

        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket, scan_id)


@router.get("/scans")
async def list_scans(db: AsyncSession = Depends(get_db)):
    """List all scan jobs."""
    result = await db.execute(
        select(ScanJob).order_by(ScanJob.started_at.desc())
    )
    scans = result.scalars().all()
    return [ScanStatusResponse.model_validate(s) for s in scans]


@router.get("/browse")
async def browse_directory(path: Optional[str] = Query(None)):
    """Browse filesystem directories for folder selection."""
    # No path provided: return drive letters (Windows) or root (Unix)
    if not path:
        if platform.system() == "Windows":
            drives = []
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    drives.append({"name": f"{letter}:", "path": drive, "is_dir": True})
            return {"current_path": "", "parent_path": None, "entries": drives}
        else:
            path = "/"

    # Normalize path
    path = os.path.normpath(path)

    if not os.path.isdir(path):
        raise HTTPException(status_code=400, detail=f"Not a valid directory: {path}")

    # Compute parent
    parent = os.path.dirname(path)
    if parent == path:
        # At root — on Windows, return None to show drive list
        parent = None if platform.system() == "Windows" else None
    elif platform.system() == "Windows" and len(parent) == 2 and parent[1] == ":":
        parent = parent + "\\"

    entries = []
    try:
        for entry in sorted(os.scandir(path), key=lambda e: e.name.lower()):
            if entry.is_dir(follow_symlinks=False):
                try:
                    entries.append({
                        "name": entry.name,
                        "path": entry.path,
                        "is_dir": True,
                    })
                except PermissionError:
                    pass
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Access denied: {path}")

    return {"current_path": path, "parent_path": parent, "entries": entries}
