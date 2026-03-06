"""Scan API endpoints with pause / stop support."""
from __future__ import annotations

import asyncio
import json
import os
import platform
import string
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from models.database import get_db, ScanJob, VideoFile, DuplicateGroup
from models.schemas import ScanRequest, ScanStatusResponse
from services.scanner import discover_videos, get_file_info
from services.metadata import extract_metadata
from services.hasher import extract_and_hash, extract_thumbnail
from services.audio_fingerprint import audio_fingerprint
from services.comparator import run_duplicate_pipeline
from services.quality_scorer import rank_group, calculate_wasted_space
from services import scan_control
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

    # Register this scan for control signals
    scan_control.register(scan_id)

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

            # ── Step 2: metadata + thumbnails (BATCH CONCURRENT) ─────
            scan.current_stage = "Extracting metadata..."
            scan.status = "metadata"
            await db.commit()

            thumbnails_dir = os.path.join(os.getcwd(), settings.THUMBNAILS_DIR)
            os.makedirs(thumbnails_dir, exist_ok=True)

            video_records = []
            if gpu_active:
                max_concurrent = options.get("max_concurrent", settings.GPU_MAX_CONCURRENT)
            else:
                max_concurrent = options.get("max_concurrent", settings.MAX_CONCURRENT_FFMPEG)
            sem = asyncio.Semaphore(max_concurrent)

            # Throttle WS updates: only send when progress changes by ≥2%
            _last_ws_pct = 0.0

            def _should_send_ws(new_pct: float) -> bool:
                nonlocal _last_ws_pct
                if new_pct - _last_ws_pct >= 2.0 or new_pct >= 100.0:
                    _last_ws_pct = new_pct
                    return True
                return False

            async def _process_one_meta(idx: int, vpath: str):
                """Process metadata + thumbnail for a single video (runs concurrently)."""
                # Bail immediately if stop was requested while queued
                if scan_control.is_stopped(scan_id):
                    return None
                async with sem:
                    # Check again after acquiring (may have waited on semaphore)
                    if scan_control.is_stopped(scan_id):
                        return None
                    file_info = get_file_info(vpath)
                    meta = await extract_metadata(vpath)

                if "error" in meta:
                    return None

                # Skip thumbnail if stop requested during metadata extraction
                if scan_control.is_stopped(scan_id):
                    return None

                thumb_name = f"scan{scan_id}_thumb_{idx}.jpg"
                thumb_path = os.path.join(thumbnails_dir, thumb_name)
                # Pass duration + codec to skip 2 redundant ffprobe calls
                await extract_thumbnail(
                    vpath, thumb_path,
                    duration=meta.get("duration"),
                    codec=meta.get("video_codec"),
                )

                video = VideoFile(
                    scan_job_id=scan_id,
                    file_path=file_info["file_path"],
                    file_name=file_info["file_name"],
                    file_size=file_info["file_size"],
                    created_at=file_info.get("created_at"),
                    modified_at=file_info.get("modified_at"),
                    duration=meta.get("duration"),
                    width=meta.get("width"),
                    height=meta.get("height"),
                    bitrate=meta.get("bitrate"),
                    video_codec=meta.get("video_codec"),
                    audio_codec=meta.get("audio_codec"),
                    fps=meta.get("fps"),
                    audio_channels=meta.get("audio_channels"),
                    audio_sample_rate=meta.get("audio_sample_rate"),
                    thumbnail_path=f"/thumbnails/{thumb_name}" if os.path.exists(thumb_path) else None,
                )
                # Stash SAR/rotation for the hashing step (avoids another ffprobe)
                video._meta_video_info = {
                    "width": meta.get("width") or 0,
                    "height": meta.get("height") or 0,
                    "sar_num": meta.get("sar_num", 1),
                    "sar_den": meta.get("sar_den", 1),
                    "rotation": meta.get("rotation", 0),
                }
                return video

            # Process in batches to allow pause/stop checks between batches
            BATCH = max_concurrent * 4
            for batch_start in range(0, total_files, BATCH):
                # ── check pause / stop between batches ──
                await _pipeline_check("metadata", "Extracting metadata...")

                batch_end = min(batch_start + BATCH, total_files)
                batch_paths = video_paths[batch_start:batch_end]

                tasks = [
                    _process_one_meta(batch_start + i, vpath)
                    for i, vpath in enumerate(batch_paths)
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Post-batch: check if stop/pause arrived during the batch
                await _pipeline_check("metadata", "Extracting metadata...")

                for r in results:
                    if isinstance(r, Exception):
                        print(f"Error processing video: {r}")
                        continue
                    if r is not None:
                        db.add(r)
                        video_records.append(r)

                scan.scanned_files = batch_end
                scan.progress_percent = round((batch_end / total_files) * 40 + 5, 1)
                scan.current_file = os.path.basename(batch_paths[-1])
                await db.commit()

                if _should_send_ws(scan.progress_percent):
                    await _send_status_ws(scan_id, {
                        "type": "progress", "scan_id": scan_id,
                        "status": "metadata",
                        "current_stage": "Extracting metadata...",
                        "current_file": scan.current_file,
                        "progress_percent": scan.progress_percent,
                        "total_files": total_files,
                        "scanned_files": batch_end,
                        "message": f"Processed {batch_end}/{total_files} files",
                    }, gpu_active, gpu_name)

            await db.commit()

            # ── Step 3: perceptual hashes (BATCH CONCURRENT) ─────────
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
            num_videos = len(video_records)

            async def _hash_one(video):
                """Hash a single video (runs concurrently under semaphore)."""
                if scan_control.is_stopped(scan_id):
                    return None
                async with sem:
                    if scan_control.is_stopped(scan_id):
                        return None
                    # Pass pre-computed video_info to skip redundant ffprobe
                    vinfo = getattr(video, "_meta_video_info", None)
                    return await extract_and_hash(
                        video.file_path,
                        num_frames,
                        duration=video.duration,
                        codec=video.video_codec,
                        video_info=vinfo,
                    )

            # Process in batches for pause/stop checks + progress updates
            HASH_BATCH = max_concurrent * 4
            hash_done = 0
            for batch_start in range(0, num_videos, HASH_BATCH):
                # ── check pause / stop ──
                await _pipeline_check("hashing", "Computing perceptual hashes...")

                batch_end = min(batch_start + HASH_BATCH, num_videos)
                batch_videos = video_records[batch_start:batch_end]

                tasks = [_hash_one(v) for v in batch_videos]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Post-batch signal check
                await _pipeline_check("hashing", "Computing perceptual hashes...")

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        print(f"Error hashing {batch_videos[i].file_path}: {result}")
                        continue
                    if result and result.get("hashes"):
                        batch_videos[i].perceptual_hashes = json.dumps(result["hashes"])
                        batch_videos[i].hash_computed = True

                hash_done = batch_end
                scan.progress_percent = round(45 + (hash_done / num_videos) * 30, 1)
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
                        "scanned_files": hash_done,
                        "message": f"Hashed {hash_done}/{num_videos} files",
                    }, gpu_active, gpu_name)

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
            audio_sem = asyncio.Semaphore(max_concurrent)

            async def _fp_one(file_path: str):
                if scan_control.is_stopped(scan_id):
                    return file_path, []
                async with audio_sem:
                    if scan_control.is_stopped(scan_id):
                        return file_path, []
                    return file_path, await audio_fingerprint(file_path)

            # Only fingerprint candidate videos, in batches
            candidate_list = [v for v in video_records if v.file_path in _candidate_paths]
            FP_BATCH = max_concurrent * 4
            for batch_start in range(0, len(candidate_list), FP_BATCH):
                await _pipeline_check("comparing", "Computing audio fingerprints...")

                batch_end = min(batch_start + FP_BATCH, len(candidate_list))
                tasks = [_fp_one(v.file_path) for v in candidate_list[batch_start:batch_end]]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Post-batch signal check
                await _pipeline_check("comparing", "Computing audio fingerprints...")

                for r in results:
                    if isinstance(r, Exception):
                        print(f"Audio FP error: {r}")
                        continue
                    fpath, fp = r
                    if fp:
                        audio_fps[fpath] = fp

            video_data = []
            video_id_map = {}
            for v in video_records:
                hashes = json.loads(v.perceptual_hashes) if v.perceptual_hashes else []
                vd = {
                    "id": v.id,
                    "file_path": v.file_path,
                    "duration": v.duration,
                    "hashes": hashes,
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

            await manager.send_progress(scan_id, {
                "type": "error", "scan_id": scan_id,
                "status": "failed",
                "message": f"Scan failed: {str(e)}",
            })

        finally:
            scan_control.unregister(scan_id)
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


@router.delete("/scan/{scan_id}")
async def cancel_scan(scan_id: int, db: AsyncSession = Depends(get_db)):
    """Cancel a queued scan (remove before it starts)."""
    scan = await db.get(ScanJob, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    if scan.status != "queued":
        raise HTTPException(status_code=400, detail=f"Can only cancel queued scans (current: {scan.status})")

    await db.delete(scan)
    await db.commit()
    return {"scan_id": scan_id, "message": "Queued scan cancelled"}


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
