"""Action endpoints: delete, auto-clean, stats, settings, history."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from datetime import datetime, timezone

from models.database import (
    get_db, VideoFile, DuplicateGroup, ScanJob, DeletionLog
)
from models.schemas import (
    DeleteRequest, AutoCleanRequest,
    StatsResponse, SettingsResponse, SettingsUpdate,
    DeletionLogResponse
)
from services.file_manager import (
    move_to_trash, delete_permanently, undo_deletion, delete_files_batch
)
from config import settings

router = APIRouter()


@router.post("/delete")
async def delete_files(request: DeleteRequest, db: AsyncSession = Depends(get_db)):
    """Delete selected files."""
    results = []
    errors = []

    for file_id in request.file_ids:
        video = await db.get(VideoFile, file_id)
        if not video:
            errors.append(f"File ID {file_id} not found")
            continue

        # Get scan root
        scan = await db.get(ScanJob, video.scan_job_id)
        scan_root = scan.root_path if scan else ""

        if request.move_to_trash:
            success, message, trash_path = move_to_trash(video.file_path, scan_root)
        else:
            success, message = delete_permanently(video.file_path)
            trash_path = None

        if success:
            video.is_deleted = True
            video.deleted_at = datetime.now(timezone.utc)
            video.trash_path = trash_path

            log = DeletionLog(
                original_path=video.file_path,
                trash_path=trash_path,
                file_size=video.file_size,
                deletion_mode="trash" if request.move_to_trash else "permanent",
                scan_job_id=video.scan_job_id,
            )
            db.add(log)
            results.append({"file_id": file_id, "path": video.file_path, "file_size": video.file_size, "success": True})
        else:
            errors.append(f"{video.file_path}: {message}")

    await db.commit()

    # Compute freed space from the results list
    total_freed = sum(
        r.get("file_size", 0) for r in results if r.get("success")
    )

    return {
        "deleted": results,
        "errors": errors,
        "space_freed": total_freed,
    }


@router.post("/auto-clean")
async def auto_clean(request: AutoCleanRequest, db: AsyncSession = Depends(get_db)):
    """Auto-delete all lower-quality duplicates."""
    if not request.confirm:
        # Return preview of what would be deleted
        result = await db.execute(
            select(DuplicateGroup)
            .options(selectinload(DuplicateGroup.videos))
            .where(DuplicateGroup.status != "resolved")
        )
        groups = result.scalars().unique().all()

        files_to_delete = []
        total_space = 0

        for group in groups:
            for video in group.videos:
                if not video.is_best_quality and not video.is_deleted:
                    files_to_delete.append({
                        "id": video.id,
                        "path": video.file_path,
                        "size": video.file_size,
                        "quality_score": video.quality_score,
                    })
                    total_space += video.file_size

        return {
            "preview": True,
            "files_to_delete": files_to_delete,
            "total_files": len(files_to_delete),
            "total_space": total_space,
            "message": f"Will delete {len(files_to_delete)} files, freeing {total_space / (1024*1024*1024):.2f} GB"
        }

    # Execute auto-clean
    result = await db.execute(
        select(DuplicateGroup)
        .options(selectinload(DuplicateGroup.videos))
        .where(DuplicateGroup.status != "resolved")
    )
    groups = result.scalars().unique().all()

    deleted = []
    errors = []

    for group in groups:
        scan = await db.get(ScanJob, group.scan_job_id)
        scan_root = scan.root_path if scan else ""

        for video in group.videos:
            if not video.is_best_quality and not video.is_deleted:
                if request.move_to_trash:
                    success, message, trash_path = move_to_trash(video.file_path, scan_root)
                else:
                    success, message = delete_permanently(video.file_path)
                    trash_path = None

                if success:
                    video.is_deleted = True
                    video.deleted_at = datetime.now(timezone.utc)
                    video.trash_path = trash_path
                    deleted.append(video.file_path)

                    log = DeletionLog(
                        original_path=video.file_path,
                        trash_path=trash_path,
                        file_size=video.file_size,
                        deletion_mode="trash" if request.move_to_trash else "permanent",
                        scan_job_id=group.scan_job_id,
                        duplicate_group_id=group.id,
                    )
                    db.add(log)
                else:
                    errors.append(f"{video.file_path}: {message}")

        group.status = "resolved"

    await db.commit()

    return {
        "preview": False,
        "deleted_count": len(deleted),
        "errors": errors,
        "deleted_paths": deleted,
    }


@router.get("/stats")
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Get dashboard statistics."""
    # Total videos
    total_videos = await db.execute(
        select(func.count(VideoFile.id)).where(VideoFile.is_deleted == False)
    )

    # Total scans
    total_scans = await db.execute(select(func.count(ScanJob.id)))

    # Duplicate groups
    dup_groups = await db.execute(select(func.count(DuplicateGroup.id)))

    # Total duplicates (non-best files in groups)
    total_dups = await db.execute(
        select(func.count(VideoFile.id)).where(
            VideoFile.duplicate_group_id.isnot(None),
            VideoFile.is_best_quality == False,
            VideoFile.is_deleted == False
        )
    )

    # Recoverable space
    recoverable = await db.execute(
        select(func.sum(VideoFile.file_size)).where(
            VideoFile.duplicate_group_id.isnot(None),
            VideoFile.is_best_quality == False,
            VideoFile.is_deleted == False
        )
    )

    # Space recovered (deleted files)
    recovered = await db.execute(
        select(func.sum(DeletionLog.file_size)).where(
            DeletionLog.is_undone == False
        )
    )

    # Last scan
    last_scan = await db.execute(
        select(ScanJob.completed_at)
        .where(ScanJob.status == "completed")
        .order_by(ScanJob.completed_at.desc())
        .limit(1)
    )

    return StatsResponse(
        total_videos=total_videos.scalar() or 0,
        total_scans=total_scans.scalar() or 0,
        duplicate_groups=dup_groups.scalar() or 0,
        total_duplicates=total_dups.scalar() or 0,
        recoverable_space=recoverable.scalar() or 0.0,
        space_recovered=recovered.scalar() or 0.0,
        last_scan_date=last_scan.scalar(),
    )


@router.get("/settings")
async def get_settings():
    """Get current application settings."""
    return SettingsResponse(
        similarity_threshold=settings.SIMILARITY_THRESHOLD_PERCENT,
        duration_tolerance=settings.DURATION_TOLERANCE_SECONDS,
        key_frames_count=settings.KEY_FRAMES_COUNT,
        hash_threshold=settings.HASH_SIMILARITY_THRESHOLD,
        max_concurrent=settings.MAX_CONCURRENT_FFMPEG,
        resolution_weight=settings.RESOLUTION_WEIGHT,
        bitrate_weight=settings.BITRATE_WEIGHT,
        codec_weight=settings.CODEC_WEIGHT,
        file_size_weight=settings.FILE_SIZE_WEIGHT,
        fps_weight=settings.FPS_WEIGHT,
        default_trash_mode=settings.DEFAULT_TRASH_MODE,
        video_extensions=settings.VIDEO_EXTENSIONS,
        protected_paths=settings.PROTECTED_PATHS,
    )


@router.put("/settings")
async def update_settings(update: SettingsUpdate):
    """Update application settings."""
    updated = {}
    for field, value in update.model_dump(exclude_none=True).items():
        if hasattr(settings, field.upper()):
            setattr(settings, field.upper(), value)
            updated[field] = value
        else:
            # Map field names
            field_map = {
                "similarity_threshold": "SIMILARITY_THRESHOLD_PERCENT",
                "duration_tolerance": "DURATION_TOLERANCE_SECONDS",
                "key_frames_count": "KEY_FRAMES_COUNT",
                "hash_threshold": "HASH_SIMILARITY_THRESHOLD",
                "max_concurrent": "MAX_CONCURRENT_FFMPEG",
                "resolution_weight": "RESOLUTION_WEIGHT",
                "bitrate_weight": "BITRATE_WEIGHT",
                "codec_weight": "CODEC_WEIGHT",
                "file_size_weight": "FILE_SIZE_WEIGHT",
                "fps_weight": "FPS_WEIGHT",
                "default_trash_mode": "DEFAULT_TRASH_MODE",
                "video_extensions": "VIDEO_EXTENSIONS",
                "protected_paths": "PROTECTED_PATHS",
            }
            attr = field_map.get(field)
            if attr:
                setattr(settings, attr, value)
                updated[field] = value

    return {"updated": updated, "message": "Settings updated"}


@router.get("/history")
async def get_history(
    page: int = 1,
    per_page: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """Get deletion history."""
    total_result = await db.execute(select(func.count(DeletionLog.id)))
    total = total_result.scalar() or 0

    offset = (page - 1) * per_page
    result = await db.execute(
        select(DeletionLog)
        .order_by(DeletionLog.deleted_at.desc())
        .offset(offset).limit(per_page)
    )
    logs = result.scalars().all()

    return {
        "items": [DeletionLogResponse.model_validate(l) for l in logs],
        "total": total,
        "page": page,
        "per_page": per_page,
    }


@router.delete("/history")
async def clear_history(db: AsyncSession = Depends(get_db)):
    """Delete all history records."""
    result = await db.execute(select(DeletionLog))
    logs = result.scalars().all()
    for log in logs:
        await db.delete(log)
    await db.commit()
    return {"success": True, "deleted_count": len(logs)}


@router.post("/history/{log_id}/undo")
async def undo_delete(log_id: int, db: AsyncSession = Depends(get_db)):
    """Undo a deletion (restore from trash)."""
    log = await db.get(DeletionLog, log_id)
    if not log:
        raise HTTPException(status_code=404, detail="Deletion log not found")

    if log.deletion_mode == "permanent":
        raise HTTPException(status_code=400, detail="Cannot undo permanent deletion")

    if log.is_undone:
        raise HTTPException(status_code=400, detail="Already undone")

    success, message = undo_deletion(log.original_path, log.trash_path)

    if not success:
        raise HTTPException(status_code=500, detail=message)

    log.is_undone = True
    log.undone_at = datetime.now(timezone.utc)

    # Also mark the video as not deleted
    result = await db.execute(
        select(VideoFile).where(VideoFile.file_path == log.original_path)
    )
    video = result.scalar_one_or_none()
    if video:
        video.is_deleted = False
        video.deleted_at = None
        video.trash_path = None

    await db.commit()

    return {"success": True, "message": "File restored successfully"}
