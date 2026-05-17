"""Duplicate group API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from typing import Optional

from models.database import get_db, DuplicateGroup, VideoFile, ScanJob, DeletionLog
from models.schemas import (
    DuplicateGroupSummary, DuplicateGroupDetail,
    ResolveRequest, VideoFileResponse
)
from services.file_manager import move_to_trash, delete_permanently
from datetime import datetime, timezone

router = APIRouter()


@router.get("/duplicates")
async def list_duplicates(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    sort_by: str = Query("wasted_space", regex="^(wasted_space|similarity|file_count|date)$"),
    min_similarity: Optional[float] = None,
    status: Optional[str] = None,
    scan_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
):
    """List duplicate groups with pagination and filtering."""
    query = select(DuplicateGroup).options(selectinload(DuplicateGroup.videos))

    if scan_id:
        query = query.where(DuplicateGroup.scan_job_id == scan_id)
    if min_similarity:
        query = query.where(DuplicateGroup.similarity_score >= min_similarity)
    if status:
        query = query.where(DuplicateGroup.status == status)

    # Sorting
    if sort_by == "wasted_space":
        query = query.order_by(DuplicateGroup.total_wasted_space.desc())
    elif sort_by == "similarity":
        query = query.order_by(DuplicateGroup.similarity_score.desc())
    elif sort_by == "file_count":
        query = query.order_by(DuplicateGroup.file_count.desc())
    elif sort_by == "date":
        query = query.order_by(DuplicateGroup.created_at.desc())

    # Count total
    count_query = select(func.count(DuplicateGroup.id))
    if scan_id:
        count_query = count_query.where(DuplicateGroup.scan_job_id == scan_id)
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Paginate
    offset = (page - 1) * per_page
    query = query.offset(offset).limit(per_page)
    result = await db.execute(query)
    groups = result.scalars().unique().all()

    items = []
    for g in groups:
        videos = [VideoFileResponse.model_validate(v) for v in g.videos if not v.is_deleted]
        items.append({
            "id": g.id,
            "similarity_score": g.similarity_score,
            "total_wasted_space": g.total_wasted_space,
            "file_count": g.file_count,
            "status": g.status,
            "best_file_id": g.best_file_id,
            "videos": videos,
        })

    return {
        "items": items,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page if per_page > 0 else 0,
    }


@router.get("/duplicates/{group_id}")
async def get_duplicate_group(group_id: int, db: AsyncSession = Depends(get_db)):
    """Get details of a specific duplicate group."""
    result = await db.execute(
        select(DuplicateGroup)
        .options(selectinload(DuplicateGroup.videos))
        .where(DuplicateGroup.id == group_id)
    )
    group = result.scalar_one_or_none()

    if not group:
        raise HTTPException(status_code=404, detail="Duplicate group not found")

    videos = [VideoFileResponse.model_validate(v) for v in group.videos]

    return {
        "id": group.id,
        "similarity_score": group.similarity_score,
        "total_wasted_space": group.total_wasted_space,
        "file_count": group.file_count,
        "status": group.status,
        "best_file_id": group.best_file_id,
        "videos": videos,
    }


@router.post("/duplicates/{group_id}/resolve")
async def resolve_duplicate_group(
    group_id: int,
    request: ResolveRequest,
    db: AsyncSession = Depends(get_db)
):
    """Resolve a duplicate group by keeping/deleting specified files."""
    result = await db.execute(
        select(DuplicateGroup)
        .options(selectinload(DuplicateGroup.videos))
        .where(DuplicateGroup.id == group_id)
    )
    group = result.scalar_one_or_none()

    if not group:
        raise HTTPException(status_code=404, detail="Duplicate group not found")

    # Get scan root for trash directory
    scan = await db.get(ScanJob, group.scan_job_id)
    scan_root = scan.root_path if scan else ""

    deleted_files = []
    errors = []

    for file_id in request.delete_file_ids:
        video = await db.get(VideoFile, file_id)
        if not video:
            errors.append(f"File ID {file_id} not found")
            continue

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
                scan_job_id=group.scan_job_id,
                duplicate_group_id=group_id,
            )
            db.add(log)
            deleted_files.append(video.file_path)
        else:
            errors.append(f"{video.file_path}: {message}")

    group.status = "resolved"
    await db.commit()

    return {
        "success": len(errors) == 0,
        "deleted": deleted_files,
        "errors": errors,
        "group_status": "resolved"
    }
