"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ---- Scan ----

class ScanOptions(BaseModel):
    similarity_threshold: float = 70.0
    duration_tolerance: float = 2.0
    key_frames_count: int = 8
    hash_threshold: int = 10
    max_concurrent: int = 4


class ScanRequest(BaseModel):
    path: str
    options: Optional[ScanOptions] = None


class ScanStatusResponse(BaseModel):
    id: int
    root_path: str
    status: str
    total_files: int
    scanned_files: int
    current_file: Optional[str] = None
    current_stage: Optional[str] = None
    progress_percent: float
    duplicate_groups_found: int
    recoverable_space: float
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


# ---- Video ----

class VideoFileResponse(BaseModel):
    id: int
    file_path: str
    file_name: str
    file_size: float
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    bitrate: Optional[int] = None
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    fps: Optional[float] = None
    audio_channels: Optional[int] = None
    audio_sample_rate: Optional[int] = None
    quality_score: Optional[float] = None
    is_best_quality: bool = False
    thumbnail_path: Optional[str] = None
    is_deleted: bool = False

    class Config:
        from_attributes = True


# ---- Duplicate Group ----

class DuplicateGroupSummary(BaseModel):
    id: int
    similarity_score: float
    total_wasted_space: float
    file_count: int
    status: str
    best_file_id: Optional[int] = None
    videos: List[VideoFileResponse] = []

    class Config:
        from_attributes = True


class DuplicateGroupDetail(DuplicateGroupSummary):
    pass


class ResolveRequest(BaseModel):
    keep_file_ids: List[int]
    delete_file_ids: List[int]
    move_to_trash: bool = True


# ---- Delete ----

class DeleteRequest(BaseModel):
    file_ids: List[int]
    move_to_trash: bool = True


class AutoCleanRequest(BaseModel):
    move_to_trash: bool = True
    confirm: bool = False


# ---- Stats ----

class StatsResponse(BaseModel):
    total_videos: int = 0
    total_scans: int = 0
    duplicate_groups: int = 0
    total_duplicates: int = 0
    recoverable_space: float = 0.0
    space_recovered: float = 0.0
    last_scan_date: Optional[datetime] = None


# ---- Settings ----

class SettingsResponse(BaseModel):
    similarity_threshold: float
    duration_tolerance: float
    key_frames_count: int
    hash_threshold: int
    max_concurrent: int
    resolution_weight: float
    bitrate_weight: float
    codec_weight: float
    file_size_weight: float
    fps_weight: float
    default_trash_mode: bool
    video_extensions: List[str]
    protected_paths: List[str]


class SettingsUpdate(BaseModel):
    similarity_threshold: Optional[float] = None
    duration_tolerance: Optional[float] = None
    key_frames_count: Optional[int] = None
    hash_threshold: Optional[int] = None
    max_concurrent: Optional[int] = None
    resolution_weight: Optional[float] = None
    bitrate_weight: Optional[float] = None
    codec_weight: Optional[float] = None
    file_size_weight: Optional[float] = None
    fps_weight: Optional[float] = None
    default_trash_mode: Optional[bool] = None
    video_extensions: Optional[List[str]] = None
    protected_paths: Optional[List[str]] = None


# ---- Deletion Log ----

class DeletionLogResponse(BaseModel):
    id: int
    original_path: str
    trash_path: Optional[str] = None
    file_size: float
    deletion_mode: str
    deleted_at: Optional[datetime] = None
    is_undone: bool = False

    class Config:
        from_attributes = True


# ---- WebSocket ----

class ScanProgressMessage(BaseModel):
    type: str = "progress"
    scan_id: int
    status: str
    total_files: int
    scanned_files: int
    current_file: Optional[str] = None
    current_stage: Optional[str] = None
    progress_percent: float
    duplicate_groups_found: int = 0
    recoverable_space: float = 0.0
    message: Optional[str] = None
