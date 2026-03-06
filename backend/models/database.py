"""SQLAlchemy database models and setup."""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey,
    UniqueConstraint, create_engine, event
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime, timezone

from config import settings

Base = declarative_base()

engine = create_async_engine(settings.DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with async_session() as session:
        yield session


class ScanJob(Base):
    __tablename__ = "scan_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    root_path = Column(String, nullable=False)
    status = Column(String, default="pending")  # queued, pending, scanning, metadata, hashing, comparing, completed, failed, paused, stopped
    total_files = Column(Integer, default=0)
    scanned_files = Column(Integer, default=0)
    current_file = Column(String, nullable=True)
    current_stage = Column(String, nullable=True)
    progress_percent = Column(Float, default=0.0)
    duplicate_groups_found = Column(Integer, default=0)
    recoverable_space = Column(Float, default=0.0)  # in bytes
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    options = Column(Text, nullable=True)  # JSON string of scan options

    videos = relationship("VideoFile", back_populates="scan_job", cascade="all, delete-orphan")


class VideoFile(Base):
    __tablename__ = "video_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_job_id = Column(Integer, ForeignKey("scan_jobs.id"), nullable=False)
    file_path = Column(String, nullable=False)
    file_name = Column(String, nullable=False)

    __table_args__ = (
        UniqueConstraint("scan_job_id", "file_path", name="uq_scan_file_path"),
    )

    file_size = Column(Float, default=0)  # bytes
    created_at = Column(DateTime, nullable=True)
    modified_at = Column(DateTime, nullable=True)

    # Video metadata
    duration = Column(Float, nullable=True)  # seconds
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    bitrate = Column(Integer, nullable=True)  # bps
    video_codec = Column(String, nullable=True)
    audio_codec = Column(String, nullable=True)
    fps = Column(Float, nullable=True)
    audio_channels = Column(Integer, nullable=True)
    audio_sample_rate = Column(Integer, nullable=True)

    # Hashing
    perceptual_hashes = Column(Text, nullable=True)  # JSON array of hashes
    hash_computed = Column(Boolean, default=False)

    # Quality
    quality_score = Column(Float, nullable=True)

    # Duplicate group
    duplicate_group_id = Column(Integer, ForeignKey("duplicate_groups.id"), nullable=True)
    is_best_quality = Column(Boolean, default=False)

    # Thumbnail
    thumbnail_path = Column(String, nullable=True)

    # Status
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime, nullable=True)
    trash_path = Column(String, nullable=True)

    scan_job = relationship("ScanJob", back_populates="videos")
    duplicate_group = relationship("DuplicateGroup", back_populates="videos")


class DuplicateGroup(Base):
    __tablename__ = "duplicate_groups"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_job_id = Column(Integer, ForeignKey("scan_jobs.id"), nullable=False)
    similarity_score = Column(Float, default=0.0)
    total_wasted_space = Column(Float, default=0.0)  # bytes
    file_count = Column(Integer, default=0)
    status = Column(String, default="pending")  # pending, in_queue, resolved
    best_file_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    videos = relationship("VideoFile", back_populates="duplicate_group")


class DeletionLog(Base):
    __tablename__ = "deletion_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    original_path = Column(String, nullable=False)
    trash_path = Column(String, nullable=True)
    file_size = Column(Float, default=0)
    deletion_mode = Column(String, default="trash")  # trash, permanent
    deleted_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_undone = Column(Boolean, default=False)
    undone_at = Column(DateTime, nullable=True)
    scan_job_id = Column(Integer, nullable=True)
    duplicate_group_id = Column(Integer, nullable=True)
