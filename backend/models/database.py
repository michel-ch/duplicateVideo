"""SQLAlchemy database models and setup."""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey,
    UniqueConstraint, Index, create_engine, event
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
        await conn.run_sync(_migrate_add_columns)


def _migrate_add_columns(conn) -> None:
    """SQLite-only forward-only column adds.

    `create_all` only creates missing TABLES, not missing columns on
    existing tables.  When the schema gains a new nullable column, this
    helper does a one-shot `ALTER TABLE ADD COLUMN` so users on a pre-
    existing DB don't have to delete it.  Run on every startup; the
    PRAGMA check makes it idempotent.
    """
    from sqlalchemy import text

    def _existing_cols(table: str) -> set:
        rows = conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
        return {r[1] for r in rows}

    required: dict = {
        "file_cache": {
            "head_tail_xxh3": "VARCHAR",
            "aggregate_hash": "VARCHAR",
        },
    }
    for table, cols in required.items():
        try:
            existing = _existing_cols(table)
        except Exception:
            continue  # table doesn't exist yet — create_all will handle it
        for col_name, col_type in cols.items():
            if col_name not in existing:
                conn.exec_driver_sql(
                    f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"
                )


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

    # Cache linkage (Phase 1 incremental scans)
    file_cache_id = Column(Integer, ForeignKey("file_cache.id"), nullable=True)
    cache_hit = Column(Boolean, default=False)

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


class FileCache(Base):
    """Persistent per-file cache of pipeline outputs, keyed by content identity.

    A cache row's identity is (file_path, file_size, mtime_ns).  When a future
    scan finds the same tuple, every stage whose output is already populated
    is skipped: stages 2 (metadata + thumbnail), 3 (perceptual hashes), and
    4b (audio fingerprint) all read straight from this row.

    Cache rows outlive the scans that produced them.  They are pruned by the
    end-of-scan sweep when their `file_path` falls under the just-scanned root
    but they were not seen during the scan.
    """

    __tablename__ = "file_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    mtime_ns = Column(Integer, nullable=False)

    # Optional full-file hash (Phase 3 SHA-256 fast path; nullable until computed)
    sha256_full = Column(String, nullable=True)

    # Cached metadata (stage 2 output)
    duration = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    bitrate = Column(Integer, nullable=True)
    video_codec = Column(String, nullable=True)
    audio_codec = Column(String, nullable=True)
    fps = Column(Float, nullable=True)
    audio_channels = Column(Integer, nullable=True)
    audio_sample_rate = Column(Integer, nullable=True)
    sar_num = Column(Integer, default=1)
    sar_den = Column(Integer, default=1)
    rotation = Column(Integer, default=0)

    # Cached pipeline outputs (stages 3 and 4b)
    perceptual_hashes = Column(Text, nullable=True)  # JSON array of hex hashes
    audio_fp = Column(Text, nullable=True)           # JSON array of 64 floats
    thumbnail_path = Column(String, nullable=True)

    # Quick-rejection cascade
    #   head_tail_xxh3 — xxh3_64 hex of first 64 KiB + last 64 KiB.  Byte-
    #     identical files always share this; used as an O(1) exact-duplicate
    #     fast path before any decode happens.
    #   aggregate_hash — 256-bit hex from per-bit majority vote across the
    #     12 perceptual hashes.  Used by the FAISS binary index as the
    #     screening key; the 12-hash set is then the verifier.
    head_tail_xxh3 = Column(String, nullable=True)
    aggregate_hash = Column(String, nullable=True)

    # Bookkeeping
    first_seen_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_seen_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    cache_version = Column(Integer, nullable=False, default=1)

    __table_args__ = (
        UniqueConstraint("file_path", "file_size", "mtime_ns", name="uq_cache_identity"),
        Index("idx_file_cache_path", "file_path"),
        Index("idx_file_cache_lastseen", "last_seen_at"),
    )


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
