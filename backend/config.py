"""Application configuration and settings."""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import os


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./duplicate_detector.db"

    # Scanning
    VIDEO_EXTENSIONS: List[str] = [
        ".mp4", ".mkv", ".avi", ".mov", ".wmv",
        ".flv", ".webm", ".m4v", ".ts", ".3gp"
    ]
    MAX_CONCURRENT_FFMPEG: int = 8
    KEY_FRAMES_COUNT: int = 12

    # GPU acceleration
    GPU_ENABLED: bool = True       # Set False to force CPU-only
    GPU_MAX_CONCURRENT: int = 12   # Higher concurrency when GPU-decoding

    # Duplicate detection
    DURATION_TOLERANCE_SECONDS: float = 3.0
    HASH_SIMILARITY_THRESHOLD: int = 14  # Hamming distance threshold (higher = more lenient)
    SIMILARITY_THRESHOLD_PERCENT: float = 85.0

    # Quality scoring weights
    RESOLUTION_WEIGHT: float = 0.40
    BITRATE_WEIGHT: float = 0.25
    CODEC_WEIGHT: float = 0.15
    FILE_SIZE_WEIGHT: float = 0.10
    FPS_WEIGHT: float = 0.10

    # Codec scores
    CODEC_SCORES: dict = {
        "hevc": 1.0, "h265": 1.0,
        "h264": 0.8, "avc": 0.8,
        "vp9": 0.85,
        "av1": 1.0,
    }

    # Deletion
    DEFAULT_TRASH_MODE: bool = True
    TRASH_FOLDER_NAME: str = ".duplicate_trash"

    # Protected paths
    PROTECTED_PATHS: List[str] = []

    # FFprobe path
    FFPROBE_PATH: str = "ffprobe"

    # Thumbnails directory (resolved at runtime relative to backend cwd)
    THUMBNAILS_DIR: str = "thumbnails"

    class Config:
        env_file = ".env"


settings = Settings()
