"""Directory walking and video file discovery."""

import os
from pathlib import Path
from typing import List, AsyncGenerator
from datetime import datetime, timezone

from config import settings


def get_file_info(file_path: str) -> dict:
    """Get basic file information."""
    p = Path(file_path)
    stat = p.stat()
    return {
        "file_path": str(p.resolve()),
        "file_name": p.name,
        "file_size": stat.st_size,
        "created_at": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
    }


def discover_videos(root_path: str) -> List[str]:
    """
    Recursively walk a directory tree and discover all video files
    by their extension.
    """
    video_files = []
    root = Path(root_path)

    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root_path}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root_path}")

    extensions = set(ext.lower() for ext in settings.VIDEO_EXTENSIONS)

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden directories and trash folder
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith('.') and d != settings.TRASH_FOLDER_NAME
        ]

        for filename in filenames:
            ext = Path(filename).suffix.lower()
            if ext in extensions:
                full_path = os.path.join(dirpath, filename)
                video_files.append(str(Path(full_path).resolve()))

    return video_files


def count_videos(root_path: str) -> int:
    """Quick count of video files without storing paths."""
    count = 0
    root = Path(root_path)
    extensions = set(ext.lower() for ext in settings.VIDEO_EXTENSIONS)

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith('.') and d != settings.TRASH_FOLDER_NAME
        ]
        for filename in filenames:
            if Path(filename).suffix.lower() in extensions:
                count += 1

    return count
