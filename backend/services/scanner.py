"""Directory walking and video file discovery."""

import hashlib
import os
from pathlib import Path
from typing import List, AsyncGenerator, Optional
from datetime import datetime, timezone

from config import settings


# Bytes to read from each end of the file for the head+tail content hash.
# 64 KiB × 2 = 128 KiB total — completes in <1 ms even on spinning rust,
# while being large enough that an mp4's moov atom (at the end for streaming
# files, at the start for fast-start files) is almost always covered.
_HEAD_TAIL_BYTES = 64 * 1024


def compute_head_tail_hash(file_path: str, file_size: Optional[int] = None) -> Optional[str]:
    """Hash the first 64 KiB + last 64 KiB of a file (blake2b, 16 hex chars).

    Two files that share this hash AND a file size are almost certainly
    byte-identical — they can be declared duplicates without any decode or
    pHash work.  Two files that share content but differ in container
    (mp4 vs mkv re-mux) typically DO NOT share this hash, so a mismatch
    is not evidence of difference.

    blake2b (stdlib) is ~1 GiB/s — plenty fast for 128 KiB and avoids the
    xxhash dependency.
    """
    try:
        if file_size is None:
            file_size = os.path.getsize(file_path)
        h = hashlib.blake2b(digest_size=8)
        with open(file_path, "rb") as f:
            if file_size <= _HEAD_TAIL_BYTES * 2:
                h.update(f.read())
            else:
                h.update(f.read(_HEAD_TAIL_BYTES))
                f.seek(-_HEAD_TAIL_BYTES, os.SEEK_END)
                h.update(f.read(_HEAD_TAIL_BYTES))
        return h.hexdigest()
    except OSError:
        return None


def get_file_info(file_path: str) -> dict:
    """Get basic file information."""
    p = Path(file_path)
    stat = p.stat()
    return {
        "file_path": str(p.resolve()),
        "file_name": p.name,
        "file_size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
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
