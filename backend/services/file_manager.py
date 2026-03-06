"""File deletion, trash, and undo operations."""

import os
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Tuple

from config import settings


def get_trash_dir(root_path: str) -> str:
    """Get or create the trash directory."""
    trash_dir = os.path.join(root_path, settings.TRASH_FOLDER_NAME)
    os.makedirs(trash_dir, exist_ok=True)
    return trash_dir


def is_protected_path(file_path: str) -> bool:
    """Check if a file is in a protected directory."""
    abs_path = str(Path(file_path).resolve())
    for protected in settings.PROTECTED_PATHS:
        protected_abs = str(Path(protected).resolve())
        if abs_path.startswith(protected_abs):
            return True
    return False


def move_to_trash(file_path: str, scan_root: str) -> Tuple[bool, str, Optional[str]]:
    """
    Move a file to the trash directory.
    Returns: (success, message, trash_path)
    """
    if not os.path.exists(file_path):
        return False, "File not found", None

    if is_protected_path(file_path):
        return False, "File is in a protected directory", None

    trash_dir = get_trash_dir(scan_root)

    # Preserve relative structure in trash
    rel_path = os.path.relpath(file_path, scan_root)
    trash_path = os.path.join(trash_dir, rel_path)

    # Handle name conflicts
    if os.path.exists(trash_path):
        base, ext = os.path.splitext(trash_path)
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        trash_path = f"{base}_{counter}{ext}"

    os.makedirs(os.path.dirname(trash_path), exist_ok=True)

    try:
        shutil.move(file_path, trash_path)
        return True, "Moved to trash", trash_path
    except Exception as e:
        return False, f"Failed to move: {str(e)}", None


def delete_permanently(file_path: str) -> Tuple[bool, str]:
    """Permanently delete a file."""
    if not os.path.exists(file_path):
        return False, "File not found"

    if is_protected_path(file_path):
        return False, "File is in a protected directory"

    try:
        os.remove(file_path)
        return True, "Permanently deleted"
    except Exception as e:
        return False, f"Failed to delete: {str(e)}"


def undo_deletion(original_path: str, trash_path: str) -> Tuple[bool, str]:
    """Restore a file from trash to its original location."""
    if not trash_path or not os.path.exists(trash_path):
        return False, "Trash file not found"

    os.makedirs(os.path.dirname(original_path), exist_ok=True)

    try:
        shutil.move(trash_path, original_path)
        return True, "File restored"
    except Exception as e:
        return False, f"Failed to restore: {str(e)}"


def delete_files_batch(
    files: List[dict],
    move_to_trash_mode: bool = True,
    scan_root: str = ""
) -> List[dict]:
    """
    Delete multiple files, either to trash or permanently.
    Returns list of results with success/failure info.
    """
    results = []

    for file_info in files:
        file_path = file_info.get("file_path", "")
        file_size = file_info.get("file_size", 0)

        if move_to_trash_mode:
            success, message, trash_path = move_to_trash(file_path, scan_root)
            results.append({
                "file_path": file_path,
                "file_size": file_size,
                "success": success,
                "message": message,
                "trash_path": trash_path,
                "mode": "trash"
            })
        else:
            success, message = delete_permanently(file_path)
            results.append({
                "file_path": file_path,
                "file_size": file_size,
                "success": success,
                "message": message,
                "trash_path": None,
                "mode": "permanent"
            })

    return results
