"""In-memory per-scan error log.

Per-file failures during a scan (frame extraction timeouts, ffprobe
errors, audio decode failures, etc.) used to go to stdout only — the
user couldn't see them without watching the terminal.  This module
collects them into a small ring buffer per scan_id so the WebSocket
endpoint can fan them out to the UI in real-time and so a late
subscriber can backfill the most recent N when it connects.

In-memory only; entries die with the server.  Fatal scan-level
failures still get persisted on `ScanJob.error_message` — this is
only for the per-file noise that the UI couldn't previously surface.
"""

from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional


# Cap per scan to bound memory on a pathological scan that fails every file.
_MAX_PER_SCAN = 200


class _ScanErrors:
    __slots__ = ("entries",)

    def __init__(self) -> None:
        self.entries: Deque[dict] = deque(maxlen=_MAX_PER_SCAN)


_registry: Dict[int, _ScanErrors] = {}


def register(scan_id: int) -> None:
    """Initialise an error buffer for a scan."""
    _registry[scan_id] = _ScanErrors()


def unregister(scan_id: int) -> None:
    """Drop the buffer once the scan ends (any terminal state)."""
    _registry.pop(scan_id, None)


def log(
    scan_id: int,
    stage: str,
    message: str,
    file_path: Optional[str] = None,
    level: str = "error",
) -> dict:
    """Record one entry.  Returns the entry as a dict suitable for WS
    broadcast (so callers can both store and send in one step).
    """
    entry = {
        "scan_id": scan_id,
        "stage": stage,
        "level": level,
        "message": message,
        "file_path": file_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    buf = _registry.get(scan_id)
    if buf is not None:
        buf.entries.append(entry)
    return entry


def get_recent(scan_id: int, limit: Optional[int] = None) -> List[dict]:
    """Return the most recent error entries for `scan_id`, oldest first.

    Returns an empty list if the scan has no registered buffer (already
    unregistered, or never had errors).
    """
    buf = _registry.get(scan_id)
    if buf is None:
        return []
    if limit is None or limit >= len(buf.entries):
        return list(buf.entries)
    return list(buf.entries)[-limit:]
