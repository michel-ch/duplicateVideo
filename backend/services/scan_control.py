"""In-memory scan control signals for pause / stop.

Every running scan is identified by its integer scan_id.  The pipeline
loop checks these signals between iterations so it can pause (block) or
abort (raise) cleanly.

This module is intentionally in-memory — if the server restarts, all
running scans are gone anyway, so there is nothing to persist.
"""

import asyncio
from typing import Dict


class _ScanSignals:
    """Per-scan asyncio events for pause and stop."""

    def __init__(self):
        # When *clear*, the pipeline should block (i.e. it is paused).
        # Initialised to *set* (= running).
        self.resume_event = asyncio.Event()
        self.resume_event.set()

        # When *set*, the pipeline should abort.
        self.stop_event = asyncio.Event()


# scan_id → signals
_registry: Dict[int, _ScanSignals] = {}


def register(scan_id: int) -> None:
    """Register a new scan.  Must be called when a scan starts."""
    _registry[scan_id] = _ScanSignals()


def unregister(scan_id: int) -> None:
    """Remove signals once the scan finishes (any terminal state)."""
    _registry.pop(scan_id, None)


def pause(scan_id: int) -> bool:
    """Pause a running scan.  Returns False if scan_id is unknown."""
    sig = _registry.get(scan_id)
    if not sig:
        return False
    sig.resume_event.clear()          # pipeline will block on wait()
    return True


def resume(scan_id: int) -> bool:
    """Resume a paused scan.  Returns False if scan_id is unknown."""
    sig = _registry.get(scan_id)
    if not sig:
        return False
    sig.resume_event.set()            # unblock the pipeline
    return True


def stop(scan_id: int) -> bool:
    """Request a running (or paused) scan to stop.  Returns False if unknown."""
    sig = _registry.get(scan_id)
    if not sig:
        return False
    sig.stop_event.set()              # pipeline will see this on next check
    sig.resume_event.set()            # unblock if currently paused
    return True


def is_paused(scan_id: int) -> bool:
    sig = _registry.get(scan_id)
    return sig is not None and not sig.resume_event.is_set()


def is_stopped(scan_id: int) -> bool:
    sig = _registry.get(scan_id)
    return sig is not None and sig.stop_event.is_set()


async def check_signals(scan_id: int) -> str:
    """
    Call this inside the pipeline loop.

    Returns:
      "ok"      – continue processing
      "stopped" – the scan was requested to stop; caller should break

    If the scan is paused, this coroutine will *block* until it is
    resumed (or stopped).
    """
    sig = _registry.get(scan_id)
    if sig is None:
        return "ok"

    # If stopped, return immediately
    if sig.stop_event.is_set():
        return "stopped"

    # If paused, block here until resumed (or stopped)
    if not sig.resume_event.is_set():
        # Wait for either resume or stop
        stop_task = asyncio.create_task(sig.stop_event.wait())
        resume_task = asyncio.create_task(sig.resume_event.wait())
        done, pending = await asyncio.wait(
            {stop_task, resume_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()

        if sig.stop_event.is_set():
            return "stopped"

    return "ok"
