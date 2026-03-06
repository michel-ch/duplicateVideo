"""Audio fingerprinting for duplicate video detection.

Uses audio energy profiles to detect duplicate content regardless of
video encoding, aspect ratio, or orientation differences.

How it works:
  1. Extract audio as raw PCM (8 kHz mono) via FFmpeg
  2. Divide into N equal segments
  3. Compute RMS energy for each segment → "energy profile"
  4. Compare profiles via normalised cross-correlation

Two re-encodes of the same video will have nearly identical audio
energy profiles (correlation > 85 %) even if the video track looks
completely different (portrait vs landscape, different bitrate, etc.).
"""

import subprocess
import asyncio
import os
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from config import settings

_CREATION_FLAGS = (
    subprocess.CREATE_NO_WINDOW
    if hasattr(subprocess, "CREATE_NO_WINDOW")
    else 0
)

_executor = ThreadPoolExecutor(max_workers=8)

# Number of energy samples in a fingerprint
_NUM_POINTS = 64


def _audio_fingerprint_sync(
    file_path: str,
    num_points: int = _NUM_POINTS,
) -> List[float]:
    """Extract a compact audio energy profile.

    Steps:
      1. Decode the full audio track to 8 kHz mono 16-bit PCM via FFmpeg
      2. Split into `num_points` equal-length segments
      3. Compute RMS energy per segment
      4. Normalise to [0, 1]

    Returns a list of `num_points` floats, or [] on failure.
    """
    try:
        cmd = [
            "ffmpeg",
            "-i", str(file_path),
            "-vn",
            "-ac", "1",
            "-ar", "8000",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "pipe:1",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=120,
            creationflags=_CREATION_FLAGS,
        )

        if result.returncode != 0 or len(result.stdout) < num_points * 2:
            return []

        samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
        if len(samples) < num_points * 100:  # need at least 100 samples per segment
            return []

        seg_size = len(samples) // num_points
        energies: List[float] = []
        for i in range(num_points):
            seg = samples[i * seg_size : (i + 1) * seg_size]
            rms = float(np.sqrt(np.mean(seg ** 2)))
            energies.append(rms)

        # Normalise to [0, 1]
        peak = max(energies) if energies else 1.0
        if peak > 0:
            energies = [e / peak for e in energies]

        return energies

    except Exception as e:
        print(f"[AUDIO FP] Error for {file_path}: {e}")
        return []


def compare_audio_fingerprints(
    fp1: List[float],
    fp2: List[float],
) -> float:
    """Compare two audio fingerprints via normalised cross-correlation.

    Returns a similarity percentage 0-100.
    Values > 85 % almost certainly indicate the same audio content.
    """
    if not fp1 or not fp2:
        return 0.0

    min_len = min(len(fp1), len(fp2))
    a = np.array(fp1[:min_len], dtype=np.float64)
    b = np.array(fp2[:min_len], dtype=np.float64)

    a_mean = a - np.mean(a)
    b_mean = b - np.mean(b)

    denom = np.sqrt(np.sum(a_mean ** 2) * np.sum(b_mean ** 2))
    if denom < 1e-10:
        return 0.0

    correlation = float(np.sum(a_mean * b_mean) / denom)
    return max(0.0, min(100.0, correlation * 100.0))


# ── Async wrappers ────────────────────────────────────────────────────────────

async def audio_fingerprint(file_path: str) -> List[float]:
    """Async wrapper for audio fingerprint extraction."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, _audio_fingerprint_sync, file_path
    )
