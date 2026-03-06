"""Frame extraction and perceptual hashing for duplicate detection.

GPU Acceleration (NVIDIA RTX 3060 Ti):
  - Uses CUVID hardware decoders (h264_cuvid, hevc_cuvid, …) when available
  - Extracts ALL frames in a single FFmpeg call (batch)
  - Falls back gracefully to CPU if GPU is unavailable

Duplicate Detection Robustness:
  - Applies SAR (Sample Aspect Ratio) during extraction so anamorphic
    encodes (e.g. SAR 81:256 → 9:16 display) produce the same frames as
    non-anamorphic encodes of the same content
  - Detects portrait display orientation and rotates to landscape so the
    same content in 9:16 and 16:9 produces matching hashes
  - Best-match hash comparison (not positional) handles different FPS
"""

import subprocess
import tempfile
import os
import asyncio
import json
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np

try:
    import imagehash
    from PIL import Image
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

from config import settings
from services.gpu_detector import get_gpu_info

# Shared thread pool — GPU can handle many decode streams concurrently
_executor = ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_FFMPEG * 3)

# ── helpers ───────────────────────────────────────────────────────────────────

_CREATION_FLAGS = (
    subprocess.CREATE_NO_WINDOW
    if hasattr(subprocess, "CREATE_NO_WINDOW")
    else 0
)


def _get_video_duration(file_path: str) -> Optional[float]:
    """Get duration via ffprobe (fast, no GPU needed)."""
    try:
        cmd = [
            settings.FFPROBE_PATH,
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "json",
            str(file_path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, timeout=15,
            creationflags=_CREATION_FLAGS,
        )
        data = json.loads(result.stdout.decode("utf-8"))
        return float(data["format"]["duration"])
    except Exception:
        return None


def _get_video_codec(file_path: str) -> Optional[str]:
    """Quick probe for the video codec name (e.g. 'h264', 'hevc')."""
    try:
        cmd = [
            settings.FFPROBE_PATH,
            "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "csv=p=0",
            str(file_path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, timeout=10,
            creationflags=_CREATION_FLAGS,
        )
        return result.stdout.decode("utf-8").strip().lower()
    except Exception:
        return None


def _get_video_info(file_path: str) -> dict:
    """Get width, height, SAR, rotation from a single ffprobe call."""
    info = {"width": 0, "height": 0, "sar_num": 1, "sar_den": 1, "rotation": 0}
    try:
        cmd = [
            settings.FFPROBE_PATH,
            "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,sample_aspect_ratio,display_aspect_ratio",
            "-show_entries", "stream_side_data=rotation",
            "-show_entries", "stream_tags=rotate",
            "-of", "json",
            str(file_path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, timeout=10,
            creationflags=_CREATION_FLAGS,
        )
        data = json.loads(result.stdout.decode("utf-8"))

        streams = data.get("streams", [])
        if not streams:
            return info

        s = streams[0]
        info["width"] = int(s.get("width", 0))
        info["height"] = int(s.get("height", 0))

        # Parse SAR (e.g. "81:256", "1:1", "0:1")
        sar_str = s.get("sample_aspect_ratio", "1:1")
        if sar_str and ":" in sar_str:
            parts = sar_str.split(":")
            sn, sd = int(parts[0]), int(parts[1])
            if sn > 0 and sd > 0:
                info["sar_num"] = sn
                info["sar_den"] = sd

        # Rotation from side_data
        for sd in s.get("side_data_list", []):
            rot = sd.get("rotation")
            if rot is not None:
                info["rotation"] = int(float(rot)) % 360

        # Rotation from tags
        tags = s.get("tags", {})
        rot = tags.get("rotate", tags.get("ROTATE"))
        if rot and info["rotation"] == 0:
            info["rotation"] = int(float(rot)) % 360

    except Exception:
        pass
    return info


def _is_display_portrait(info: dict) -> bool:
    """Check if the VIDEO's display orientation is portrait (taller than wide).

    Takes into account both SAR (sample aspect ratio) and rotation metadata.
    """
    w = info["width"]
    h = info["height"]
    sar_num = info["sar_num"]
    sar_den = info["sar_den"]
    rotation = info["rotation"]

    if w <= 0 or h <= 0:
        return False

    # Compute display dimensions
    display_w = w * sar_num / sar_den
    display_h = h

    # Apply rotation
    if rotation in (90, 270):
        display_w, display_h = display_h, display_w

    return display_h > display_w


def _has_non_square_sar(info: dict) -> bool:
    """Check if SAR is non-trivial (i.e. anamorphic content)."""
    return info["sar_num"] != info["sar_den"]


# ── Frame extraction ──────────────────────────────────────────────────────────

def _build_frame_extract_cmd(
    file_path: str,
    output_pattern: str,
    num_frames: int,
    duration: float,
    codec: Optional[str] = None,
    video_info: Optional[dict] = None,
) -> List[str]:
    """
    Build FFmpeg command for batch frame extraction.

    Key normalisation steps (ensure same content → same hashes):
      1. Apply SAR via  scale=iw*sar:ih,setsar=1  so anamorphic content
         is un-squeezed to its display proportions
      2. Rotate portrait to landscape via  transpose=1  so 9:16 and 16:9
         encodes of the same content produce matching frames
      3. Scale to fixed width  scale=320:-2  for consistent hash input

    GPU decode is used when available; all pixel transforms run on CPU
    after hwdownload for maximum compatibility.

    Pass `video_info` (dict with width/height/sar_num/sar_den/rotation)
    from a prior metadata step to skip the redundant ffprobe call.
    """
    gpu = get_gpu_info()
    use_gpu = False
    cuvid_decoder = None

    if gpu.available and gpu.hwaccel_supported and codec:
        cuvid_decoder = gpu.get_decoder(codec)
        if cuvid_decoder:
            use_gpu = True

    # Calculate FPS to produce exactly `num_frames` evenly spaced
    target_fps = num_frames / duration if duration > 0 else 1

    # Use pre-computed video properties if available, otherwise probe
    vinfo = video_info if video_info else _get_video_info(file_path)
    is_portrait = _is_display_portrait(vinfo)
    has_sar = _has_non_square_sar(vinfo)

    cmd: List[str] = ["ffmpeg", "-y"]

    if use_gpu:
        cmd += [
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-c:v", cuvid_decoder,
        ]

    cmd += ["-i", str(file_path)]

    # ── Build filter chain ──
    # GPU path: decode on GPU → hwdownload → all transforms on CPU
    # CPU path: everything on CPU
    filters: List[str] = [f"fps={target_fps}"]

    if use_gpu:
        filters.append("hwdownload")
        filters.append("format=nv12")

    # Step 1: Apply SAR to get correct display proportions
    # This is critical for anamorphic encodes (SAR != 1:1)
    if has_sar:
        filters.append("scale=iw*sar:ih")
        filters.append("setsar=1")

    # Step 2: Rotate portrait to landscape for consistent hashing
    if is_portrait:
        filters.append("transpose=1")

    # Step 3: Scale to standard width (height auto, even number)
    filters.append("scale=320:-2")

    cmd += ["-vf", ",".join(filters)]
    cmd += [
        "-q:v", "2",
        "-frames:v", str(num_frames),
        output_pattern,
    ]
    return cmd


def _extract_frames_sync(
    file_path: str,
    num_frames: int = 8,
    output_dir: Optional[str] = None,
    duration: Optional[float] = None,
    codec: Optional[str] = None,
    video_info: Optional[dict] = None,
) -> List[str]:
    """
    Extract N evenly spaced frames using a SINGLE FFmpeg call.
    Uses GPU decoding when available.  All frames are normalised
    (SAR applied, portrait rotated to landscape) for consistent hashing.

    If `duration`, `codec`, and `video_info` are provided (from a prior
    metadata pass), all extra ffprobe calls are skipped entirely.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="viddup_frames_")
    os.makedirs(output_dir, exist_ok=True)

    if duration is None:
        duration = _get_video_duration(file_path)
    if not duration or duration <= 0:
        return []

    if codec is None:
        codec = _get_video_codec(file_path)
    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")

    cmd = _build_frame_extract_cmd(
        file_path, output_pattern, num_frames, duration, codec,
        video_info=video_info,
    )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=60,
            creationflags=_CREATION_FLAGS,
        )

        # Collect produced frames
        frame_paths = []
        for i in range(1, num_frames + 1):
            fp = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            if os.path.exists(fp) and os.path.getsize(fp) > 0:
                frame_paths.append(fp)

        # If GPU extraction failed (0 frames), retry on CPU
        if not frame_paths and codec:
            gpu_info = get_gpu_info()
            if gpu_info.available and gpu_info.get_decoder(codec):
                cpu_cmd = _build_frame_extract_cmd(
                    file_path, output_pattern, num_frames, duration, codec=None
                )
                subprocess.run(
                    cpu_cmd,
                    capture_output=True,
                    timeout=60,
                    creationflags=_CREATION_FLAGS,
                )
                for i in range(1, num_frames + 1):
                    fp = os.path.join(output_dir, f"frame_{i:04d}.jpg")
                    if os.path.exists(fp) and os.path.getsize(fp) > 0:
                        frame_paths.append(fp)

        return frame_paths

    except Exception as e:
        print(f"[GPU/CPU] Frame extraction error for {file_path}: {e}")
        return []


# ── Perceptual hashing ────────────────────────────────────────────────────────

def _compute_hashes_sync(frame_paths: List[str]) -> List[str]:
    """Compute perceptual hashes for a list of frame images."""
    if not HAS_IMAGEHASH:
        return []

    hashes = []
    for fp in frame_paths:
        try:
            img = Image.open(fp)
            h = imagehash.phash(img, hash_size=16)
            hashes.append(str(h))
        except Exception:
            continue

    return hashes


def _extract_and_hash_sync(
    file_path: str,
    num_frames: int = 8,
    duration: Optional[float] = None,
    codec: Optional[str] = None,
    video_info: Optional[dict] = None,
) -> dict:
    """Extract frames (GPU-accelerated, normalised) and compute hashes.

    Pass `duration`, `codec`, and `video_info` from a prior metadata
    step to skip all redundant ffprobe calls.
    """
    tmp_dir = tempfile.mkdtemp(prefix="viddup_")
    try:
        frames = _extract_frames_sync(
            file_path, num_frames, tmp_dir,
            duration=duration, codec=codec,
            video_info=video_info,
        )
        hashes = _compute_hashes_sync(frames)
        return {
            "file_path": file_path,
            "hashes": hashes,
            "error": None,
        }
    except Exception as e:
        return {
            "file_path": file_path,
            "hashes": [],
            "error": str(e),
        }
    finally:
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


# ── Thumbnail extraction ─────────────────────────────────────────────────────

def _extract_thumbnail_sync(
    file_path: str,
    output_path: str,
    duration: Optional[float] = None,
    codec: Optional[str] = None,
) -> Optional[str]:
    """
    Extract a single thumbnail from the middle of the video.
    Uses GPU decoding + scale_cuda when available.

    Pass `duration` and `codec` from a prior metadata step to skip
    2 redundant ffprobe subprocess calls per video.
    """
    try:
        if duration is None:
            duration = _get_video_duration(file_path)
        if not duration or duration <= 0:
            return None
        mid_point = duration / 2

        if codec is None:
            codec = _get_video_codec(file_path)
        gpu = get_gpu_info()
        use_gpu = False
        cuvid_decoder = None

        if gpu.available and gpu.hwaccel_supported and codec:
            cuvid_decoder = gpu.get_decoder(codec)
            if cuvid_decoder:
                use_gpu = True

        cmd = ["ffmpeg", "-y"]

        if use_gpu:
            cmd += [
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
                "-c:v", cuvid_decoder,
            ]

        cmd += ["-ss", str(mid_point), "-i", str(file_path)]

        if use_gpu and "scale_cuda" in gpu.cuda_filters:
            cmd += ["-vf", "scale_cuda=320:-1,hwdownload,format=nv12"]
        elif use_gpu:
            cmd += ["-vf", "hwdownload,format=nv12,scale=320:-1"]
        else:
            cmd += ["-vf", "scale=320:-1"]

        cmd += [
            "-vframes", "1",
            "-q:v", "4",
            output_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=15,
            creationflags=_CREATION_FLAGS,
        )

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path

        # GPU fallback → CPU
        if use_gpu:
            cpu_cmd = [
                "ffmpeg", "-y",
                "-ss", str(mid_point),
                "-i", str(file_path),
                "-vf", "scale=320:-1",
                "-vframes", "1",
                "-q:v", "4",
                output_path,
            ]
            subprocess.run(
                cpu_cmd, capture_output=True, timeout=15,
                creationflags=_CREATION_FLAGS,
            )
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path

    except Exception as e:
        print(f"[GPU/CPU] Thumbnail extraction error: {e}")

    return None


# ── Hash comparison ───────────────────────────────────────────────────────────

def _hex_to_bits(hex_str: str) -> Optional[np.ndarray]:
    """Convert a hex hash string to a numpy uint8 bit-array (fast)."""
    try:
        byte_arr = bytes.fromhex(hex_str)
        return np.unpackbits(np.frombuffer(byte_arr, dtype=np.uint8))
    except Exception:
        return None


def compute_hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hex-encoded hashes."""
    if not hash1 or not hash2:
        return 999

    try:
        b1 = _hex_to_bits(hash1)
        b2 = _hex_to_bits(hash2)
        if b1 is not None and b2 is not None and len(b1) == len(b2):
            return int(np.count_nonzero(b1 != b2))

        # Fallback for mismatched lengths
        if HAS_IMAGEHASH:
            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            return h1 - h2

        val1 = int(hash1, 16)
        val2 = int(hash2, 16)
        return bin(val1 ^ val2).count("1")
    except Exception:
        return 999


def compare_hash_sets(
    hashes1: List[str],
    hashes2: List[str],
    threshold: int = 10,
) -> Tuple[bool, float]:
    """
    Compare two sets of perceptual hashes using BEST-MATCH pairing.

    Instead of comparing positionally (hash1[0] vs hash2[0]), we find the
    best match for each hash in set 1 from set 2.  This handles different
    frame-rates, slight trims, and different starting points.

    Uses vectorised numpy for the distance matrix and early-exit when the
    first few matches are very close.

    Returns (is_similar, similarity_percentage).
    """
    if not hashes1 or not hashes2:
        return False, 0.0

    n1, n2 = len(hashes1), len(hashes2)
    min_len = min(n1, n2)

    # Pre-convert all hashes to bit arrays once
    bits1 = [_hex_to_bits(h) for h in hashes1]
    bits2 = [_hex_to_bits(h) for h in hashes2]

    # Build distance matrix with numpy (vectorised per-row)
    dist = np.full((n1, n2), 999, dtype=np.int32)
    for i, b1 in enumerate(bits1):
        if b1 is None:
            continue
        for j, b2 in enumerate(bits2):
            if b2 is not None and len(b1) == len(b2):
                dist[i, j] = int(np.count_nonzero(b1 != b2))

    # Flatten and argsort for greedy best-match
    flat = dist.ravel()
    order = np.argsort(flat, kind="quicksort")

    used_i: set = set()
    used_j: set = set()
    matched_distances: List[int] = []
    running_sum = 0

    for idx in order:
        d = int(flat[idx])
        if d >= 999:
            break  # no more valid pairs
        i, j = divmod(int(idx), n2)
        if i in used_i or j in used_j:
            continue
        matched_distances.append(d)
        running_sum += d
        used_i.add(i)
        used_j.add(j)

        count = len(matched_distances)
        # Early exit: if we have enough matches and running average is clearly below/above threshold
        if count >= min(4, min_len):
            avg_so_far = running_sum / count
            if avg_so_far <= threshold * 0.5:
                # Very strong match — pad remaining with current average
                remaining = min_len - count
                avg_distance = (running_sum + avg_so_far * remaining) / min_len
                similarity = max(0, (1 - avg_distance / 256)) * 100
                return True, similarity
            if avg_so_far > threshold * 2:
                # Clearly not a match — bail out early
                return False, max(0, (1 - avg_so_far / 256)) * 100

        if count >= min_len:
            break

    if not matched_distances:
        return False, 0.0

    avg_distance = running_sum / len(matched_distances)
    is_similar = avg_distance <= threshold

    max_bits = 256  # 16×16 hash
    similarity = max(0, (1 - avg_distance / max_bits)) * 100

    return is_similar, similarity


# ── Async wrappers ────────────────────────────────────────────────────────────

async def extract_and_hash(
    file_path: str,
    num_frames: int = 8,
    duration: Optional[float] = None,
    codec: Optional[str] = None,
    video_info: Optional[dict] = None,
) -> dict:
    """Async wrapper for frame extraction and hashing (GPU-accelerated).

    Pass `duration`, `codec`, and `video_info` from a prior metadata
    step to avoid all redundant ffprobe calls.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        lambda: _extract_and_hash_sync(
            file_path, num_frames,
            duration=duration, codec=codec,
            video_info=video_info,
        ),
    )


async def extract_thumbnail(
    file_path: str,
    output_path: str,
    duration: Optional[float] = None,
    codec: Optional[str] = None,
) -> Optional[str]:
    """Async wrapper for thumbnail extraction (GPU-accelerated).

    Pass `duration` and `codec` to skip redundant ffprobe calls.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        lambda: _extract_thumbnail_sync(
            file_path, output_path,
            duration=duration, codec=codec,
        ),
    )
