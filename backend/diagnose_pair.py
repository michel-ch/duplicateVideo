"""Diagnostic script: test the full pipeline on two files.

Usage:
    python diagnose_pair.py "path/to/video1" "path/to/video2"
"""
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from services.hasher import (
    _get_video_duration,
    _get_video_codec,
    _get_video_info,
    _is_display_portrait,
    _has_non_square_sar,
    _extract_frames_sync,
    _compute_hashes_sync,
    compare_hash_sets,
    compute_hamming_distance,
)
from services.audio_fingerprint import (
    _audio_fingerprint_sync,
    compare_audio_fingerprints,
)
from config import settings


def diagnose(file1: str, file2: str):
    sep = "=" * 70
    print(sep)
    print("DUPLICATE DETECTION DIAGNOSTIC")
    print(sep)

    for idx, fp in enumerate([file1, file2], 1):
        print(f"\n--- FILE {idx} ---")
        print(f"  Path: {fp}")
        if not os.path.exists(fp):
            print("  FILE NOT FOUND")
            continue

        vinfo = _get_video_info(fp)
        duration = _get_video_duration(fp)
        w, h = vinfo["width"], vinfo["height"]
        sn, sd = vinfo["sar_num"], vinfo["sar_den"]
        display_w = w * sn / sd
        display_h = h
        if vinfo["rotation"] in (90, 270):
            display_w, display_h = display_h, display_w

        print(f"  Duration: {duration}")
        print(f"  Coded: {w}x{h}  SAR: {sn}:{sd}  Display: {display_w:.0f}x{display_h:.0f}")
        print(f"  Portrait: {_is_display_portrait(vinfo)}  Anamorphic: {_has_non_square_sar(vinfo)}")

    # Duration
    print(f"\n--- DURATION ---")
    d1, d2 = _get_video_duration(file1), _get_video_duration(file2)
    if d1 and d2:
        diff = abs(d1 - d2)
        tol = max(settings.DURATION_TOLERANCE_SECONDS, max(d1, d2) * 0.05)
        ok = diff <= tol
        print(f"  {d1:.3f}s vs {d2:.3f}s  (diff={diff:.3f}s, tol={tol:.1f}s)  {'PASS' if ok else 'FAIL'}")

    # Frames
    print(f"\n--- VIDEO HASHES ---")
    tmp1, tmp2 = tempfile.mkdtemp(), tempfile.mkdtemp()
    nf = settings.KEY_FRAMES_COUNT
    frames1 = _extract_frames_sync(file1, nf, tmp1)
    frames2 = _extract_frames_sync(file2, nf, tmp2)
    if frames1:
        from PIL import Image
        print(f"  File 1: {len(frames1)} frames, size={Image.open(frames1[0]).size}")
    if frames2:
        from PIL import Image
        print(f"  File 2: {len(frames2)} frames, size={Image.open(frames2[0]).size}")

    h1 = _compute_hashes_sync(frames1)
    h2 = _compute_hashes_sync(frames2)
    if h1 and h2:
        ok, sim = compare_hash_sets(h1, h2, settings.HASH_SIMILARITY_THRESHOLD)
        print(f"  Video similarity: {sim:.1f}%  threshold={settings.HASH_SIMILARITY_THRESHOLD}  {'MATCH' if ok else 'NO MATCH'}")

    # Audio fingerprints
    print(f"\n--- AUDIO FINGERPRINTS ---")
    print(f"  Extracting audio from file 1...")
    afp1 = _audio_fingerprint_sync(file1)
    print(f"  Got {len(afp1)} points")
    print(f"  Extracting audio from file 2...")
    afp2 = _audio_fingerprint_sync(file2)
    print(f"  Got {len(afp2)} points")

    if afp1 and afp2:
        audio_sim = compare_audio_fingerprints(afp1, afp2)
        print(f"  Audio correlation: {audio_sim:.1f}%")
        if audio_sim >= 80:
            print(f"  AUDIO MATCH (>= 80%)")
        else:
            print(f"  NO AUDIO MATCH (< 80%)")

    # Overall
    print(f"\n--- OVERALL ---")
    video_match = h1 and h2 and compare_hash_sets(h1, h2, settings.HASH_SIMILARITY_THRESHOLD)[0]
    audio_match = afp1 and afp2 and compare_audio_fingerprints(afp1, afp2) >= 80
    if video_match:
        print("  DUPLICATE (matched by video hashes)")
    elif audio_match:
        print("  DUPLICATE (matched by audio fingerprint)")
    else:
        print("  NOT DUPLICATE (neither video nor audio matched)")

    import shutil
    shutil.rmtree(tmp1, ignore_errors=True)
    shutil.rmtree(tmp2, ignore_errors=True)
    print(sep)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <video1> <video2>")
        sys.exit(1)
    diagnose(sys.argv[1], sys.argv[2])
