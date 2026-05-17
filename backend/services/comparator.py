"""Duplicate comparison logic with multi-stage pipeline.

Stages:
  1. Duration grouping  — cluster videos with similar durations
  2. Video hash match   — pHash comparison (best-match) within each cluster.
                          For large clusters with cached aggregate hashes a
                          FAISS binary index does an O(n) candidate filter
                          before the expensive 12-frame verifier runs.
  3. Audio fallback     — if video hashes are inconclusive, compare audio
                          fingerprints to catch re-encodes with different
                          visual formatting (portrait ↔ landscape, SAR, etc.)
"""

import json
import asyncio
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

import numpy as np

from services.hasher import compare_hash_sets
from services.audio_fingerprint import compare_audio_fingerprints
from config import settings

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# Below this group size, building a FAISS index costs more than the
# O(n²) it would save.
_FAISS_MIN_GROUP_SIZE = 16


# ── Stage 1: Duration grouping ───────────────────────────────────────────────

def group_by_duration(
    videos: List[dict],
    tolerance: float = 2.0
) -> List[List[dict]]:
    """
    Group videos by approximate duration.

    Uses both absolute tolerance AND 5 % relative tolerance (whichever
    is larger) to handle re-encodes whose container reports a slightly
    different duration.
    """
    if not videos:
        return []

    valid_videos = [v for v in videos if v.get("duration") is not None]
    if not valid_videos:
        return []

    valid_videos.sort(key=lambda v: v["duration"])

    groups: List[List[dict]] = []
    current_group: List[dict] = [valid_videos[0]]
    anchor_duration: float = valid_videos[0]["duration"]

    for i in range(1, len(valid_videos)):
        vid = valid_videos[i]
        abs_diff = abs(vid["duration"] - anchor_duration)
        pct_tolerance = anchor_duration * 0.05 if anchor_duration > 0 else 0
        effective_tolerance = max(tolerance, pct_tolerance)

        if abs_diff <= effective_tolerance:
            current_group.append(vid)
        else:
            if len(current_group) > 1:
                groups.append(current_group)
            current_group = [vid]
            anchor_duration = vid["duration"]

    if len(current_group) > 1:
        groups.append(current_group)

    return groups


def _file_size_compatible(v1: dict, v2: dict, ratio: float = 20.0) -> bool:
    """Quick check: if file sizes differ by more than `ratio`x, skip comparison.

    Very generous default (20×) so we only skip extreme outliers
    (e.g. a 50 MB clip vs a 2 GB raw).  Same-duration same-content
    re-encodes rarely differ by more than 10×.
    """
    s1 = v1.get("file_size") or 0
    s2 = v2.get("file_size") or 0
    if s1 <= 0 or s2 <= 0:
        return True  # unknown size → don't skip
    lo, hi = (s1, s2) if s1 <= s2 else (s2, s1)
    return hi <= lo * ratio


# ── Stage 2 + 3: Hash comparison with audio fallback ─────────────────────────

def _faiss_phash_candidates(
    videos: List[dict],
    hash_threshold: int,
) -> Optional[Set[Tuple[int, int]]]:
    """Return the set of (i, j) pairs flagged by a FAISS binary range query
    on the per-video aggregate hash, or None if FAISS isn't usable here.

    The aggregate hash is a per-bit majority vote over the 12 frame pHashes
    (256 bits when hash_size=16).  Two videos whose aggregates are within
    `hash_threshold * 1.5` Hamming bits are SHORTLISTED — they still get
    the full 12×12 best-match verification via `compare_hash_sets`.

    Returns None to mean "use all-pairs", which the caller treats as the
    pre-FAISS code path.
    """
    if not HAS_FAISS or len(videos) < _FAISS_MIN_GROUP_SIZE:
        return None
    with_agg = [(idx, v["aggregate_hash"]) for idx, v in enumerate(videos)
                if v.get("aggregate_hash")]
    if len(with_agg) < _FAISS_MIN_GROUP_SIZE:
        return None
    try:
        # All aggregate hashes must be the same bit width to share an index.
        first_bytes = bytes.fromhex(with_agg[0][1])
        bit_width = len(first_bytes) * 8
        rows = []
        idxs = []
        for idx, agg in with_agg:
            try:
                b = bytes.fromhex(agg)
            except ValueError:
                continue
            if len(b) * 8 != bit_width:
                continue
            rows.append(np.frombuffer(b, dtype=np.uint8))
            idxs.append(idx)
        if len(rows) < _FAISS_MIN_GROUP_SIZE:
            return None
        arr = np.stack(rows, axis=0)
        index = faiss.IndexBinaryFlat(bit_width)
        index.add(arr)
        # Inflate the radius a little — the aggregate is a lossy summary of
        # the 12 frame hashes, so a true match on the 12-frame verifier can
        # show a slightly larger Hamming distance at the aggregate level.
        radius = max(1, int(hash_threshold * 1.5))
        lims, _D, I = index.range_search(arr, radius)
        candidates: Set[Tuple[int, int]] = set()
        for local_i in range(len(idxs)):
            global_i = idxs[local_i]
            for offset in range(int(lims[local_i]), int(lims[local_i + 1])):
                local_j = int(I[offset])
                if local_i == local_j:
                    continue
                global_j = idxs[local_j]
                a, b = (global_i, global_j) if global_i < global_j else (global_j, global_i)
                candidates.add((a, b))
        return candidates
    except Exception as e:
        # FAISS error → fall back to all-pairs.  Print so a programming
        # bug (shape mismatch, version skew) doesn't silently regress
        # large duration buckets to O(n²).
        print(f"[FAISS] candidate query failed, falling back to all-pairs: {e}")
        return None


def find_duplicates_in_group(
    videos: List[dict],
    hash_threshold: int = 10,
    audio_threshold: float = 80.0,
) -> List[List[dict]]:
    """
    Within a duration group, compare perceptual hashes AND audio fingerprints
    to find duplicate subsets.

    Two videos are considered duplicates if EITHER:
      a) their pHash distance ≤ hash_threshold  (visual match)
      b) their audio correlation ≥ audio_threshold  (audio match, catches
         re-encodes with different visual formatting)

    Uses Union-Find so transitive duplicates are merged.

    For large duration groups (≥ _FAISS_MIN_GROUP_SIZE) a FAISS binary
    range_search on the aggregate hash filters the O(n²) pHash pair set
    down to a shortlist before the expensive 12×12 verifier runs.
    """
    if len(videos) < 2:
        return []

    n = len(videos)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    phash_candidates = _faiss_phash_candidates(videos, hash_threshold)
    no_agg_idx: Set[int] = (
        {idx for idx, v in enumerate(videos) if not v.get("aggregate_hash")}
        if phash_candidates is not None else set()
    )

    for i in range(n):
        hashes_i = videos[i].get("hashes") or []
        audio_i = videos[i].get("audio_fp") or []

        for j in range(i + 1, n):
            # Quick file-size sanity check — skip extreme mismatches
            if not _file_size_compatible(videos[i], videos[j]):
                continue

            hashes_j = videos[j].get("hashes") or []
            audio_j = videos[j].get("audio_fp") or []

            similarity = 0.0
            matched = False
            match_method = None

            # FAISS-shortlisted pHash compare.  When FAISS is off (small
            # group / no aggregates / lib missing) phash_candidates is None
            # and every pair is checked, matching the pre-FAISS behaviour.
            should_check_phash = (
                phash_candidates is None
                or i in no_agg_idx
                or j in no_agg_idx
                or (i, j) in phash_candidates
            )
            if should_check_phash and hashes_i and hashes_j:
                is_similar, sim = compare_hash_sets(hashes_i, hashes_j, hash_threshold)
                if is_similar:
                    similarity = sim
                    matched = True
                    match_method = "video"

            # Audio fallback — if video didn't match but audio is available
            if not matched and audio_i and audio_j:
                audio_sim = compare_audio_fingerprints(audio_i, audio_j)
                if audio_sim >= audio_threshold:
                    similarity = audio_sim
                    matched = True
                    match_method = "audio"

            if matched:
                union(i, j)
                videos[i].setdefault("_similarities", {})[j] = similarity
                videos[j].setdefault("_similarities", {})[i] = similarity
                if match_method:
                    videos[i].setdefault("_match_methods", {})[j] = match_method
                    videos[j].setdefault("_match_methods", {})[i] = match_method

    group_map: Dict[int, List[dict]] = defaultdict(list)
    for i in range(n):
        has_data = videos[i].get("hashes") or videos[i].get("audio_fp")
        if has_data:
            group_map[find(i)].append(videos[i])

    return [g for g in group_map.values() if len(g) > 1]


def calculate_group_similarity(videos: List[dict]) -> float:
    """Average pairwise similarity score for videos in a duplicate group."""
    similarities = []
    for v in videos:
        for s in v.get("_similarities", {}).values():
            similarities.append(float(s))

    if similarities:
        return sum(similarities) / len(similarities)

    total, count = 0.0, 0
    for i in range(len(videos)):
        for j in range(i + 1, len(videos)):
            h1 = videos[i].get("hashes") or []
            h2 = videos[j].get("hashes") or []
            if h1 and h2:
                _, sim = compare_hash_sets(h1, h2)
                total += sim
                count += 1

    return total / count if count > 0 else 0.0


def run_duplicate_pipeline(
    videos: List[dict],
    duration_tolerance: float = 2.0,
    hash_threshold: int = 10,
    audio_threshold: float = 80.0,
) -> List[dict]:
    """
    Full duplicate detection pipeline.

    Stage 1 — Duration filter
    Stage 2 — Video hash comparison (pHash best-match)
    Stage 3 — Audio fingerprint fallback (catches visual re-formats)

    Returns a list of group dicts:
        { "videos": [...], "similarity_score": float (0–100) }
    """
    # Stage 1
    duration_groups = group_by_duration(videos, duration_tolerance)

    # Stages 2 + 3
    duplicate_groups = []
    for dgroup in duration_groups:
        sub_groups = find_duplicates_in_group(dgroup, hash_threshold, audio_threshold)
        for sg in sub_groups:
            similarity = calculate_group_similarity(sg)
            for v in sg:
                v.pop("_similarities", None)
                v.pop("_match_methods", None)
            duplicate_groups.append({
                "videos": sg,
                "similarity_score": round(similarity, 2),
            })

    return duplicate_groups
