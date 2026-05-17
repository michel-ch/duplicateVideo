# Duplicate Detection

How the system decides that two videos are "the same content."

## The OR rule

Two videos are duplicates if **any** of three conditions hold:

1. **Byte-identical fast-path** — same `file_size` AND same `head_tail_xxh3` (blake2b digest of first + last 64 KiB). The pipeline declares these duplicates before stage 3 even runs; one representative per cluster does the pHash + audio work, all others inherit the result.
2. **Visual match** — average best-match pHash Hamming distance ≤ `HASH_SIMILARITY_THRESHOLD` (default 14, on a 256-bit hash). For duration groups of ≥ 16 videos with cached aggregate hashes, a FAISS `IndexBinaryFlat.range_search` at radius `1.5 × threshold` shortlists candidate pairs before the 12×12 verifier runs.
3. **Audio match** — normalised cross-correlation of audio energy profiles ≥ 80% (RMS profile of the middle 60 s of audio).

Conditions 2 and 3 must first survive a duration pre-filter and a 20× file-size sanity check.

The OR semantics are deliberate. The byte-identical path catches literal backup copies (cheapest of all); visual matching catches re-encodes that look the same; audio matching catches re-encodes that **don't** look the same (portrait vs landscape, severe re-cropping, different overlays) but share the soundtrack.

## Stage-by-stage

### Stage 1 — duration grouping

Implemented by `comparator.py:group_by_duration`.

```
sort videos by duration
take videos[0] as the anchor
for each next video:
    tolerance = max(DURATION_TOLERANCE_SECONDS, 0.05 * anchor)
    if |duration - anchor| <= tolerance: add to group
    else: close group; start new group with new anchor
drop singleton groups
```

The combined absolute + 5% relative tolerance is the smallest tweak that handles re-encodes whose container reports a slightly different duration. A 30-minute clip needs ~90s of headroom; the default `DURATION_TOLERANCE_SECONDS=3.0` would be far too tight.

Note that the anchor moves to the next out-of-tolerance video, which means a slow drift in durations (e.g. 100s, 102s, 104s, 106s with tol=3) **breaks** at the first gap. This is intentional — these are not duplicates of each other in any reasonable sense.

### Stage 2 — file-size sanity check

`_file_size_compatible(v1, v2, ratio=20.0)` returns False if the larger file is more than 20× the smaller. This is generous on purpose — same-content re-encodes can vary 10× or more (e.g. uncompressed source vs HEVC at 2 Mbps).

If either file size is unknown, the check passes.

### Stage 2.5 — byte-identical fast-path

Implemented inline in `api/scan.py` between stages 2 and 3. Videos sharing `(file_size, head_tail_xxh3)` cluster together; the lexicographically-first path is the representative, the rest are followers. Followers skip stages 3 and 4b and inherit the representative's hashes / audio FP after each runs. This both saves work AND preserves transitive matching — a follower can still link to a transcode via the shared pHashes.

The probability of two genuinely different files sharing both `file_size` and a 64-bit blake2b head/tail digest is on the order of `2^-64`. In practice this catches only true byte-identical duplicates (backup copies, mirrors, etc.).

### Stage 3 — perceptual hash comparison

The interesting algorithm. Implemented in `hasher.py:compare_hash_sets`. For large duration groups, FAISS does the candidate filtering first (see "FAISS prescreen" below).

#### Why best-match instead of positional

Naive comparison would do `hash1[i] vs hash2[i]` for each i. That fails when:

- Source and re-encode have different fps (so frame N in each comes from different timestamps).
- One version has a slightly different start time (a few seconds of trim).
- Frame extraction picks slightly different timestamps due to keyframe alignment.

Best-match comparison instead pairs each hash in set 1 with its **nearest** hash in set 2.

#### The implementation

```
1. Convert all hashes to numpy bit arrays.
2. Build distance matrix dist[n1, n2] of Hamming distances.
3. Flatten and argsort -> ascending order of distances.
4. Greedy assignment: for each (i, j) in order, claim it if neither i nor j is taken yet.
5. After enough matches, check running average:
     - if avg_so_far <= threshold * 0.5  → strong match, extrapolate, return True
     - if avg_so_far  > threshold * 2    → clearly not a match, return False
6. Otherwise: average distance over min(n1, n2) pairs; compare to threshold.
```

Why the early-exit: vectorised over 12-vs-12 = 144 pairs the matrix is cheap, but if you can return after 4 lopsided pairs you save time on huge groups.

#### Similarity score

Distance is converted to a percentage:

```
similarity = max(0, (1 - avg_distance / 256)) * 100
```

256 is the bit count of a 16×16 pHash. This is the value stored on `DuplicateGroup.similarity_score` after averaging across the connected component.

#### FAISS prescreen (large duration groups)

When a duration group has ≥ 16 videos AND most of them have a cached `aggregate_hash` AND `faiss-cpu` is installed, `_faiss_phash_candidates` builds an in-memory `faiss.IndexBinaryFlat` keyed by aggregate hash (one 256-bit code per video — a per-bit majority vote over the 12 frame hashes), then `range_search(arr, radius = 1.5 × hash_threshold)` returns a set of candidate `(i, j)` pairs.

The verifier (`compare_hash_sets`) then runs only on the shortlist. Pairs not shortlisted (and pairs where at least one side has no aggregate hash) still get checked as before.

The radius is widened to `1.5 ×` because the aggregate is a lossy summary of the 12 individual hashes — a pair whose verifier distance is at the boundary can show a slightly larger Hamming distance at the aggregate level.

Falls back to all-pairs (the pre-FAISS behaviour) when the group is small, aggregates are missing, FAISS isn't installed, or anything raises (logged via `print` so a programming bug doesn't silently regress).

#### Letterbox stripping

After each frame is extracted, `_strip_letterbox` runs a Python/numpy bbox detection (threshold 24/255, only crops when ≥ 5 % of either dimension is dark border) before `imagehash.phash`. Two encodes that differ only in black-bar padding now produce matching hashes — closes the case where the same content was once delivered in 16:9 letterbox and once in pillarbox 2.35:1.

### Stage 4 — audio fallback

Applied when the visual stage didn't match the pair. Computed by `audio_fingerprint.py:compare_audio_fingerprints`.

#### How the fingerprint is computed

```
1. If duration > 60s: ffmpeg -ss (duration - 60) / 2 -t 60 -i FILE ... pipe:1
   Else:               ffmpeg -t 60 -i FILE ... pipe:1
   → up to 60 s of audio decoded to 8 kHz mono 16-bit PCM, piped to stdout.
2. Split samples into 64 equal segments.
3. RMS energy per segment: sqrt(mean(samples^2))
4. Normalise by the peak energy.
```

Result: 64-element vector in `[0, 1]`. Small enough that comparing 1000² pairs of them is microseconds.

Sampling the middle 60 seconds is a major perf win on long videos — decoding a full 2-hour movie's audio used to take many seconds; the 60-second slice takes a fraction of one. The discriminative power of the RMS profile is preserved because two re-encodes of the same source will have nearly identical middle-60s envelopes too. `AUDIO_FP_VERSION = 2` exists to invalidate older full-track 64-point profiles that aren't comparable to the new middle-60s profiles.

#### How the comparison works

Pearson cross-correlation:

```
a' = a - mean(a)
b' = b - mean(b)
correlation = sum(a' * b') / sqrt(sum(a'²) * sum(b'²))
similarity_pct = max(0, min(100, correlation * 100))
```

A correlation > 0.85 between RMS energy profiles is essentially impossible by chance — same audio at different bitrates / containers / sample rates almost always lands at > 0.95.

The threshold for the audio match is the constant `audio_threshold = 80.0`, hard-coded in `comparator.py:find_duplicates_in_group`.

### Stage 5 — Union-Find

Pairwise matches are merged with Union-Find so transitive duplicates collapse. If A↔B by audio and B↔C by video, A/B/C end up in one group.

```python
parent = list(range(n))
def find(x): ...
def union(a, b): ...

for each (i, j) pair that matched:
    union(i, j)

groups = defaultdict(list)
for i in range(n):
    groups[find(i)].append(videos[i])
return [g for g in groups.values() if len(g) > 1]
```

This is what the user sees as a "duplicate group."

## Robustness tricks worth knowing

### SAR normalisation

Anamorphic encodes (e.g. SAR `81:256`) compress wide content into a narrow coded frame. Two encodes of the same content with different SARs would produce completely different pHashes if you just used coded pixels.

Fix: every frame extraction applies `scale=iw*sar:ih,setsar=1` **before** scaling to `320:-2`. After this, both encodes produce frames at the same display proportions.

### Portrait → landscape rotation

Phone videos arrive as 1920×1080 + rotation=90. Web re-uploads might bake the rotation in and arrive as 1080×1920. Without normalisation, the pHashes would differ wildly.

Fix: `_is_display_portrait(vinfo)` checks both rotation metadata and (post-SAR) display orientation. If portrait, we apply `transpose=1` before scaling. Two portrait videos of the same content always end up landscape with identical orientation.

### Best-match over positional

Already covered above — the most important single algorithmic decision in the comparator. Two re-encodes at 24fps and 30fps produce frames at different timestamps; positional comparison would fail.

## What this system does *not* detect

- **Same content with different audio** that also has visually distinct re-formatting. Both signals must fail.
- **Mirror-flipped content.** pHashes of mirrored frames are typically very different (Hamming distance > 100 on a 256-bit hash). Audio still works, however.
- **Same content with completely different durations** (e.g. trailer vs full film). The duration pre-filter rejects these. There's a 5% relative tolerance for safety, but a 2-minute trailer vs a 90-minute film won't make it past stage 1.

## Tuning

| Setting | Default | Effect of raising |
|---|---|---|
| `HASH_SIMILARITY_THRESHOLD` | 14 | More tolerant visual matching, more false positives |
| `DURATION_TOLERANCE_SECONDS` | 3.0 | Catches more re-encodes with mistimed containers, but groups widen |
| `KEY_FRAMES_COUNT` | 12 | More frames = more robust to single-frame outliers, slower |
| `audio_threshold` (const) | 80.0 | Stricter audio match required (raise to 90 if false positives, lower to 70 for noisy re-encodes) |

The constants `audio_threshold` and the `0.5×` / `2×` early-exit multipliers in `compare_hash_sets` are not exposed in the Settings API.

## Diagnostics

If two specific files aren't matching the way you expect, run:

```bash
python backend/diagnose_pair.py "path/to/A.mp4" "path/to/B.mp4"
```

It prints duration/SAR/rotation/portrait flags, frame counts and dimensions, video similarity, audio correlation, and the final verdict — exactly what the pipeline computes, but for a single pair, with full output.
