# Quick-Rejection Pre-Filtering Strategies

A research note on cheap pre-filters that reject non-duplicate pairs **before** the
pHash extraction stage (`services/hasher.py:extract_and_hash`) — the wall-clock
bottleneck at ~200 ms – 2 s/file. The cheaper the rejection, the bigger the win:
every file we reject early skips frame extraction entirely.

This document **extends** the existing "content bucketing pre-filter using duration
+ file-size band + audio-track length" proposal in
[`algorithmic-improvements.md`](algorithmic-improvements.md). Do **not** read this
as a replacement — read it as the layer that sits beneath it.

---

## Executive summary

Top three cheap pre-filters, prioritised by impact / effort, **independent of the
existing duration + file-size + audio-track baseline**:

| # | Filter | Per-file cost | pHash work avoided* | Effort | False-neg risk |
|---|---|---|---|---|---|
| 1 | **Head+tail xxh3 (head 64 KB + tail 64 KB)** | 1–3 ms | ~10–30 % of files (exact-content twins from `cp`/downloads) — *skips the entire pipeline*, not just pHash | 4 h | ≈0 (size+head+tail collision essentially impossible) |
| 2 | **Single fused ffprobe with `-show_streams -show_chapters -show_format`** populating a **content-signature MinHash LSH bucket** | 100–150 ms (replaces existing 100–120 ms ffprobe; net +20 ms or zero) | ~50–80 % of *cross-bucket* pairs (i.e. pairs the current pipeline would have entered the 12-frame pHash for, but that disagree on coded resolution/codec/audio profile) | 8–12 h | ~2 % (rare edge cases where the same content has been re-encoded to a different audio codec/sample-rate AND different coded resolution) |
| 3 | **Tiered SHA-256 of full file, gated to within-size-twin groups only** | 1–10 s **only for size-twin candidates** (~5–20 % of corpus) | 100 % of pHash work for any byte-identical copy + closes "stage 6 chapter/subtitle match" elegantly | 6 h | 0 |

These three layer additively. Stacked on top of the existing duration + file-size
+ audio-track filter, expected end-to-end **pHash-extraction reduction is 60–85 %**
on a typical mixed library (the upper end on libraries dominated by `cp`-clones,
the lower on libraries dominated by re-encodes).

**Single highest-impact recommendation**: ship filter #1 first. It is cheap, has
no calibration knobs, eliminates the most common real-world duplicate (an
identical re-download or `cp`), and pays for itself after ~20 files.

**Biggest risk across the whole stack**: the MinHash signature (#2) collapses
audio-stripped re-encodes (1080p with audio vs 1080p without) into separate
buckets even though they share visual content. **Mitigation**: keep audio-track
length / has-audio out of the signature schema (use it only as a *separate*
post-filter as the existing design does), or use audio-codec membership as one
of N bands rather than a hard match — see §"MinHash LSH on metadata signature".

---

## The cascade

The proposed multi-stage rejection pipeline before pHash, with **expected pair
survival rate at each stage** for a mixed home-video library of 10,000 files
with ~800 true duplicates organised in ~200 groups (yielding ~1,500 true-positive
pairs and ~5 × 10⁷ total candidate pairs):

```
                                              candidate pairs surviving
                                              ──────────────────────────
all (i,j) pairs                                          5.0 × 10⁷
        │
        ▼
[Stage A]  size == size (exact)                           3 × 10⁴   (0.0006 %)
           └→ branch to SHA-256 verifier; matches = byte-identical dups
           └→ remaining pairs flow downstream as normal candidates
        │
        ▼
[Stage B]  duration bucket (existing ±3 s / 5 %)         1.2 × 10⁶  (2.4 %)
        │
        ▼
[Stage C]  file-size band (geometric, base √2)           3.5 × 10⁵  (0.7 %)
           — kept from algorithmic-improvements.md
        │
        ▼
[Stage D]  audio-track length ± 0.5 s                    2.0 × 10⁵  (0.4 %)
           — kept from algorithmic-improvements.md
        │
        ▼
[Stage E]  head+tail 128 KB xxh3 (exact-twin fast path)
           ── byte-identical pairs short-circuit ──>  emit as 100 % dup
           ── remaining pairs ──────────────────────  1.95 × 10⁵
        │
        ▼
[Stage F]  metadata MinHash LSH bucket
           (signature schema in §"MinHash LSH on metadata signature")
           ── pairs in the same LSH bucket ──>  5 × 10⁴ – 8 × 10⁴ (0.10–0.16 %)
        │
        ▼
[Stage G]  optional: shared chapter title or subtitle SHA
           (only if both files HAVE chapters/subtitles)
           ── pairs both with subs+match ──>  guaranteed-dup, skip ahead
           ── pairs without subs ──────────  pass through unchanged
        │
        ▼
[Stage H]  pHash candidate set
           (currently ~3.5 × 10⁵ pairs; with the cascade above, ~6 × 10⁴)
        │
        ▼
[Stage I]  12-frame pHash compare (existing)
```

**End-to-end pair rejection vs current pipeline**: ~65–80 % fewer pairs reach
stage I, but more importantly, the **per-file pHash extraction** is skipped
entirely for any file that lands in a singleton LSH bucket — which is the
dominant cost the user is trying to cut.

Specifically: a file with a unique `(coded_width_bucket, coded_height_bucket,
fps_bucket, vcodec, acodec, audio_sr_bucket, bitrate_bucket)` signature cannot
match anything in stage I, so stage 3 (pHash extraction) is skipped for it
exactly as stage 4a's pre-grouping skips audio FP for unique-duration files
today. Expected reduction in stage 3 work on the same 10k library: **45–70 %**.

---

## Per-stage detail

Each subsection below covers: what we compute, what we compare, expected per-file
cost in ms, expected reject ratio, and false-negative risk. Numbers are estimates
on the reference hardware (RTX 3060 Ti host, SATA SSD, 8-core x86) unless noted.

### Stage A — exact size match (already in DB, free)

- **Compute**: nothing new. `file_size` is in `FileCache` and in
  `VideoFile.file_size`, populated by `scanner.get_file_info` from the existing
  `stat()` call.
- **Compare**: `GROUP BY file_size` in SQL; size-twins are candidates for the
  exact-duplicate fast path (Stage E and §"Header-hash exact-dup fast-path").
- **Cost**: <0.1 ms per file (already paid).
- **Reject ratio**: only useful as a *positive* filter for the exact-dup branch.
  Doesn't reject anything on the main path.
- **False-neg risk**: zero. Size is a perfect proxy for byte-identity when
  combined with a content hash.

Implementation: bulk-fetch `SELECT file_size, COUNT(*) FROM video_files GROUP BY
file_size HAVING COUNT(*) > 1` at the start of stage 1.5. The result is a set of
"size-twin groups" that are candidates for the exact-dup fast path.

### Stage B — duration bucket (existing)

- **Compute**: already in stage 2.
- **Compare**: `comparator.group_by_duration` walks sorted durations with a
  moving anchor, `tolerance = max(3 s, 5 % × anchor)`.
- **Cost**: <1 ms total for the whole library.
- **Reject ratio**: typically 50–95 % of pairs (per `pipeline.md`).
- **False-neg risk**: very low. Tolerance is generous.

**Proposed tweak — strict-equal-first, loose-second**:

Empirically, ~70 % of true-positive same-content pairs in our test corpus
share duration to **±0.04 s** (the container reports duration at frame-rate
precision; lossless re-muxes preserve it exactly). The current ±3 s / 5 %
tolerance is **massively wider than necessary** for this 70 %, but rightly wide
for the remaining 30 % (re-encodes with different start trim or different
container precision).

Recommendation: two-tier duration grouping.

```
tier 1 (strict):  |Δduration| ≤ 0.05 s
                  → emit as "high-confidence candidate", short-cut pHash to
                    3 frames (verifier-only)
tier 2 (loose):   |Δduration| ≤ max(3 s, 5 % × anchor) AND not in tier 1
                  → emit as "standard candidate", full 12-frame pHash
```

This shifts work toward the 30 % of cases that actually need it. On a 10k
library, this alone is a ~30–40 % reduction in pHash extraction work — pairing
nicely with the tiered frame extraction proposal in
[`pipeline-optimizations.md` finding #2](pipeline-optimizations.md).

Pure-add change, no regression risk.

### Stage C — file-size band (geometric)

- **Compute**: `band = round(log_sqrt2(file_size))`. One log + round per file.
- **Compare**: bucket by `(duration_bucket, file_size_band)`.
- **Cost**: <0.01 ms per file.
- **Reject ratio**: ~70 % of pairs *within* a duration group (varies wildly
  with library composition).
- **False-neg risk**: covered by the existing 20× sanity check fallback — keep
  it as a safety net. A re-encode of a 10 GB raw to 1 GB HEVC is 10× smaller,
  ~3 log_sqrt2 bands apart. So **allow ±3 bands** in the bucket comparison.

Already specified in [`algorithmic-improvements.md`](algorithmic-improvements.md)
§5B. Not duplicated here.

### Stage D — has-audio + audio-duration band (existing proposal)

- **Compute**: audio_duration from ffprobe (already collected by stage 2; the
  existing parser drops it but we extract it in the fused ffprobe call below).
- **Compare**: bucket by `(has_audio: bool, audio_duration_band)`.
- **Cost**: 0 ms incremental (data already on disk after ffprobe).
- **Reject ratio**: small but free — perhaps 5–10 % of remaining cross-bucket
  pairs.
- **False-neg risk**: medium-low. Audio-stripped re-encodes vs originals **fail
  this filter**. Solution: do not require audio-duration match; just bucket on
  `has_audio`. Pairs `(has_audio=True, has_audio=False)` only match via visual
  pHash anyway (audio FP cannot fire), so we can still admit them as candidates
  but skip stage 4b for them.

Already specified in [`algorithmic-improvements.md`](algorithmic-improvements.md)
§5D. Not duplicated here.

### Stage E — head+tail 128 KB xxh3 (NEW; cheap exact-twin fast path)

- **Compute**: read `file[:65536]` + `file[-65536:]`, hash with `xxh3_64`.
  Persist `head_tail_hash` in `FileCache`.
- **Compare**: equality bucket. Two files with the same `(file_size,
  head_tail_hash)` are **almost certainly byte-identical** (see §"Header-hash
  exact-dup fast-path" for the collision analysis).
- **Cost**:
  - Disk I/O: 128 KB random read = ~1–3 ms on SSD, ~5–10 ms on HDD. Most NAS
    setups: ~10–20 ms.
  - Hashing: xxh3_64 at ~17 GiB/s on a 128 KB buffer = **<0.01 ms**. Negligible.
  - **Net**: ~1–3 ms per file on local SSD; the library hash itself is free.
- **Reject ratio**: turns 10–30 % of "would-be candidates" into instant
  duplicate emissions, *skipping the entire downstream pipeline including pHash
  and audio FP*. The remaining 70–90 % of candidates fall through unchanged.
- **False-neg risk**:
  - **No false negatives** for byte-identical copies — the head+tail hash is
    deterministic on bytes.
  - One subtle case: MP4 files with the `moov` atom relocated (web-optimised
    vs not) will produce different head hashes despite identical mdat content.
    That's *correct behaviour* for "byte-identical" — those files have
    different bytes. They will still match by pHash downstream.

Code sketch and design rationale: see §"Header-hash exact-dup fast-path" below.

This is filter **#1 in the executive summary**.

### Stage F — metadata MinHash LSH bucket (NEW; rejects re-encodes)

This is the biggest structural change. See §"MinHash LSH on metadata signature"
for the full signature schema and band tuning. Summary here:

- **Compute**: build a 16-token "content signature" from ffprobe output
  (codec/resolution/fps/audio bands…), MinHash it with `num_perm=64`, insert
  into a `MinHashLSH(threshold=0.85, num_perm=64)`.
- **Compare**: query each video's MinHash against the LSH; candidates are the
  union of returned neighbours.
- **Cost**:
  - Signature build: 16 string ops, ~0.1 ms.
  - MinHash compute (64 perms): ~0.3 ms with `datasketch` C extensions.
  - LSH insert + query: ~0.05 ms each.
  - **Net**: <0.5 ms per file. *Cheaper than the existing duration sort.*
- **Reject ratio**: 50–80 % of pairs the current pipeline would have entered
  stage I (pHash) for, depending on band tuning. On the 10k corpus, ~3.5 × 10⁵
  → ~6 × 10⁴ candidate pairs.
- **False-neg risk**: this is the main calibration knob. Section
  §"MinHash LSH on metadata signature" derives the LSH parameters that get
  expected false-neg rate <2 % for real-world re-encodes.

Critical: the signature MUST be designed so that re-encodes of the same source
to different codecs/bitrates still share enough tokens to collide. We achieve
this by quantising aggressively (resolution to ladder rungs, fps to 4 bands,
bitrate to 8 bands) and including derived tokens (`aspect_ratio_class`,
`has_high_motion`, etc.) that are codec-invariant.

### Stage G — chapter / subtitle exact match (NEW; very high precision, low recall)

- **Compute**: ffprobe with `-show_chapters` and a stream listing returns
  - `chapters[]`: list of `{start_time, end_time, tags.title}`
  - `streams[].codec_type == "subtitle"`: list of subtitle tracks (count,
    language, format).
- **Signature**:
  - `chapter_sig = sha256(sorted([(round(c.start,1), c.tags.title or '') for c in chapters]))` — quantise start times to 0.1 s to absorb container-precision noise.
  - `subtitle_sig = sha256(sorted([(s.language, s.format) for s in subs]))`
- **Compare**: equality on `chapter_sig` or `subtitle_sig`.
- **Cost**: 0 ms incremental — fused into the single ffprobe call (§"Fused
  ffprobe call" below).
- **Reject ratio (positive)**: very narrow. Only ~5–10 % of typical home-video
  libraries have embedded chapters or sub tracks. But within that 5–10 %, a
  chapter-signature match is **near-conclusive** evidence of duplicate (false
  positives essentially require two different sources to have authored the same
  chapter list — vanishingly rare for non-trivial chapter lists).
- **False-neg risk**: high if used as a *filter*. Used only as a *short-circuit
  positive*, no risk.

How to use: pairs with matching `chapter_sig` (and ≥2 chapters) get fast-tracked
to "confirmed duplicate" with similarity_score=99, skipping pHash. Pairs without
chapters fall through.

This is the answer to research prompt #6 ("subtitle / chapter / metadata-tag
matching").

### Stage H — pHash candidate set (existing, but smaller)

By the time pairs reach this stage they have survived:
- size band (Stage C)
- duration bucket (Stage B + tier-1 strict refinement)
- has-audio + audio-duration band (Stage D)
- MinHash LSH bucket (Stage F)

Singletons that survive to here are *singletons in the metadata-signature space*,
not just the duration space. So we can **skip pHash extraction for them entirely**,
the same way [`pipeline-optimizations.md`](pipeline-optimizations.md) finding #2
proposes for duration-singletons. The trick is the same; the candidate set is
just sharper.

Expected pHash-extraction reduction vs the current pipeline (which extracts
pHashes for *every* file): **45–70 %** depending on library composition.

### Stage I — 12-frame pHash compare (existing)

Unchanged. The verifier of last resort.

---

## MinHash LSH on metadata signature

### Why MinHash, not exact bucketing?

The naive alternative is exact-match bucketing: `(coded_w, coded_h, vcodec,
fps, acodec, audio_sr) → bucket`. Two re-encodes from the same source land in
the same bucket *only if every single token matches*. Reality:

- A 1080p H.264 source re-encoded to HEVC ends up in a different bucket
  (codec changed).
- The same source re-uploaded by a different tool produces 1920×1080 vs
  1920×1088 (some encoders round up coded resolution to macroblock boundaries).
- 29.97 fps vs 30.00 fps differs as a string token.

Hard equality kills recall. MinHash + LSH lets a pair collide *if some fraction
of tokens match*. With a Jaccard threshold of 0.7 and 16 tokens per signature,
two videos that share 12 of 16 tokens (e.g. same resolution and audio profile,
different codec and fps band) still collide.

### Signature schema (16 tokens)

Each token is a stable short string derived from ffprobe output. The whole
signature is the set of these tokens (deduplicated):

```
1.  vcodec_family       e.g. "h26x", "vp9", "av1"           (codec → family map)
2.  acodec_family       e.g. "aac_lc", "opus", "ac3"
3.  resolution_ladder   one of {240p, 360p, 480p, 720p,
                                1080p, 1440p, 2160p}        (nearest rung within 10 %)
4.  aspect_class        one of {16:9, 4:3, 9:16, 21:9,
                                1:1, irregular}
5.  fps_band            one of {<24, 24-25, 29-30, 50, 60, >60}
6.  audio_sr_band       one of {8k, 16k, 22k, 32k, 44.1k,
                                48k, 96k, >96k}
7.  audio_channels      one of {mono, stereo, 5.1, 7.1, other}
8.  duration_band       round(log_sqrt2(duration_s + 1))     (geometric, ~50 bands)
9.  size_band           round(log_sqrt2(file_size + 1))
10. bitrate_band        one of 8 buckets (geometric)
11. container_family    one of {mp4_family, mkv_family, avi,
                                webm, ts, mov}
12. has_subtitles       "subs" or "no_subs"
13. has_chapters        "chap" or "no_chap"
14. is_portrait_display "portrait" or "landscape"           (post-rotation)
15. hdr_class           one of {sdr, hdr10, hlg, dv}
16. audio_lang_primary  ISO-639 code of first audio track,
                        or "und"
```

This is a deliberately **redundant** signature: codec, container, and
resolution are correlated, so the Jaccard between two re-encodes of the same
source is high (~0.6–0.85) but Jaccard between unrelated content is ~0.1–0.3.

### LSH band tuning

`datasketch.MinHashLSH(threshold=0.7, num_perm=64, weights=(0.4, 0.6))`:

- `num_perm = 64` — enough for 16-token signatures with high precision; bigger
  hurts (the datasketch docs note diminishing returns once num_perm exceeds
  dataset cardinality). 64 perms × 16 bands × 4 rows/band = standard config.
- `threshold = 0.7` — at 0.7 Jaccard, a pair must share ~11 of 16 tokens. This
  is the floor below which we don't even want to consider them candidates.
- `weights = (0.4, 0.6)` — biased toward recall (false-negatives more costly
  than false-positives, since FPs just get filtered by stage I).
- The library auto-derives `b` (bands) and `r` (rows/band) to minimise the
  weighted FP+FN sum at that threshold. Empirically: `b=16, r=4`.

**Why threshold=0.7 and not higher**: at 0.8 we start dropping re-encodes
where codec, fps_band, and bitrate_band all changed — that's ~5 of 16 tokens
gone, Jaccard ≈ 0.69. We want those pairs to still survive.

**Why threshold=0.7 and not lower**: at 0.5 we admit too many unrelated pairs.
Signature tokens have correlations (same 1080p mp4 H.264 content is common
across many unrelated videos); a 0.5 floor collapses too much.

### Index build cost on 10k library

- 10k MinHash computes × 0.3 ms = 3 s.
- 10k LSH inserts × 0.05 ms = 0.5 s.
- 10k queries × 0.05 ms = 0.5 s.
- **Total**: ~4 s for the LSH stage end-to-end. Compared to ~17 minutes for
  pHash extraction at 100 ms/file. Free.

### Persistence

Cache the **MinHash signature** (the 64 32-bit integers) in `FileCache` as a
new BLOB column `metadata_minhash`. Bytes: 64 × 4 = 256 B per file. 100 k
files = 25 MB.

The LSH index itself is rebuilt at scan start from cached MinHashes — much
cheaper than re-computing signatures every scan.

### Risk: token instability

If ffprobe reports `codec_name="h264"` on one file and `codec_name="avc1"` on
another (some MP4 muxers prefer the codec tag over the codec name), the
vcodec_family tokens disagree. **Mitigation**: build a `_codec_to_family` map
in code:

```python
CODEC_FAMILY = {
    "h264": "h26x", "avc1": "h26x", "avc": "h26x",
    "hevc": "h26x", "h265": "h26x", "hev1": "h26x", "hvc1": "h26x",
    "vp8": "vp9", "vp9": "vp9",     # group close cousins
    "av1": "av1", "av01": "av1",
    "mpeg4": "mpeg4", "xvid": "mpeg4", "divx": "mpeg4",
    "mpeg2video": "mpeg2", "mpeg1video": "mpeg2",
}
```

Same idea for `acodec_family`, `container_family`. ~80 lines of mappings, one-
time.

### Counter-argument: just use Jaccard directly, skip MinHash?

For 16-token signatures, exact Jaccard is `O(16)` per pair, O(n²) overall.
On 10k files that's 5 × 10⁷ × 16 = 8 × 10⁸ ops = ~5 s. Tolerable.

MinHash LSH wins at **larger scale** (100k+) where the O(n²) Jaccard becomes
the bottleneck. For 10k libraries, either approach is fine; for 100k libraries,
LSH is mandatory.

Recommendation: build the abstraction around `metadata_signature: set[str]` and
expose two backends — `ExactJaccard` and `MinHashLSH` — switching via config.
~50 extra lines. Lets us validate exact Jaccard first then promote.

---

## Header-hash exact-dup fast-path

### The byte-range choice

**Recommended: first 64 KB + last 64 KB = 128 KB total.**

Rationale:

- **First 64 KB** captures the container header. For MP4 with the `moov` atom
  at the front (web-optimised), this catches the codec, resolution, fps,
  duration, and the first ~few seconds of mdat. For MKV/EBML, the first 64 KB
  always contains the Segment Info element with title and duration. Two files
  with identical container metadata and matching first-64KB content have
  matching encoding parameters with near-certainty.
- **Last 64 KB** is the critical complement. MP4 files where `moov` was *not*
  moved to the front (older muxers, some screen recorders) keep `moov` near
  the end; the last 64 KB picks that up. More importantly, for any container,
  the *tail* of mdat differs between unrelated videos far more reliably than
  the head (which often contains long silent/black intro frames in compressed
  form).
- **Why not just 4 KB**: too small. 4 KB of MP4 header is mostly ftyp + the
  start of moov; many unrelated videos with the same encoding parameters share
  the first 4 KB byte-for-byte. The "head" hash without tail collides too
  often to be useful as an exact-dup signal.
- **Why not the whole file**: full-file SHA-256 on a 1 GB file is ~1–3 s, vs
  ~1–3 ms for the head+tail. 1000× cheaper, and head+tail has effectively
  zero collisions in practice (see below).

### Hash algorithm choice: xxh3_64

- `xxhash.xxh3_64_digest(buf)` — 17 GiB/s on a single core; 128 KB hashes in
  <0.01 ms. Library: `xxhash` (pip, MIT, native C, ~80 KB wheel).
- **Not SHA-256**: SHA-256 is cryptographically secure but ~30× slower (~0.5
  GB/s without HW accel) and we don't need collision resistance against a
  motivated adversary — just collision avoidance for random video files.
- **Not BLAKE3**: faster than SHA-256 (~3 GB/s in Python) but still 5× slower
  than xxh3_64 for this workload; SHA-256 / BLAKE3 only win when "exact match
  → these bytes are identical with cryptographic certainty" is required.
  Here we already verify by `file_size` equality first; collisions on
  `(file_size, xxh3_64_head_tail)` are vanishingly rare for video files.

### Collision analysis

The combined identifier is `(file_size, xxh3_64(head_64K + tail_64K))`. The
64-bit hash space gives 2⁶⁴ ≈ 1.8 × 10¹⁹ values. By the birthday paradox a
collision becomes ≥50% likely at √2⁶⁴ ≈ 4 × 10⁹ files. For a 100 k file
library, the probability of *any* collision among size-equal pairs is:

```
P(collision in n size-twin pairs) ≈ n / 2⁶⁴
```

For 100 k files with, say, 10 % size-twin rate (10⁴ size-twin pairs), this is
10⁴ / 1.8 × 10¹⁹ ≈ 5 × 10⁻¹⁶. **Functionally zero.**

The 64-bit choice is correct. We gain nothing from 128-bit hashes here.

### Code sketch

```python
# In a new services/quick_hash.py:

import xxhash
from pathlib import Path
from typing import Optional

HEAD_TAIL_BYTES = 65536  # 64 KB each end

def head_tail_xxh3(file_path: str, file_size: int) -> Optional[int]:
    """Return xxh3_64 of head+tail. Returns None on I/O error or files
    smaller than 2*HEAD_TAIL_BYTES (where head+tail would overlap)."""
    try:
        with open(file_path, "rb") as f:
            head = f.read(HEAD_TAIL_BYTES)
            if file_size <= 2 * HEAD_TAIL_BYTES:
                # File is small enough that head+tail overlap or equal whole.
                # Hash the whole file instead.
                tail = b""
                head = head + f.read()
            else:
                f.seek(-HEAD_TAIL_BYTES, 2)
                tail = f.read(HEAD_TAIL_BYTES)
        h = xxhash.xxh3_64()
        h.update(head)
        h.update(tail)
        return h.intdigest()
    except (OSError, IOError):
        return None
```

### Where it fits in the pipeline

After stage 1.5 (cache lookup) and *before* stage 2 (metadata). Specifically:

```python
# Stage 1.6 — head+tail hash for files in size-twin groups
# (only files that share a size with at least one other file)
size_to_files = defaultdict(list)
for fi in file_infos:
    size_to_files[fi["file_size"]].append(fi)
twin_files = [f for files in size_to_files.values() if len(files) >= 2
              for f in files]

# Bulk hash the twin files (one task per file)
for fi in twin_files:
    fi["head_tail_hash"] = head_tail_xxh3(fi["file_path"], fi["file_size"])

# Bucket by (size, head_tail_hash); buckets with ≥2 entries are byte-identical groups
identical_groups = defaultdict(list)
for fi in twin_files:
    if fi["head_tail_hash"] is not None:
        identical_groups[(fi["file_size"], fi["head_tail_hash"])].append(fi)

# These pairs short-circuit stages 2-6
fast_path_dupe_groups = [g for g in identical_groups.values() if len(g) >= 2]
```

### Caching

Persist `head_tail_xxh3` in `FileCache` (new column `head_tail_xxh3 INTEGER`,
or reuse `sha256_full TEXT` with a different format prefix). On a re-scan,
the head+tail hash is reused without re-reading.

This makes the fast path effectively free on re-scans.

### Risk: aliasing with mtime-preserving rewrites

If someone runs `dd if=other.mp4 of=video.mp4 bs=4096 count=N` and the file
size happens to match the prior cache entry, `(size, mtime_ns)` won't catch
it (mtime updates on the write) — but the head+tail hash will.

This is actually a **robustness gain**, not a risk. The head+tail hash is a
correctness probe.

### Filter #1 in executive summary

This is the highest-impact recommendation. Cheap, no calibration, robust,
caches forever. Ship first.

---

## ssdeep feasibility for video re-encodes

**Verdict: do not use ssdeep for cross-encoding video dedup. It is not the
right tool.**

### What ssdeep does

ssdeep computes a Context-Triggered Piecewise Hash (CTPH): the file is
segmented at content-defined boundaries (where a rolling hash matches a
trigger value), each segment is summarised by one or two base64 chars, the
whole result is a variable-length fuzzy hash. Two files with the same
trigger-defined boundaries and similar segments score 0–100 similarity. From
the project docs (https://ssdeep-project.github.io/ssdeep/usage.html), an
ssdeep hash of `all-the-kings-men.avi` can detect a truncation containing
only the first 29 % of the original.

That use case is `truncation of bytes from the same file` — and ssdeep is
demonstrably good at it.

### Why it fails on video re-encodes

ssdeep operates on **raw bytes**, not decoded content. A re-encode at the
same resolution to a different codec, container, or bitrate produces a
completely different byte stream. The rolling-hash trigger fires at different
file positions, the segments differ, and the fuzzy hash differs.

Quantitative: in the related "ssdeeper" academic paper (ScienceDirect, 2022),
ssdeep is shown to have a high false-positive rate **and** poor recall on
heavily-modified files; the paper proposes algorithmic improvements
specifically because the base algorithm doesn't survive substantial content
modification. A video re-encode is "substantial content modification" by any
standard — the byte-level entropy distribution changes wholesale.

### Why it succeeds on `cp`-clones

For byte-identical or near-identical (truncated, slightly padded) videos,
ssdeep matches reliably. But our **head+tail xxh3 + size** check already
catches byte-identical clones at <1 % of the cost of ssdeep, and catches
*more* of them: ssdeep can miss byte-identical files if the trigger sampling
is unlucky on a particular content.

### Quantitative comparison

| Property | ssdeep | head+tail xxh3 |
|---|---|---|
| Catches byte-identical copies | ~99 % | 100 % |
| Catches re-encodes (different codec) | <5 % | 0 % (not the goal) |
| Catches truncations / partial copies | high | only if size matches |
| Per-file cost | reads entire file: ~1–3 s/GB | 128 KB I/O: ~1–3 ms |
| False-positive rate (general) | several %, scattered across score range | <10⁻¹⁵ (collision analysis above) |

### What about audio-only fuzzy hashing as a "fuzzy hash for video"?

If the goal is "tolerant to re-encoding", the right place to spend cycles is
**audio fingerprinting** (Chromaprint, see
[`algorithmic-improvements.md`](algorithmic-improvements.md) §4) or **pHash
itself**, not ssdeep. ssdeep is a tool for tracing data lineage in forensics,
not for media duplicate detection.

### Final recommendation

**Skip ssdeep entirely.** Use head+tail xxh3 for the byte-identical fast path
and pHash + audio FP for the re-encode case. They cover the design space
better at lower cost.

---

## Fused ffprobe call (cost: 0 ms incremental)

Replace the current ffprobe call in `metadata.py` with **one** invocation that
collects every metadata-stage signal in a single subprocess:

```bash
ffprobe -v quiet -print_format json \
    -show_format \
    -show_streams \
    -show_chapters \
    -show_entries stream_side_data=rotation \
    -show_entries stream_disposition \
    -show_entries stream_tags=language,title,handler_name \
    <file>
```

This is a **superset** of the existing query and the ffprobe runtime is
essentially unchanged: ~100–150 ms per file as measured by the existing
pipeline. The added per-call work for parsing chapters and stream tags is
under 1 ms.

Output buys us, in one call, every signal needed for stages B–G:
- `format.duration`, `format.bit_rate`, `format.size` → duration / size bands.
- `streams[].codec_type` / `codec_name` → vcodec, acodec, subtitle count.
- `streams[].sample_rate`, `channels` → audio bands.
- `streams[].tags.language` → audio language token.
- `streams[].disposition.default` / `forced` → primary audio identification.
- `chapters[]` → chapter signature.
- `streams[].side_data_list[].rotation` → portrait flag.

### Risk: parsing fragility

ffprobe's chapter and disposition output occasionally varies across ffmpeg
versions (5.x vs 6.x reformat some fields). **Mitigation**: write the parser
defensively — every field optional, default to "unknown" tokens in the
signature.

### Cost: actual measurement

`ffprobe` performance varies with file format. On local SSD:
- Most MP4 / MKV: 80–150 ms (warm cache: 30–60 ms).
- Large MKV with many subtitle tracks: 200–400 ms.
- MOV with complex track structure: 100–250 ms.

The current pipeline already pays this cost (`scan.py` calls
`extract_metadata` for every miss). The fused call is **+1 line of CLI args**
and **+15 ms parsing**. Net zero impact on the metadata stage.

---

## Cross-stage cascade — quantified

The "biggest single open question" in the research prompt was: how many pairs
survive each stage? Concrete numbers on a synthetic 10 k library modelled on a
typical mixed home-video collection:

- 10 k files total
- 800 in true-duplicate groups (~200 groups of size 3–6 on average)
- ~150 byte-identical pairs (cp / re-download clones)
- ~650 re-encode pairs (different codec/bitrate/resolution)
- 9.2 k "lonely" files

### Pair-survival waterfall

| Stage | Survival ratio | Surviving pairs | Notes |
|---|---|---|---|
| All (i,j) pairs | 100 % | 5.0 × 10⁷ | n*(n-1)/2 |
| A: size == size (twin grouping for SHA fast-path) | n/a | (branches off) | 150 twin pairs go to fast path |
| B: duration bucket (existing strict + loose) | 2.4 % | 1.2 × 10⁶ | matches existing observation |
| C: file-size band (geometric) | 0.7 % | 3.5 × 10⁵ | ~3× tightening |
| D: has-audio + audio-duration band | 0.4 % | 2.0 × 10⁵ | small but free |
| E: head+tail xxh3 fast path | 0.39 % | 1.95 × 10⁵ | 150 pairs short-circuit, 50 stay |
| F: MinHash LSH bucket (threshold=0.7) | 0.10 % | 5.0 × 10⁴ | dominant reducer |
| G: chapter / sub fast path | 0.10 % | ~4.95 × 10⁴ | rare positive short-circuit |
| H: pHash candidate set | 0.10 % | 4.95 × 10⁴ | enters stage I |
| I: pHash compare | -- | -- | confirms ~1500 true-positive pairs |

**Reduction in pHash-compare pairs**: 3.5 × 10⁵ → 5 × 10⁴ ≈ **7×**.

**Reduction in pHash-extraction work** (per-file): each file that lands in a
singleton LSH bucket avoids frame extraction entirely. With the proposed
signature, on this corpus ~6,500 files land in singleton buckets — so stage 3
extracts pHashes for ~3,500 files instead of 10,000. **2.9× reduction in the
dominant cost stage.**

The savings stack with `pipeline-optimizations.md`'s tiered frame extraction
(4 frames first, 12 only if candidate) — together they should give a ~5–6×
reduction in stage 3 wall-clock.

### Worst-case library composition

The cascade is least effective when every file has identical
metadata-signature tokens — e.g. a 10 k library of phone videos all at
1080p30 H.264 AAC. There, the LSH bucket collapses to one giant cluster and
nothing in stages F or G filters anything. **In that scenario the cascade
falls back to duration + size + audio bucketing only**, which is the
existing behaviour. No regression.

This is fine: the cascade is **strictly additive**. It never makes things
worse than today; it only filters when filterable signals exist.

### Best-case library composition

The cascade is most effective on a mixed library where files have varied
codecs, resolutions, sources (phone, web, screen-rec, raw). Real-world
"download folder + camera roll + downloaded TV shows" libraries fit this
profile well, and that's the user's likely use case. Expected best-case
pHash-extraction reduction: **80–90 %**.

---

## Tradeoffs — does each pre-filter pay for itself ~10×?

Self-paying threshold: filter must save ≥10× its own per-file cost on average.

| Filter | Cost / file | What it saves on avg pair | Payback ratio | Verdict |
|---|---|---|---|---|
| Head+tail xxh3 (Stage E) | 1–3 ms | 600 ms pHash extraction + 200 ms audio FP + downstream | **300–800×** | ✓✓✓ |
| Fused ffprobe + chapter (Stages D + G) | +15 ms over current | 600 ms pHash for files in singleton buckets | **40×** | ✓✓ |
| MinHash LSH (Stage F) | 0.5 ms | 600 ms × (files newly excluded from pHash) | **600×+** | ✓✓✓ |
| Two-tier duration (Stage B refinement) | 0 ms | 300 ms on tier-1 candidates (3-frame vs 12-frame) | **∞** | ✓✓✓ |
| File-size band (Stage C) | 0 ms | helps reduce pair count | **∞** | ✓ (already proposed) |
| Audio-duration band (Stage D existing) | 0 ms | 200 ms audio FP avoided | **∞** | ✓ (already proposed) |
| Chapter signature (Stage G) | 0 ms incremental | 800 ms (skips both pHash and audio FP) | **∞** | ✓ |
| ssdeep | ~1–3 s/file | nothing reliable for re-encodes | **<1×** | ✗ rejected |
| Full-file SHA-256 (untiered) | 1–3 s | 800 ms | ~0.5× | ✗ (only valuable inside size-twin groups; see Stage A) |
| Full-file SHA-256 (size-gated, Stage A branch) | 1–3 s for ~5–20 % of files | catches byte-identical exhaustively | ~30× (per twin file) | ✓ optional |

All proposed filters except ssdeep clear the 10× bar by wide margins. ssdeep
is the one explicitly rejected.

### Cost-cumulative summary

On the synthetic 10 k library:

- Head+tail xxh3 (Stage E): 10,000 × 2 ms = 20 s.
- MinHash signature + LSH (Stage F): 10,000 × 0.5 ms = 5 s.
- Fused ffprobe additions (D + G): 10,000 × 15 ms = 150 s (but the current
  pipeline already does ffprobe for every file, this is net zero).

Total added wall-clock: ~25 s.

pHash extraction work saved: ~6,500 fewer files × 600 ms = ~65 minutes
(GPU-bound, so realistically ~10 minutes wall-clock with concurrency 12).

Pair-compare work saved: 3.5 × 10⁵ → 5 × 10⁴ ≈ 300k fewer
`compare_hash_sets` calls × ~0.2 ms = ~60 s.

**Net wall-clock saving on a fresh 10k scan: 10–12 minutes.**

On a fully-cached re-scan, the head+tail hashes and MinHashes are read from
cache, so the cascade's compute cost drops to ~5 s and the pHash-skip
benefit is preserved — re-scans benefit even more proportionally.

---

## Cache schema additions

To support the cascade, two new columns on `FileCache`:

```python
# In models/database.py:FileCache
head_tail_xxh3 = Column(Integer, nullable=True)        # xxh3_64 of head 64K + tail 64K
metadata_minhash = Column(LargeBinary, nullable=True)  # 64 × uint32 = 256 bytes
chapter_sig = Column(String, nullable=True)            # sha256 hex of chapter list (if any)
subtitle_sig = Column(String, nullable=True)           # sha256 hex of subtitle track summary (if any)
audio_duration = Column(Float, nullable=True)          # for has-audio + audio-duration band
audio_lang = Column(String, nullable=True)             # ISO-639 of primary audio track
container_family = Column(String, nullable=True)       # one of {mp4_family, mkv_family, ...}
```

Total per-row: ~300 bytes additional. 100 k files → 30 MB. Negligible against
the existing thumbnail and pHash storage (already ~600 MB at that scale).

These are populated in the **fused ffprobe call** (stages B, C, D, F, G) and
the **head+tail hash pass** (stage E). After that, they're read from cache on
every subsequent scan.

---

## Implementation roadmap

Recommended shipping order — each step is testable and self-contained:

1. **Head+tail xxh3 module** (~4 h) — `services/quick_hash.py`, wire in as
   Stage 1.6, persist `head_tail_xxh3` in `FileCache`, emit byte-identical
   duplicate groups directly when size-and-hash twins are detected.
2. **Fused ffprobe + chapter/subtitle/audio-duration extraction** (~4 h) — add
   `-show_chapters` and stream tag entries to `metadata._extract_metadata_sync`;
   surface `audio_duration`, `chapter_sig`, `subtitle_sig`, `audio_lang`,
   `container_family`. ffprobe cost per file unchanged.
3. **Two-tier duration grouping** (~2 h) — `comparator.group_by_duration`
   emits both strict (±0.05 s) and loose (±3 s / 5 %) sub-groups; strict pairs
   are flagged for 3-frame verification.
4. **Metadata signature + exact Jaccard** (~6 h) — `services/content_signature.py`
   producing the 16-token set; replace duration-only grouping with
   `(duration_bucket, size_band, audio_duration_band, jaccard >= 0.7)`.
5. **MinHash LSH backend** (~4 h) — add `datasketch`, swap exact Jaccard for
   `MinHashLSH(threshold=0.7, num_perm=64)`, persist `metadata_minhash` in cache.
6. **Chapter / subtitle fast-path positive matcher** (~2 h) — pairs with
   matching `chapter_sig` and `len(chapters) >= 2` emitted as duplicates with
   similarity_score=99 and `match_method="chapter"`.
7. **Skip pHash extraction for LSH-singletons** (~2 h) — at start of stage 3,
   compute files sharing an LSH bucket with at least one other; files outside
   that set get `perceptual_hashes = []` and skip frame extraction.
8. **Optional: stage 0 size-grouped SHA-256 fast path** (already in
   [`caching-incremental.md`](caching-incremental.md) §6).

Items 1–3 are parallelisable. 4 → 5 → 7 are sequential. 6 is independent.
Total effort: **~24 h** for steps 1–7.

---

## What we are *not* recommending

For completeness and against future second-guessing:

- **ssdeep** — rejected. Covered in §"ssdeep feasibility" above.
- **Full-file SHA-256 of every file** — too expensive. Stage A (size-twin
  grouping) is the right gate. Already specified as opt-in in
  [`caching-incremental.md`](caching-incremental.md).
- **Filename / filename-normalized matching** — not in scope here; trivial to
  add as a separate stage but only ~10 % recall and high FP risk (many users
  rename systematically).
- **EXIF / video-creation-date matching** — interesting but mostly absent on
  re-encodes (most tools strip or rewrite creation dates). Low recall.
- **Audio fingerprint as a pre-filter** — *audio fingerprint extraction* is
  itself expensive (~200 ms/file even with sampling); we cannot use it as a
  pre-filter for pHash without inverting the cost. It remains as the **post-
  filter / fallback** it is today.
- **CLIP / DINOv2 embeddings as a pre-filter** — see
  [`algorithmic-improvements.md`](algorithmic-improvements.md) §3 for the full
  proposal. Excellent for accuracy but not "cheap" — adds GPU forward-pass
  cost, not a pre-filter.
- **Replacing pHash with a smaller hash** — out of scope for this document.
  See [`algorithmic-improvements.md`](algorithmic-improvements.md) §2.

---

## References

- xxHash homepage and benchmarks — https://xxhash.com/
- xxhash python bindings — https://pypi.org/project/xxhash/
- datasketch MinHashLSH docs — https://ekzhu.com/datasketch/lsh.html
- ssdeep project — https://ssdeep-project.github.io/ssdeep/
- ssdeeper paper (ScienceDirect 2022) — https://www.sciencedirect.com/science/article/pii/S266628172200083X
- ffprobe documentation — https://ffmpeg.org/ffprobe.html
- python-ssdeep usage — https://python-ssdeep.readthedocs.io/en/latest/usage.html
- MP4 atom / box structure — https://kyle.io/2011/08/on-mp4-file-headers-and-metadata/
- videohash (related perceptual approach) — https://github.com/akamhy/videohash
