# Research Index — Faster & Better Duplicate-Video Detection

This folder contains 14 research notes on making `run_scan_pipeline()` faster and
more accurate. The three originals (`algorithmic-improvements`,
`caching-incremental`, `pipeline-optimizations`) cover the baseline wins
(BK-tree, Chromaprint, content bucketing, `file_cache` table, surgical pipeline
fixes). The 11 newer notes go deeper or sideways.

This file is the **starting point**: it indexes every doc and proposes a single
integrated adoption roadmap that resolves overlaps and contradictions across the
notes.

---

## Per-document index

### Originals (already integrated as references)

| File | One-liner |
|---|---|
| [`algorithmic-improvements.md`](algorithmic-improvements.md) | BK-tree replaces O(n²) loop; Chromaprint replaces RMS; content-bucket before pHash |
| [`caching-incremental.md`](caching-incremental.md) | `(path,size,mtime)`-keyed `file_cache` persists pHash/audio-FP/metadata across scans |
| [`pipeline-optimizations.md`](pipeline-optimizations.md) | 11 surgical fixes — sample audio, tiered extraction, separate GPU/CPU semaphores, etc. |

### Round-2 deep-dives (this batch)

| File | Recommendation | Headline number |
|---|---|---|
| [`quick-rejection-strategies.md`](quick-rejection-strategies.md) | xxh3 head+tail + fused ffprobe + 16-token MinHash-LSH cascade before pHash | **45-70% of pHash extractions avoided** |
| [`codec-aware-shortcuts.md`](codec-aware-shortcuts.md) | Byte-sample xxhash + per-packet frame-size signature pre-pHash | **80-95% pHash avoided, 3-4× end-to-end** |
| [`frame-sampling-strategies.md`](frame-sampling-strategies.md) | I-frame seeking (`-skip_frame nokey`) inside a `blackdetect`-trimmed window, N scaled log(duration) | **40-50% reduction in stage 3** |
| [`gpu-acceleration-deep-dive.md`](gpu-acceleration-deep-dive.md) | NVDEC-to-tensor pipeline (TorchCodec) + GPU-side pHash (cuPy DCT) | **~5× per-file (485 ms → 93 ms)** |
| [`parallelism-patterns.md`](parallelism-patterns.md) | Streaming pipeline with bounded `asyncio.Queue`s, `TaskGroup`, separate GPU/CPU/audio worker pools | **25-50% wall-clock reduction** |
| [`ann-indexing-structures.md`](ann-indexing-structures.md) | `faiss.IndexBinaryFlat` per duration bucket + 12×12 verifier | **20-60× speedup of compare stage at 50k** |
| [`single-video-hashing.md`](single-video-hashing.md) | Median-bit aggregation of the 12 pHashes → 1 prescreen hash; existing 12-hash compare kept as verifier | **5-210× faster comparison (n=50…5000)** |
| [`audio-fingerprint-alternatives.md`](audio-fingerprint-alternatives.md) | Multi-segment Chromaprint (3×10s windows, 2-of-3 vote) + spectrogram-pHash fallback for short clips | **10-20× discriminative power vs RMS** |
| [`deep-learning-embeddings.md`](deep-learning-embeddings.md) | Add SSCD (ResNet-50, 512-d) as a verifier between pHash and audio-fallback, NOT as a replacement | **+8-12 pp recall** on cropped/watermarked re-encodes |
| [`perceptual-hash-comparison.md`](perceptual-hash-comparison.md) | Stay on pHash (codebase already uses 256-bit); optional dHash second-channel ensemble | **No throughput change; +3-8% recall** |
| [`production-systems-survey.md`](production-systems-survey.md) | Borrow Meta's PDQ (drop-in for `imagehash.phash`) + VideoDuplicateFinder's temporal-delta channel | **Calibrated threshold + +recall on heavy compression** |

---

## Cross-cutting consensus

Several recommendations show up across multiple notes — those are the safest
bets:

- **Frame-size signature as a cheap pre-filter** (proposed independently in
  `quick-rejection-strategies.md` and `codec-aware-shortcuts.md`).
- **Hybrid "single aggregate hash for screening + existing 12-hash verifier"**
  (proposed in `ann-indexing-structures.md` and `single-video-hashing.md`).
  These two converge on the same architecture from different angles — index by
  one hash, verify with twelve.
- **Streaming pipeline + worker pools with separate GPU/CPU semaphores**
  (proposed in `pipeline-optimizations.md` (#3/#4) and elaborated in
  `parallelism-patterns.md`).
- **I-frame / blackdetect-trimmed extraction over uniform sampling** —
  endorsed by `frame-sampling-strategies.md` and presupposed by
  `codec-aware-shortcuts.md`.

## Notable contradictions

- **Hash algorithm**: `perceptual-hash-comparison.md` says "stay on pHash, it's
  already 256-bit"; `production-systems-survey.md` says "swap to Meta's PDQ
  for calibrated thresholds." Both are right — same shape, similar quality.
  PDQ buys you Meta's calibration tables; pHash buys you zero migration. The
  user's existing threshold (Hamming ≤ 14 at 256-bit) is already conservative,
  so the upside of PDQ is marginal unless the user wants Meta's published
  threshold guidance.
- **GPU pHash vs better sampling**: `gpu-acceleration-deep-dive.md` attacks
  stage 3 with a GPU rewrite (~5×); `frame-sampling-strategies.md` attacks
  the same stage with cheaper sampling (~2×). They **stack** — the GPU
  rewrite still has to decode the I-frames the sampler picked.
- **Add neural model or not**: `deep-learning-embeddings.md` says SSCD adds
  +8-12pp recall on cropped/watermarked cases; `perceptual-hash-comparison.md`
  says fixing the missing `cropdetect` ffmpeg filter would close most of that
  gap at near-zero cost. Add `cropdetect` first, measure, then decide on SSCD.

---

## Integrated adoption roadmap

Ordered by impact-per-effort, with dependencies called out. Quoted speedups
are the headline numbers from the individual docs — they **do not all stack
multiplicatively** because they attack overlapping costs.

### Tier 0 — already specified by the original 3 docs (do these first)

1. `file_cache` table (`caching-incremental.md`) — single biggest win on
   re-scans, no algorithmic risk. Everything below assumes this is in place.
2. ✅ **SHIPPED** — Sample audio instead of decoding full track
   (`pipeline-optimizations.md` finding #1). `audio_fingerprint.py` now decodes the middle 60 s. `AUDIO_FP_VERSION=2` invalidates old full-track FPs.
3. ✅ **SHIPPED** — Separate GPU vs CPU semaphores
   (`pipeline-optimizations.md` finding #4). `scan.py` uses `cpu_concurrent = min(MAX_CONCURRENT_FFMPEG, cpu_count())` for stage 4b.
4. ✅ **SHIPPED (variant)** — Tiered frame extraction. Not the proposed 4-then-12 — instead, skip pHash entirely for videos with unique durations (which can never match anyone). Same end effect for the dominant case.

### Tier 1 — cheap pre-filter cascade before pHash (huge funnel reduction)

5. ✅ **SHIPPED (variant)** — head+tail content hash of every file
   (`quick-rejection-strategies.md` finding 2, `codec-aware-shortcuts.md`
   finding 10). Implemented as `blake2b(8)` of first + last 64 KiB rather than xxh3_64 (no new dep; speed difference is sub-ms either way for 128 KiB). Cached as `FileCache.head_tail_xxh3`. **Drives the byte-identical fast-path** in `scan.py:2.5`.
6. ⏸ Deferred — Single fused ffprobe for metadata + chapters + frame-packet sizes. Current `metadata.py` is already one ffprobe call; the additions (chapters, packet sizes) need separate flags and a more invasive refactor.
7. ⏸ Deferred — MinHash LSH on the 16-token metadata signature. Needs `datasketch` and a refactor of the candidate gate.
8. ⏸ Deferred — Frame-size signature correlation. Needs `ffprobe -show_packets`.

After what shipped: byte-identical clusters are skipped entirely, and unique-duration videos skip pHash. The bigger funnel reduction (#6-#8) is still on the roadmap.

### Tier 2 — fix what pHash extraction does for the remaining files

9. ⏸ Deferred — I-frame seeking via `-skip_frame nokey` inside a `blackdetect`-trimmed window. **Partially addressed**: `hasher.py` now does per-frame `-ss` fast-seek for videos ≥ 180 s (the original `fps=N/duration` filter blew the timeout on long network HEVCs). True I-frame extraction with blackdetect trim is still future work.
10. ✅ **SHIPPED** — `cropdetect` via Python/numpy bbox (`hasher.py:_strip_letterbox`). `PHASH_VERSION=3` invalidates old non-cropdetect hashes. Closes the letterbox false-negative class.
11. ⏸ Deferred — GPU pHash via cuPy DCT + NVDEC-to-tensor. Big-bang change; current GPU usage is at the ffmpeg-subprocess layer only.

### Tier 3 — fix the matching loop now that the funnel is small

12. ✅ **SHIPPED** — `faiss.IndexBinaryFlat` per duration bucket + 12×12 verifier on the shortlist (`comparator.py:_faiss_phash_candidates`). Fires for groups ≥ 16 with cached aggregates; falls back to all-pairs otherwise. `faiss-cpu` is in `requirements.txt`.
13. ✅ **SHIPPED** — Median-bit (per-bit-majority) aggregate of the 12 pHashes (`hasher.py:compute_aggregate_hash`). Cached as `FileCache.aggregate_hash`. Used as the FAISS index key.

### Tier 4 — quality, not speed

14. ⏸ Deferred — Multi-segment Chromaprint. Needs `pyacoustid` + the `fpcalc` binary on PATH. The current RMS approach was made faster (middle 60 s) but not more discriminative.
15. ⏸ Deferred — dHash second channel. Trivial to add (`imagehash.dhash` is already pulled in), but adds a new field to cache.
16. ⏸ Deferred — SSCD verifier. Needs a labelled holdout for threshold tuning; deferred until cropdetect is proven to close most of the gap.

### Tier 5 — production-grade overhaul (only if Tier 1-4 isn't enough)

17. ⏸ Deferred — Streaming pipeline with `TaskGroup` + bounded `asyncio.Queue`s + per-resource worker pools. Significant rewrite; pause/stop semantics need rework.
18. ⏸ Deferred — Swap `imagehash.phash` for Meta's `pdqhash`. Marginal upside given existing 256-bit pHash with conservative threshold.

### Also shipped this round (not in the original tier list)

- ✅ **Per-scan in-memory error log + WebSocket broadcast + UI panel** (`services/error_log.py`, `api/scan.py`, `frontend/src/pages/Dashboard.tsx`). Per-file failures that used to go only to stdout now surface in the UI in real-time.
- ✅ **Auto-migration helper** (`models/database.py:_migrate_add_columns`). Idempotent `ALTER TABLE ADD COLUMN` so users adopting `head_tail_xxh3` / `aggregate_hash` don't need to delete their DB.
- ✅ **WebSocket-driven frontend scan-list patching** + completion race fix + polling cadence 3 s → 1.5 s + WS throttle 2 % → 0.5 %. Eliminates the "page looks stale until I refresh" UX bug.
- ✅ **Per-frame fast-seek extraction for videos ≥ 180 s** (`hasher.py:_extract_frames_seek_sync`). Fixes the 60-s timeout failure on long HEVC over SMB. Wall-clock now scales with `num_frames`, not `duration`.

---

## What was investigated and rejected (so it stays rejected)

- **TMK+PDQF** as the *primary* video hash — Linux-only build chain, 256 KB
  signatures, isomorphic to median-bit aggregation in practice
  (`single-video-hashing.md`).
- **DCT-coefficient hashing from the H.264 bitstream** — no production-grade
  Python library exposes post-CABAC transform coefficients in 2026; only
  research-grade tools (`codec-aware-shortcuts.md` finding 1).
- **ssdeep / context-triggered piecewise hashing** for video re-encodes —
  operates on raw bytes, fails across codec changes; xxh3 dominates it for
  the byte-identical case at 1000× lower cost
  (`quick-rejection-strategies.md`).
- **VideoMAE / X-CLIP / InternVideo** as a duplicate detector — wrong tool
  for the problem, 10-30× slower than SSCD with marginal recall gain
  (`deep-learning-embeddings.md`).
- **Dejavu, audfprint, akamhy/videohash** — abandoned or stagnant since
  2018-2022 (`audio-fingerprint-alternatives.md`, `single-video-hashing.md`).
- **hnswlib / nmslib** for Hamming — no upstream binary support; FAISS wins
  (`ann-indexing-structures.md`).
- **`HASH_SIMILARITY_THRESHOLD` ≈ 31 (Meta's PDQ recommendation)** — too
  loose for consumer "delete-button" UX; keep existing tight threshold even
  if you swap to PDQ (`production-systems-survey.md`).

---

## Next step (suggested)

Tier 0 and most of Tier 1 are low-risk and high-impact; together they
should compress a typical re-scan by an order of magnitude with no
algorithmic risk. Tier 2-3 require some experimentation; Tier 4-5 are
larger commitments.

A reasonable first PR scope: `file_cache` + xxh3 fast-path + fused ffprobe
+ MinHash LSH bucketing + sampled audio FP. That's the highest-impact
batch that can be delivered without touching the matching loop or the
GPU code.
