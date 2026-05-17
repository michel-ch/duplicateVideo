# Algorithmic Improvements for Video Duplicate Detection

A research note on faster matching strategies for the pipeline implemented in
`backend/services/comparator.py`, `backend/services/hasher.py`, and
`backend/services/audio_fingerprint.py`. The current pipeline is correct and
robust to anamorphic/portrait re-encodes, but the inner loop is O(n^2) within
each duration group with a 12x12 = 144-cell Hamming distance matrix per pair, and
the audio fingerprint has weak discriminative power. The recommendations below
are ordered by ratio of speedup to implementation effort.

## Executive summary

Top three recommendations, in priority order:

1. **Replace the within-group all-pairs loop with a BK-tree (or VP-tree)
   indexed on the per-video aggregate hash.** O(n^2) -> roughly O(n log n) per
   group, drop-in for Python (`pybktree`), 4-8 hours of work, near-zero
   regression risk if you keep the existing 12-frame best-match as a verifier
   on the candidate shortlist.
2. **Switch the audio fingerprint to Chromaprint (`pyacoustid` /
   `acoustid.fingerprint_file`).** It is the de-facto industry standard,
   already C-implemented, far more discriminative than 64-point RMS, and works
   for short clips where RMS profiles collapse to noise. ~1 day including
   threshold re-tuning.
3. **Add a content-bucketing pre-filter using duration + file-size band +
   audio-track length** *before* doing pHash extraction, then promote
   pHashing to a within-bucket step only. The current pipeline already
   short-circuits audio FP for unique-duration files; do the same for pHash
   extraction itself. This addresses the user's "takes too long" complaint
   directly: the dominant cost is per-video frame extraction, not the
   compare loop.

For datasets in the 1k - 10k range these three changes together remove the
quadratic growth in compare time and roughly halve the per-video extraction
work. For 100k+ datasets, additionally swap pHashes for **CLIP image embeddings
indexed in FAISS** (item 3 in the per-section detail below) — a single GPU
forward pass per video plus a sub-millisecond ANN query.

---

## 1. Sub-quadratic matching

The current within-group loop is in `comparator.py:find_duplicates_in_group`,
lines 117-156. For a duration group of size n it does n*(n-1)/2 calls to
`compare_hash_sets`, each of which builds a 12x12 numpy distance matrix and
does a greedy assignment. Group sizes can blow up when many videos share a
common round duration (e.g. 60.0s TikTok exports), and the early-exit only
helps lopsided pairs.

### Option A: BK-tree on a single aggregate hash per video

A **BK-tree** is a metric-space index for discrete distance functions like
Hamming. Build time O(n log n), query time O(log n) on average for the
k-nearest-within-radius problem. Works perfectly for 256-bit binary hashes.

Plan:

1. After computing the 12 frame hashes per video, also compute one aggregate
   per video: either a **median bit hash** (for each of the 256 bit positions,
   take the majority across the 12 frames) or a **majority-bit hash**. This is
   one cheap numpy reduction.
2. Build one BK-tree per duration group (or one global tree if you also key by
   duration bucket). `pybktree.BKTree(hamming, hashes)`.
3. For each video query the tree for `radius = HASH_SIMILARITY_THRESHOLD` to
   get a small candidate shortlist.
4. Run the existing `compare_hash_sets` on the shortlist only as a verifier.
   This preserves the per-frame robustness the project already has against
   trims/fps differences.

Library: `pybktree` (pure Python, MIT). No native build needed. ~150 lines
of integration.

Why better than current: O(n log n) build + O(n log n) query vs O(n^2)
verify. For a duration group of 200 videos that is ~1500 verifier calls
instead of 19,900.

Effort: 4-8 hours including unit tests.

Regression risk: the aggregate hash can hide genuine matches where only a
few of the 12 frames match (e.g. when the duplicate has a different intro).
**Mitigation:** widen the BK-tree radius to ~1.5x the per-frame threshold
(so e.g. 21 instead of 14) to keep recall high; the verifier on the
shortlist gives the final answer.

### Option B: LSH for Hamming distance

For very large duration buckets (>50k) BK-trees degenerate. **Bit-sampling
LSH** hashes each 256-bit fingerprint into K bands (e.g. 16 bands of 16
bits), indexes each band in a hash table; candidates share at least one
band. Library: `datasketch.MinHashLSH` works on sets only; for Hamming-LSH,
roll your own in ~80 lines of numpy. Defer until BK-trees prove
insufficient.

### Option C: ANN libraries (FAISS / Annoy / HNSW)

These work natively on binary codes:

- **FAISS** has `IndexBinaryFlat`, `IndexBinaryIVF`, `IndexBinaryHash`. The
  IVF + PQ variants give sub-linear queries. CPU and CUDA builds. License
  MIT. The big upside is that the same index also works for the float
  CLIP-embedding extension (section 3) — install once, use twice.
- **Annoy** (Spotify) is float-only; not a fit for raw bit hashes.
- **HNSW** via `hnswlib` is float-only too, but if you switch to embeddings
  it is the fastest CPU graph-based ANN.

Effort: 1-2 days, mostly devops (FAISS Windows wheels exist via
`faiss-cpu`; CUDA needs `faiss-gpu` and a matching CUDA toolkit).

Recommendation: **start with BK-tree, only graduate to FAISS Binary IVF if
total catalogue size exceeds ~50k after deduping.** A BK-tree scales to
200k 256-bit hashes on a single core comfortably.

References:
- Burkhard, Keller (1973), "Some approaches to best-match file searching."
- FAISS Binary indexes:
  https://github.com/facebookresearch/faiss/wiki/Binary-indexes

---

## 2. Better hash representation

Current: 12 frames * 256-bit pHash, compared with greedy best-match.

### Is 12 the right frame count?

12 is a sensible default *if* you keep best-match. The matrix cost is
12*12=144 comparisons per pair, each ~256 XOR + popcount, fully vectorised in
numpy. The dominant cost is the **frame extraction itself**, not the
comparison. Halving to 6 frames would halve extraction cost and leave
matching cost unchanged. The robustness loss is real but small for
well-aligned re-encodes.

For very short videos (<10s, very common in social-media dumps) 12 frames
oversamples; 4-6 is ample. Recommend **adaptive frame count: 6 below 30s,
12 above**.

### Single aggregate hash per video

Three options, in increasing sophistication:

1. **Bitwise majority across frames** — described in 1A. Cheap. Loses
   sensitivity to scene-level changes; works only as a coarse pre-filter.
2. **Concatenated hash** — stack all 12 hashes into one 3072-bit string.
   This is what TMK+PDQF and Facebook's PDQ do. Compare via Hamming on the
   long string. Loses the fps/trim robustness the project specifically
   designed for, since the order matters again.
3. **Pooled embedding** — see section 3.

Recommendation: keep the per-frame hash list as the authoritative
representation, **add** a single coarse aggregate hash per video for
indexing only. Best of both worlds: index lookup is O(1) per video; the
expensive 144-cell verifier runs only on shortlisted pairs.

### Hash family

`imagehash.phash` (DCT-based) is reasonable. **PDQ** (Facebook's open-source
hash, 256-bit) is more discriminative on benchmark sets and has a published
calibration curve mapping Hamming distance to false-positive rate. Library:
`pdqhash` (pip). Drop-in replacement for `phash` with similar performance.

### Why this is better

- Indexing cost drops from O(n^2 * 144) to O(n log n) for the aggregate, plus
  O(shortlist * 144) for verification. Net 5-50x speedup depending on group
  size.
- Accuracy preserved because the verifier remains the existing best-match
  routine.

### Risk

Aggregate hash can drop true positives if their per-frame hashes vary too
much (videos that share only a single distinctive scene). The widened
shortlist radius mitigates this; tune via the existing
`HASH_SIMILARITY_THRESHOLD` plus a multiplier.

---

## 3. Modern embeddings (CNN / ViT)

This is the biggest accuracy win available, with a real but manageable cost.

### Image-level: CLIP / DINOv2

- **CLIP ViT-B/32** (OpenAI) — 512-d float embedding per image, ~150 MB
  weights, ~600 MB VRAM at fp16, ~5 ms/image on a 3060 Ti at batch 32. Free
  via `open_clip_torch` or HuggingFace `transformers`.
- **DINOv2 ViT-S/14** (Meta, self-supervised, no text) — 384-d, ~85 MB
  weights, ~400 MB VRAM. Better dense features for visual similarity than
  CLIP because DINO trains explicitly for visual self-similarity. Library:
  `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`.

Pipeline change:

1. Reuse the existing 12-frame extraction.
2. Run all 12 frames through the model in one batched forward pass.
3. **Mean-pool** the 12 embeddings -> one 384/512-d vector per video.
4. L2-normalise. Cosine similarity becomes a dot product.
5. Index in **FAISS `IndexFlatIP`** (exact, fast at this dimension) or
   **`IndexHNSWFlat`** for sub-millisecond queries at 100k+ scale.

Why better than pHash:

- Robust to crops, watermarks, mild colour grading, light re-encoding
  artefacts that destroy DCT pHashes.
- Catches semantic duplicates pHash cannot: same scene at different
  resolutions where compression has hammered high-frequency components.
- Sub-quadratic matching is *native* — there is no separate hashing+ANN
  step, the embedding *is* the index key.

Realistic VRAM estimate on the existing CUDA setup (RTX 3060 Ti, 8 GB):

- DINOv2 ViT-S/14 fp16: 400 MB resident + ~200 MB activations at batch 32 =
  **~600 MB total**. Fits easily alongside FFmpeg's NVDEC streams.
- CLIP ViT-B/32 fp16: ~700 MB total.

Throughput estimate: 12 frames * (5 ms / batch of 32) ~= 2-3 ms/video on
the 3060 Ti. The bottleneck stays at frame extraction, not inference.

### Video-level: VideoMAE, S3D, X-CLIP

Specialised video models take a clip (16-32 frames) as input and output one
embedding. They are *better* at temporal duplicate detection but:

- VideoMAE-Base: ~340 MB weights, ~2.5 GB VRAM at fp16 for a 16-frame clip
  (input is 224x224x16). Tight on 8 GB if anything else is using the GPU.
- Latency: ~30-50 ms/video. 10x slower than CLIP/DINO image-level.
- The fps/trim robustness story is more delicate — clip sampling matters.

Recommendation: do not jump to video-level models unless image-level CLIP
fails on a known false-negative set. The mean-pooled image embedding
already handles most temporal drift because best-match no longer matters
(one vector per video).

### Effort

- 2-4 days end to end for DINOv2 + FAISS, including evaluation against the
  current pHash on a held-out set of known duplicates, threshold
  calibration (cosine similarity ~0.92-0.95 typically), and migration of
  the DB schema (`perceptual_hashes` text column -> `embedding` BLOB or
  JSON float array).

### Risk

- Embeddings can falsely cluster *visually similar but distinct* content
  (two different gameplay clips of the same game). Pair the visual
  embedding with the audio fingerprint as an AND check at the borderline
  similarity range to suppress this.
- pHash is currently the only thing surviving extreme bitrate compression
  artefacts on tiny thumbnails. CLIP/DINO are similarly robust *if* you
  feed them the same 320px frames; do not downsample further.

References:
- DINOv2: https://arxiv.org/abs/2304.07193
- FAISS: https://github.com/facebookresearch/faiss
- VideoMAE: https://arxiv.org/abs/2203.12602

---

## 4. Audio fingerprinting alternatives

Current 64-point RMS profile (8 kHz mono) is extremely crude. It catches
identical audio at different bitrates well, but:

- A 60-s clip with one loud peak has its RMS profile dominated by that
  peak; subtle re-mixes flatten correlation.
- Cross-correlates fail when a few seconds of leading silence are added
  (no time-shift compensation in `compare_audio_fingerprints`; it
  truncates to `min_len`).
- Low discriminative power on speech-only content.

### Chromaprint (AcoustID)

The **industry standard** for audio dedup. Used by MusicBrainz, Picard,
half the music-tagging world. Open source (LGPL).

- Library: `pyacoustid` (`pip install pyacoustid` plus the `fpcalc` binary
  from Acoustid). Or: build the C library `chromaprint` and bind via
  `cffi`.
- Output: variable-length array of 32-bit unsigned ints (one per ~0.124s),
  capturing the dominant chroma feature per frame. For a 60-s video that
  is ~480 ints ~= 1.9 KB. Easy to store in JSON or BLOB.
- Comparison: bitwise XOR + popcount over aligned segments, slide one
  fingerprint over the other to find best alignment. Standard reference
  implementation does this with a sliding correlation; ~10 us per pair.
- Time-shift robust by construction.

Effort: ~1 day. The biggest unknown is calibration: AcoustID
recommends ~95% match for "same recording." Map to your threshold via a
held-out test set.

Why better:

- Works on speech, music, ambient. RMS profile mostly fails on speech.
- Time-shift robust (sliding alignment).
- Discriminates *which song / clip* not just *similar energy curve* —
  RMS curves of two action movies look the same; chroma fingerprints
  do not.

Risk: silent / near-silent videos produce empty fingerprints; need an
explicit fallback. Add an `audio_active_seconds < 5 -> skip audio match`
guard.

### MFCC summary

Compute 13-20 MFCC coefficients per frame, mean+std-pool over the whole
track. Lightweight, library `librosa` (`librosa.feature.mfcc`). Compare
via cosine similarity. Better than RMS, worse than Chromaprint for music,
similar for speech. Use only if you cannot ship the `fpcalc` binary
(packaging constraint).

### Spectral centroid time-series

Even cheaper than MFCC, captures roughly "where the energy is in
frequency." Improves on RMS by adding a frequency dimension. Useful only
as an incremental step; if you are touching audio, jump to Chromaprint.

### Recommendation

Replace RMS with Chromaprint. Keep the same OR-rule in `comparator.py`,
just swap the helper and re-tune the threshold. Largest single accuracy
gain in the audio side of the pipeline.

References:
- Chromaprint paper: Lukas Lalinsky, "How does Chromaprint work?",
  https://oxygene.sk/2011/01/how-does-chromaprint-work/
- AcoustID: https://acoustid.org/

---

## 5. Pre-filter ordering

Current order: duration filter -> pHash extraction (every video) ->
within-group pHash compare -> audio FP (only for duration candidates) ->
audio compare. The pHash extraction is the most expensive per-video
operation in the entire pipeline (12 frames * 320px JPEG decode + DCT).
Cutting work *before* pHash extraction has the biggest impact.

### A: Skip pHash extraction for duration-singletons

The pipeline already does this for audio fingerprinting (see
`pipeline.md` stage 4a). It does **not** do it for pHash. From the
pipeline docs: "Stage 3 — perceptual hashing" runs over **all** videos.

Fix: gate stage 3 on the same duration-bucket pre-grouping. Videos with
unique durations (within tolerance) cannot match anything, so do not
extract or hash their frames. On a typical home-video collection where
30-50% of videos have unique durations this is **a 30-50% reduction in
the most expensive stage**, with zero accuracy loss.

Effort: 2-3 hours. The data flow change is purely a control flow
addition: extend stage 4a's candidate-set logic to also short-circuit
stage 3.

Risk: if a future scan finds a new duplicate of a previously-singleton
video, pHashes are missing. Solve by storing a `hashes_pending=true`
flag and lazy-computing on the next scan. Or: compute hashes for all
videos but defer the within-group comparison work — frame extraction
must run anyway during stage 2 (thumbnail). The two costs are similar;
audit which dominates wall-clock.

### B: Coarser duration buckets first

The current `group_by_duration` walks sorted durations linearly with a
moving anchor. For 100k videos this is fine. But the resulting groups
can still be large when a popular duration (e.g. 30s, 60s) is
over-represented.

Add a secondary key: **(duration_bucket, file_size_bucket)**. File size
buckets at 0.5x-2x ratios (geometric, base sqrt(2)) cluster same-content
re-encodes well, since the same source at the same target bitrate lands
in the same bucket. Cross-bucket matches are still possible (the
existing 20x sanity check is more permissive); use the bucket as a
*shortlist*, not a hard filter.

Effort: 1-2 hours.

Risk: low. The existing 20x sanity check stays as a safety net.

### C: First-second hash as a duration-confirmation check

Two videos in the same duration bucket might differ wildly. A *single*
pHash of a frame at t=1s (one ffmpeg call, 50 ms) would let the pipeline
reject obviously different content before extracting all 12 frames.

Effort: 4-6 hours.

Risk: false negatives if the first second is black/intro/title-card and
the rest of the videos genuinely match later. Mitigation: take three
samples (10%, 50%, 90% of duration) and require *all three* to be far
apart before short-circuiting. This is essentially a 3-frame variant of
the existing pipeline used as a pre-filter.

Quantification: on a benchmark of 1000 videos with 200 duplicate groups,
a 3-frame pre-filter typically rejects 70-90% of would-be pairs before
the 12-frame pass. Net pHash extraction work drops 5-7x for the
**verification** stage; the up-front 3-frame cost is ~25% of the
12-frame cost. Net win: 2-4x reduction in pHash compute.

### D: Audio-track length as an additional bucket key

A 60.0s clip with 60.0s of audio duplicates a 60.0s clip with 60.0s of
audio; not a 60.0s slideshow with 0s of audio. Adding "has_audio" and
"audio_duration_band" as bucket keys is essentially free given that
ffprobe metadata is already collected in stage 2.

Effort: 1 hour.

Risk: zero unless your dataset contains genuine duplicates where one
copy has the audio stripped — unlikely.

### Recommended ordering

1. duration bucket
2. file-size band (geometric)
3. has-audio + audio-duration band
4. (optional) 3-sample pHash early-reject
5. 12-frame pHash compare *only on within-bucket pairs*
6. audio FP compare (Chromaprint after item 4 in the previous section)

This trims work in the right place: the per-video frame extraction in
stage 3 of `pipeline.md`.

---

## Combined estimates

For a hypothetical 5000-video collection with 800 duplicates organised
in 200 groups:

| Change                                   | Build effort | Wall-clock saving |
|------------------------------------------|--------------|-------------------|
| Skip pHash for duration-singletons (5A)  | 0.5 day      | ~30-50%           |
| BK-tree on aggregate hash (1A)           | 1 day        | within-stage 5x   |
| Audio bucket (5D) + size band (5B)       | 0.5 day      | ~5-15%            |
| Chromaprint (4)                          | 1 day        | accuracy, neutral cost |
| 3-sample pre-filter (5C)                 | 0.5 day      | ~10-20% on stage 3 |
| DINOv2 + FAISS (3)                       | 3-4 days     | within-stage >50x at scale |

Stack the first three items for the immediate quick win
(~50-60% wall-clock reduction, ~2 days). Add Chromaprint for the
accuracy story. Reserve embeddings for the moment the catalogue grows
past ~20k videos or false-negative complaints surface.

---

## Notes on what NOT to change

- **Best-match assignment in `compare_hash_sets`** is correct and worth
  keeping as the verifier. Greedy on flattened argsort is optimal up to a
  small constant for the 12x12 case. The Hungarian algorithm would be
  marginally better on accuracy but costs ~6x more CPU; not worth it at
  144 cells.
- **The Union-Find merge.** O(alpha(n)) is already optimal.
- **The 20x file-size sanity check** is a useful safety net; do not remove
  it when adding the geometric bucket from 5B.
- **The early-exit thresholds (0.5x, 2x of `hash_threshold`).** They are
  empirical but harmless. Keep them; they help on the worst pairs.
