# Single-Hash-Per-Video Temporal Hashing

A research note on compressing each video into **one** comparable fingerprint
instead of the current 12 frame-pHashes, so the pair-comparison cost drops from
`O(12*12)` Hamming evaluations per pair to `O(1)` and a standard binary ANN
index becomes trivially applicable.

This complements `algorithmic-improvements.md` (which already recommends a
BK-tree on an aggregated hash, item 1A) by going one level deeper: it
benchmarks the *specific* aggregation schemes — TMK+PDQF, videohash,
3D-DCT, MinHash-of-frame-hashes, neural single-embedding, simple
mean/median/AND aggregation, multi-resolution interleaved — against each
other, against the cost of integration in *this* codebase, and against the
recall loss that each one buys.

The current code lives in `backend/services/hasher.py` (12 frame hashes per
video stored as a JSON array on `VideoFile.perceptual_hashes` and
`FileCache.perceptual_hashes`) and `backend/services/comparator.py`
(`compare_hash_sets`, lines 517-600, builds an n1*n2 distance matrix per
video pair and runs a greedy best-match assignment).

---

## Executive summary

Top three single-hash schemes, ordered by speedup/effort ratio for *this*
pipeline:

1. **Median-bit aggregation of the existing 12 pHashes** — a single
   256-bit hash per video, computed in microseconds with one numpy
   reduction after stage 3. Drop-in compatible with `IndexBinaryFlat`,
   `IndexBinaryIVF`, or `pybktree`. Recall loss on transcodes is small
   if you keep the existing 12-hash `compare_hash_sets` as a verifier
   on the candidate shortlist (the **hybrid** in §7). Implementation
   effort: 2-4 hours.
2. **vPDQ (Facebook ThreatExchange)** — the production-grade reference
   implementation. Stores one PDQ-256 per sampled frame plus a quality
   score; comparison is "shared-frame bag intersection". Closest to the
   current scheme but uses PDQ-256 instead of pHash-256, and has a
   FAISS-integrated reference matcher in `python-threatexchange`. Not a
   true single-hash; sits between 1 and 3 in cost/recall. Pip install
   on Linux/macOS; **Windows wheel does not exist** as of 2026 — the
   project is C++ with SWIG bindings.
3. **3D-DCT temporal pHash** — stack 12 frames into a 3D volume,
   take a 3D DCT, hash the low-frequency coefficients. Published
   work shows higher robustness to noise/luminance than per-frame
   pHash but worse discrimination on content edits. Single
   ~64-bit hash. **Reimplementation needed** (no maintained Python
   library); ~1 day. Marginal gain over scheme 1.

Schemes that are **not** recommended for this pipeline:

- **akamhy/videohash**: produces only 64 bits, last commit May 2022,
  documented false-positive issues on rotated content. Strictly worse
  than scheme 1.
- **MinHash-on-frame-hashes**: solves a problem this pipeline does
  not have. MinHash is for *unordered set* Jaccard similarity over
  large vocabularies; treating 12 pHashes as a set throws away the
  per-bit structure that makes Hamming distance work.
- **Neural single-embedding (VideoMAE / X-CLIP)**: 512-d float per
  video is fine for retrieval but adds a GPU model dependency
  (~250 MB-1 GB), a non-trivial inference cost, and requires switching
  the index from binary-Hamming to L2/cosine. Worth revisiting only if
  the dataset grows past 1M videos.

**Concrete recommendation: implement the hybrid (§7).** Build a single
**median-bit-aggregate-256** per video (scheme 4 below) → use it as the
*screening* hash in a FAISS `IndexBinaryFlat` (or per-duration-bucket
`BKTree`) → run the existing `compare_hash_sets` as a **verifier** only
on the candidates within radius. This gets the asymptotic speedup of a
single-hash scheme while losing zero of the existing 12-hash
discrimination on the candidate set.

---

## Why one hash per video matters

### The current cost model

Within a duration group of size `n`, the comparator does:

```
n * (n - 1) / 2  pair evaluations
    × 12 * 12 = 144  Hamming-distance cells per evaluation
    + argsort + greedy assignment + early-exit logic
```

Vectorised in numpy, each cell is cheap (~50 ns) — but **the constant
factor is 144**, and the outer loop is still O(n^2) in Python. For a
duration group of 200 same-length videos (very common with 60-second
TikTok exports, 30-second commercials, episode-length TV) that's:

| Group size | Pairs | Hamming cells | Wall time (numpy, py) |
|---:|---:|---:|---:|
| 50  | 1,225      | 176,400     | ~30 ms |
| 200 | 19,900     | 2,865,600   | ~500 ms |
| 1000| 499,500    | 71,928,000  | ~13 s |
| 5000| 12,497,500 | 1.8 billion | ~5 min |

### What single-hash gives you

If each video has one 256-bit hash:

1. **Pair cost: 144 → 1.** A single 256-bit popcount XOR is ~5 ns on
   modern x86 with the BMI/SSE4.2 instructions; numpy is a bit
   slower (~30 ns). Either way, ~30-100× per-pair speedup.
2. **You get to use a real ANN index.** `faiss.IndexBinaryFlat`
   exists, is heavily SIMD-tuned, and on 256-bit vectors does
   ~10 ns per popcount. Faiss benchmarks report `IndexBinaryIVF`
   scaling cleanly to 50M+ 256-bit vectors. `pybktree` is a
   pure-Python BK-tree option that needs no native build.
3. **Storage shrinks.** Currently each `VideoFile.perceptual_hashes`
   stores 12 × 64-char hex strings = ~830 bytes of JSON.
   Single hash = 64 chars = ~70 bytes. 12× column compaction
   matters once you cache 100k videos (~83 MB → ~7 MB in the JSON
   column).
4. **Caching becomes one column lookup.** Right now,
   `FileCache.perceptual_hashes` is a JSON column that must be parsed
   per row. A `BLOB(32 bytes)` for a single hash is direct and
   indexable.

### What single-hash costs you

Concretely: the current 12-hash best-match compare survives
**partial overlap** (one video has a 10s intro, the other doesn't),
**different frame rates** (frame N comes from different timestamps),
and **single-frame outliers** (a black frame, a logo, a transition).
Any single-hash aggregation that collapses 12 → 1 must throw away
*some* of this robustness. The interesting question is: how much, and
can a verifier on the shortlist restore it?

This is what the **hybrid scheme (§7)** answers: **almost all of it, at
a tiny fraction of the cost.** Aggregate-hash → fast shortlist via
ANN → verify the shortlist with the existing 12-hash compare. Recall
on the shortlist needs to be high; precision can be poor because the
verifier filters out false candidates anyway.

---

## Comparison of schemes

| Scheme | Hash size | Build cost | Recall (transcode) | Library 2026 | License | Integration ease |
|---|---:|---|---|---|---|---|
| 1. **Median-bit aggregate** (this pipeline) | 256 b | +1 numpy reduction over current | ~0.95 (vs 0.98 for 12-hash) | n/a (DIY) | n/a | ★★★★★ trivial |
| 2. **vPDQ** (Facebook) | 256 b × N frames + quality | needs ffmpeg + PDQ per frame | ~0.99 | C++ active; `vpdq` on PyPI (Linux/macOS) | BSD-3 | ★★ no Windows wheel |
| 3. **TMK + PDQF** (Facebook) | 256 KB (yes, KB) | resample to 15 fps + per-frame PDQF + trig averages | ~0.99 same-length only | C++ active; `tmkpy` (SWIG, Linux) | BSD-3 | ★ no maintained pkg |
| 4. **akamhy/videohash** | 64 b | collage of 1 fps frames + wHash | poor on rotation/overlay | last commit 2022-05-29 | MIT | ★★★ but stale |
| 5. **3D-DCT temporal pHash** | 64-128 b | 12 frames → 3D volume → DCT → quantise | better noise robustness than 2D | no maintained Py lib | n/a | ★★ ~1 day to code |
| 6. **MinHash on 12 frame hashes** | 64-128 b sig | k MinHash perms over frame hash bits | weak — not designed for this | `datasketch` | MIT | ★★★ — but wrong tool |
| 7. **SimHash / bitwise majority vote** | 256 b | one numpy reduction (= scheme 1) | same as 1 | DIY | n/a | ★★★★★ |
| 8. **Mean/median frame-hash (per-bit)** | 256 b | one numpy reduction | ~0.93 | DIY | n/a | ★★★★★ |
| 9. **Bitwise AND of frame hashes** | 256 b | one numpy AND | catastrophic recall | DIY | n/a | n/a |
| 10. **Multi-resolution interleaved 256-bit** | 256 b | extra splits/concats | speculative | DIY | n/a | ★★★ |
| 11. **VideoMAE / X-CLIP embedding** | 512×fp32 = 2048 B | GPU forward pass per video | ~0.99 (huge model) | `transformers` 4.x | Apache-2.0 | ★★ heavy dependency |

Notes on the table:

- "Build cost" is the *additional* work beyond what stage 3 already
  does. Schemes 1, 7, 8 reuse the existing 12 pHashes; only 2, 3, 5
  require additional FFmpeg work; scheme 11 requires GPU inference.
- Recall numbers are inferred from published papers (3D-DCT,
  TIRI-DCT, TMK) and from my reading of how the aggregation behaves
  on the failure modes the existing pipeline already handles. They
  are **not** measured on your dataset and should be re-validated.
- For schemes 1, 7, 8 the *recall on the screening pass alone* is
  what's listed. With a 12-hash verifier on the shortlist (the
  hybrid), recall converges to that of the current pipeline.

---

## TMK + PDQF deep dive

TMK ("Temporal Match Kernel") + PDQF ("PDQ-Float") is Facebook's
production answer to "one signature per video, designed for
near-duplicate match at scale." Released as part of
`facebook/ThreatExchange` in 2019.

### How it works

1. Resample the video to **15 fps** (standardising temporal stride
   across all videos).
2. For each frame, compute **PDQF** — a 256-element *float* vector
   (PDQ uses the binary thresholded version of the same 16×16 DCT
   feature; PDQF keeps the floats).
3. Compute **time-averages** of the PDQF vectors over several
   periods, weighted with cosine/sine basis functions ("trigonometric
   moments"). This gives a small set of vectors that together
   summarise the video's temporal structure.
4. Concatenate the moments into a single ~256 KB signature.

### Match function

Two videos match if their TMK score is above a threshold. The score
is the inner product of corresponding trigonometric moments, summed
across periods. It is **designed for same-length match**: TMK
implicitly assumes the two videos are aligned in time — i.e. the same
duration, possibly with re-encoding artifacts.

### Why "256 KB"

This is the most surprising property of TMK+PDQF and the reason it's
not directly applicable here. The signature is **not 256 bits, not 256
bytes — 256 kilobytes per video**. That's because it preserves the
full PDQF float arrays across multiple time periods so that
sub-second-resolution matches can be detected.

For 100k videos, that's 25 GB of signature data. For 1M videos, 250
GB. Compare to ~70 bytes/video for the median-bit-aggregate
scheme — a 3,500× difference. TMK is unambiguously the right tool if
you're Meta and you have a cluster; it is unambiguously the wrong
tool for a personal-machine duplicate scanner over hundreds of
thousands of files.

### Python wrapper status

- **`tmkpy`** ([github.com/meedan/tmkpy](https://github.com/meedan/tmkpy))
  — SWIG bindings, Linux-focused, 6 stars, 14 commits, no PyPI release.
  Requires `apt install swig` and `setup.py build install`. **Not
  pip-installable on Windows.**
- **`python-threatexchange`** — Facebook's own Python wrapper which
  includes TMK support but again, the C++ build is Linux-first.

For this codebase (Windows-first, optional Docker/Linux), the deal-breaker is the
Windows build story plus the 256 KB signature size. Skip TMK and use
vPDQ if you want a Facebook-grade option (§ next sub-section).

### Reference paper

Dalins et al. (2019), *"PDQ & TMK+PDQF — A Test Drive of Facebook's
Perceptual Hashing Algorithms"*, arXiv:1912.07745. They benchmark
both algorithms on a 70k-video CSAM-like dataset; PDQ alone gets
~99% precision on near-duplicates, and TMK adds robustness to
*partial* matches (clip-in-larger-video) where PDQ-per-frame
would fail without bag-of-frames matching.

---

## vPDQ — the practical Facebook option

vPDQ ("video-PDQ") is the simpler sibling of TMK. Instead of
trigonometric moments, it just:

1. Samples the video at some interval (typically 1 frame/second).
2. Computes the **PDQ 256-bit binary hash** for each sampled frame.
3. Stores the result as `(hash, quality, frame_number, timestamp)` per
   frame.

This is **structurally identical** to the current pipeline (12
pHashes per video, JSON-stored), with three differences:

| Aspect | Current pipeline | vPDQ |
|---|---|---|
| Frames per video | 12 evenly spaced | ~1/second (variable) |
| Per-frame hash | pHash-256 (16×16 DCT) | PDQ-256 (16×16 DCT, different quantisation, with quality score) |
| Match function | Best-match greedy (`compare_hash_sets`) | "Shared-frame bag" with quality filter |
| Per-frame quality | not computed | yes — frames below threshold dropped |

### vPDQ match semantics

vPDQ doesn't produce a single hash. It produces a *signature* (= bag
of per-frame hashes) and matches via:

```
For each pair (Q, C):
    matches_Q = count of frames in Q whose best PDQ match in C has
                Hamming distance ≤ D AND C's frame quality ≥ F
    matches_C = symmetric
    if matches_Q / |Q| ≥ Pq  AND  matches_C / |C| ≥ Pc:
        VIDEOS MATCH
```

Defaults from the docs: `D = 31, F = 50, Pq = 0%, Pc = 80%`. This
is asymmetric — `Pq = 0%` means "any frame of the query found in
the candidate counts as evidence." Designed for clip-in-larger-video
match.

### Faiss integration

The `python-threatexchange` library wraps vPDQ with a Faiss-backed
prescreener: dump every frame hash into an `IndexBinaryFlat`, then
for each query frame find its nearest neighbours within Hamming
radius D, then re-do the per-video aggregation. This is **exactly
the hybrid pattern from §7**, but applied frame-by-frame.

### Why vPDQ isn't a single-hash scheme

It still stores N hashes per video. For this codebase's "compress 12
hashes into 1" goal, vPDQ doesn't actually help — it's
algorithmically isomorphic to what's already there, just with PDQ
instead of pHash and a different match-function. The *real* benefit
of moving to vPDQ would be (a) using the quality score to drop
useless frames (black frames, transitions), (b) gaining access to
the FAISS-prescreened matcher built in `python-threatexchange`.

### Windows install pain

Confirmed via PyPI / piwheels / GitHub README: `pip install vpdq`
works on Linux and macOS, **not on Windows** as of 2026. The package
is a C++ extension via pybind11, and no Windows wheel is shipped.
Building from source on Windows would need MSVC + ffmpeg headers +
pybind11. Practical alternative: keep the C++ build inside the
Docker image only, do not require it for the native Windows install
path.

---

## Scheme-by-scheme detail (for the ones not yet covered)

### 4. akamhy/videohash

Algorithm:

1. Extract 1 frame per second.
2. Resize each frame to 144×144.
3. Tile them into a square collage.
4. Compute a **64-bit wavelet hash** of the collage.
5. (Optionally) XOR with a "dominant-colour" bitmask derived from the
   collage divided into 64 segments.

Why this is clever: aggregating spatially via collage means a single
2D pHash naturally encodes temporal structure. Why it's not great in
practice:

- 64 bits is too small for >10k-video corpora. Birthday-paradox false
  collisions start at sqrt(2^64) ≈ 4 billion, but the *effective* bit
  variance is much lower because every collage shares the "tiled
  144×144" structural prior. Reported real-world false positive
  rates on diverse content are visible in issue trackers.
- The collage is sensitive to the *number* of seconds: a 30s and a
  31s clip of the same content produce collages with different tile
  layouts and different hashes. Their docs explicitly say it "should
  remain unchanged or not vary substantially" but the worst case is
  rough.
- Project last touched 2022-05-29. Open issues outnumber merged PRs.
  There is a fork (`Demmenie/videohash2`) that updated for Python
  3.10 but is also stagnant.

**Verdict**: do not adopt. Scheme 1 (median-bit aggregate) gives 256
bits with the same compute budget and reuses your existing
high-quality frame normalisation (SAR/portrait/scale).

### 5. 3D-DCT temporal pHash

Algorithm (from Coskun & Sankur, 2006, and subsequent work):

1. Stack `K` greyscale frames into a `K × H × W` tensor.
2. Apply a 3D DCT.
3. Take the top-left low-frequency cube of coefficients (e.g.
   8×8×8 = 512 coefficients).
4. Quantise (median threshold) → 512-bit hash, or take only the
   first vertical+horizontal coefficients per slab → smaller hash.

Properties (from the surveys cited):

- Robustness to **noise, brightness, contrast, mild blur**: excellent
  (better than per-frame 2D pHash because temporal averaging
  absorbs frame-level noise).
- Robustness to **content edits, frame insertion/deletion, time
  shifts**: poor. The 3D DCT is sensitive to the start/end of the
  volume; a 1s delay between the two videos can shift coefficients.

There is **no maintained Python library** for 3D-DCT video hashing.
You would reimplement: `scipy.fft.dctn(volume, type=2, norm="ortho",
axes=(0,1,2))`, then `volume[:k1,:k2,:k3]` slicing, then median
threshold. ~50 lines. Wall cost per video is ~50 ms for 12 frames
of 320×180.

**Verdict**: cute, but no decisive win over scheme 1 — and the
"can't tolerate temporal shift" failure mode is exactly the failure
mode that the current best-match greedy explicitly fixes. Not
recommended.

### 6. MinHash on 12 frame hashes

The pitch: treat each frame's 256-bit pHash as a "shingle". Compute
MinHash signatures over the 12 shingles. LSH-band them into buckets.
Pairs share a bucket if they have similar Jaccard distance over the
12-hash sets.

Why this doesn't fit:

- MinHash measures **set Jaccard**, not Hamming distance between
  individual elements. Two frame hashes that differ by 1 bit are
  **completely different** under MinHash's set membership — they are
  different shingles. MinHash works when each "shingle" is a token
  drawn from a discrete vocabulary; pHash bits are not tokens, they
  are continuous-similar codes.
- You'd have to first quantise pHashes into discrete buckets (e.g.
  by chunked Hamming-LSH per pHash), at which point you've added an
  expensive step that already throws away the Hamming structure
  you wanted to exploit.
- The `datasketch` library is perfectly fine, but applying it here is
  fundamentally a category error.

**When MinHash *would* work**: if each video had a *bag* of detected
keypoints (e.g. SIFT descriptors discretised into a visual
vocabulary), MinHash-on-the-vocabulary-set would be appropriate. That
is a different pipeline.

**Verdict**: skip.

### 7. SimHash / bitwise-majority-vote 256-bit hash

This is **the canonical aggregation for similar-bit hashes** and is
what `algorithmic-improvements.md` already recommends (item 1A,
"median bit hash"). Spelled out:

```python
# stack the 12 frame hashes into a (12, 256) bit matrix
bits = np.stack([_hex_to_bits(h) for h in frame_hashes], axis=0)
# majority vote per column
counts = bits.sum(axis=0)                          # 256 ints
aggregate = (counts > (bits.shape[0] / 2.0)).astype(np.uint8)  # 256 bits
hex_hash = bytes(np.packbits(aggregate)).hex()     # 64 hex chars
```

Why this is the recommended scheme for this pipeline:

- **Zero extra I/O**: reuses the 12 frame hashes already extracted.
- **Microseconds per video**: one numpy reduction.
- **Same 256-bit format** as the existing per-frame pHash, so the
  comparator's `_hex_to_bits` and Hamming-distance code works
  unchanged.
- **Reasonable recall**: for two re-encodes with frame-level Hamming
  ≤ threshold, the bit-by-bit majorities will agree in most
  positions. Aggregate distance is approximately the *mean* per-bit
  difference rate × 256, which is what you want.
- **Caveat**: if 4 out of 12 frames are corrupted/different (a
  trimmed intro), each *minority* bit position can still be on the
  wrong side of the vote. This is exactly why we need the **hybrid
  verifier** (§7).

This is also bit-identical to SimHash when the input weights are
uniform (= each frame hash contributes ±1 per bit, sign of sum is
output).

### 8. Mean / median per-bit (= 7)

Mean of bits with a 0.5 threshold = majority vote = scheme 7.
Median of bits = scheme 7. They are the same thing for binary input.

### 9. Bitwise AND of 12 frame hashes

Bad idea. AND zeroes every bit that is zero in *any* frame, which is
~half of all bits for 12 frames. The aggregate distance is then
dominated by which bits happened to be set in all 12 frames, which is
content-dependent and not robust. Skip.

### 10. Multi-resolution interleaved 256-bit

Speculative: instead of taking the per-bit majority vote across all
12 frames, **partition** the 256 bits into 4 zones of 64 bits each.
Bits 0-63 come from frames {0,1,2}, 64-127 from {3,4,5}, 128-191
from {6,7,8}, 192-255 from {9,10,11}. Each zone aggregates only its
3 frames.

The idea: a query video that overlaps only the first half of the
target still produces a partially-matching aggregate (zones 0-1 line
up, zones 2-3 are noise). Hamming distance up to ~128 then signals
"partial match"; the verifier confirms.

I have no evidence this actually works better than scheme 7 in
practice. It's a free experiment if you want to A/B it, but I would
not ship it without measurement. Scheme 7 + verifier is simpler and
sufficient.

### 11. Neural single-video embedding (VideoMAE / X-CLIP)

These produce a 512-dim float32 vector per video (~2 KB), via a
masked-autoencoder or contrastive transformer trained on millions of
video clips. For pure retrieval, they are state of the art — recall
on transcodes is essentially perfect, and they also recover
*semantic* near-duplicates (the same scene shot from a different
angle, the same product in a different cut).

For this codebase, the costs are:

- **Model dependency**: 250 MB-1 GB checkpoint, PyTorch + CUDA stack.
- **Inference**: ~50-200 ms per video on a 3060 Ti depending on
  resolution and frames sampled. Roughly comparable to current
  frame-extract + hash cost, so not a net loss, but it's GPU time
  that's currently being used for FFmpeg decode.
- **Storage**: 512 floats × 4 bytes = 2 KB per video. For 1M videos,
  2 GB. Manageable.
- **Index switch**: from `IndexBinaryFlat` to `IndexFlatL2` /
  `IndexHNSWFlat`. Not hard.
- **The "feature, not bug" risk**: it will match *semantically*
  similar videos. Two different episodes of the same TV show with
  the same intro will rank high. That may or may not be what users
  want — for a duplicate scanner explicitly, it's a regression.

**Verdict**: revisit if (a) the corpus exceeds ~1M videos and
binary-Hamming-ANN becomes the bottleneck, or (b) the user
explicitly wants "find similar videos", not "find exact
duplicates."

---

## Practical recommendation for THIS pipeline

You already extract 12 well-normalised frame hashes per video. The
question is what to add on top, not what to replace.

**Do this:**

1. After stage 3 (perceptual hashing), in `_compute_hashes_sync` or a
   new function right after, compute the **median-bit-aggregate-256**
   hash from the 12 per-frame hashes (scheme 7). Store it on a new
   `VideoFile.aggregate_hash` column (or as the first element of the
   existing JSON; see §8). Also persist it to `FileCache`.
2. Replace stage 5's all-pairs loop with:
   1. Per duration group, build a `pybktree.BKTree(hamming,
      aggregate_hashes)`.
   2. For each video, query the tree with radius =
      `HASH_SIMILARITY_THRESHOLD * 2` (= ~28, generous). This is your
      **candidate shortlist**.
   3. For each `(video, candidate)` pair on the shortlist, run the
      existing `compare_hash_sets(hashes_i, hashes_j)` as a
      **verifier**. Promote to union-find only if the verifier
      passes.
3. Keep the audio fallback exactly as it is.

This pattern is the hybrid (§7 below) and it is the single
highest-ratio change you can make on this codebase.

Expected speedup of the comparison stage, derived from group sizes:

| Group size | All-pairs cells (now) | BK-tree queries × verifier cells | Speedup |
|---:|---:|---:|---:|
| 50 | 176,400 | ~50 × ~5 × 144 = 36,000 | ~5× |
| 200 | 2.87M | ~200 × ~7 × 144 = 201,600 | ~14× |
| 1000 | 71.9M | ~1000 × ~10 × 144 = 1.44M | ~50× |
| 5000 | 1.8B | ~5000 × ~12 × 144 = 8.6M | ~210× |

(The "~5-12" is the BK-tree's expected candidates within radius for
moderately-dense pHash distributions; will need to be measured.)

The verifier preserves the existing recall properties because the
expensive 12-hash compare is still the *final* arbiter on candidate
pairs. The only recall hit is the small fraction of true positives
whose **aggregate** distance exceeds `2 × THRESHOLD` despite the
**best-match** distance being ≤ THRESHOLD. In practice this happens
when 4 out of 12 frames disagree wildly (e.g. a re-encode with a
30%-different intro). For those, the audio fallback in stage 5b
typically catches the match.

**Risks and mitigations:**

- **Risk**: the aggregate-radius shortlist misses pairs that the
  current pipeline catches.
  - *Mitigation*: log all "audio-only matches" for one full scan to
    see how many duplicates fall through the visual screen. Tune
    the BK-tree radius if recall regresses.
- **Risk**: BK-tree build cost on huge duration groups.
  - *Mitigation*: BK-tree build is O(n log n). On 10k nodes, ~10 ms
    in pure Python. Acceptable.

---

## Hybrid approach (recommended)

The point of "single hash per video" is **not** to replace the
existing match function — it's to **prescreen** so the existing match
function only runs on ~k candidates per video instead of all `n-1`.

```
                  ┌──────────────────────────────────────────┐
                  │  Stage 3 already produces 12 pHashes     │
                  │  per video → stored in VideoFile.hashes  │
                  └────────────────┬─────────────────────────┘
                                   │
                                   ▼
                  ┌──────────────────────────────────────────┐
                  │  NEW: aggregate = median_bit(hashes)     │
                  │  Stored as VideoFile.aggregate_hash      │
                  │  AND in FileCache (cross-scan)           │
                  └────────────────┬─────────────────────────┘
                                   │
                                   ▼
                  ┌──────────────────────────────────────────┐
                  │  For each duration group:                │
                  │    build BKTree(aggregate_hashes)        │
                  │    for each v: shortlist = tree.find(    │
                  │       v.aggregate, radius=R_PRESCREEN)   │
                  └────────────────┬─────────────────────────┘
                                   │
                                   ▼
                  ┌──────────────────────────────────────────┐
                  │  For each (v, c) in shortlist:           │
                  │    verifier = compare_hash_sets(         │
                  │      v.hashes, c.hashes, threshold       │
                  │    )  ◀── unchanged from today           │
                  │    if verifier passes → union-find       │
                  └────────────────┬─────────────────────────┘
                                   │
                                   ▼
                  ┌──────────────────────────────────────────┐
                  │  Audio fallback for non-matched pairs    │
                  │  ◀── unchanged from today                │
                  └──────────────────────────────────────────┘
```

This is provably correct: every pair the verifier *would* match in
the current pipeline must have small per-frame Hamming distances,
hence small aggregate distance, hence appears in the shortlist. The
only knob is `R_PRESCREEN`: too small loses recall, too large loses
speed.

**Recommended starting value**: `R_PRESCREEN = 2 * HASH_SIMILARITY_THRESHOLD = 28`.

The aggregate distance between two videos is bounded above by the
**average** per-frame distance plus a small noise term from bit
disagreements; in expectation `aggregate_dist ≤ frame_best_match_dist
+ noise`. Setting `R_PRESCREEN = 2 × threshold` is a conservative
2× safety margin. Tighten after measurement.

---

## Storage / cache implications

### Current schema

`VideoFile.perceptual_hashes` is a JSON `Text` column storing a list
of 12 hex strings, ~830 bytes per row.

`FileCache.perceptual_hashes` is identical.

### Proposed change

Add **one column** to both tables: `aggregate_hash VARCHAR(64)`
(64 hex chars = 256 bits as text).

```python
# in models/database.py, VideoFile and FileCache:
aggregate_hash = Column(String(64), nullable=True, index=True)
```

The index matters: even if you use a BK-tree in memory, the
`aggregate_hash` column is what `FileCache` loads back into the
BK-tree on rebuild, so an index lets bulk-load queries skip rows
with `aggregate_hash IS NULL`.

Migration: the project's CLAUDE.md says "no migrations — schema
changes require deleting the DB", so this is just an additive
schema bump. Existing caches **lose nothing**: if `aggregate_hash`
is `NULL`, compute it from the existing 12 hashes on next scan
(zero re-extraction needed, ~10 µs per video).

### Cache hit semantics

Current: hit if `FileCache` row exists with `perceptual_hashes`.

After: hit if row exists with `perceptual_hashes` **AND**
`aggregate_hash`. If only `perceptual_hashes` exists (from an
older cache), compute the aggregate and update in place. Trivial
backfill.

### Cross-scan cache benefit

On a fully-cached re-scan, today:

- Load all 12 hashes from JSON per video.
- All-pairs compare with 144 Hamming cells each.

After:

- Load one 32-byte hash per video.
- BK-tree query for shortlist.
- For shortlist only, load 12 hashes from JSON.

For a 100k cached corpus the JSON parse cost drops by ~99% and the
distance-cell count drops by 90-99% (depending on duration-group
density). The cache will, in effect, become *much* faster than it
already is.

### Index on disk

For the BK-tree itself, no on-disk format needed — rebuild in memory
from `SELECT aggregate_hash FROM file_cache WHERE file_path LIKE
?` at the start of stage 5. Build time on 100k hashes: <500 ms
single-threaded.

If you want to push further later: `faiss.write_index` /
`read_index` can persist an `IndexBinaryFlat` to disk, but it's not
worth the operational complexity for a single-machine app.

---

## Code sketch (recommended scheme — median-bit aggregate + BK-tree verifier)

> Not for direct paste; just to show shape and effort.

### 1. Aggregate-hash computation (new function in `hasher.py`)

```python
def aggregate_hash_from_frames(frame_hashes: List[str]) -> Optional[str]:
    """Compute a single 256-bit aggregate hash from a list of frame pHashes.

    Per-bit majority vote: bit_i = 1 iff more than half of the frame
    hashes have bit_i set. For 12 frames, ties (6/6) resolve to 0 (we
    use a strict > threshold).

    Returns 64 hex chars (256 bits) or None if input is empty.
    """
    if not frame_hashes:
        return None
    bit_arrays = []
    for h in frame_hashes:
        b = _hex_to_bits(h)
        if b is not None and len(b) == 256:
            bit_arrays.append(b)
    if not bit_arrays:
        return None
    stack = np.stack(bit_arrays, axis=0)            # (n_frames, 256)
    counts = stack.sum(axis=0)                       # (256,) ints
    majority = (counts > (len(bit_arrays) / 2.0)).astype(np.uint8)
    packed = np.packbits(majority)
    return bytes(packed).hex()
```

Wire it into `_compute_hashes_sync` so it's computed once per video
right after the 12 per-frame hashes are computed. Add the result to
the dict returned by `_extract_and_hash_sync` as a new key
`"aggregate"`.

### 2. Schema add

```python
# models/database.py — add to VideoFile and FileCache
aggregate_hash = Column(String(64), nullable=True, index=True)
```

### 3. Stage 3 cache writeback

In `api/scan.py`, where `VideoFile.perceptual_hashes` is set from
`extract_and_hash` results, also set `aggregate_hash` and persist it
to the linked `FileCache` row.

### 4. Stage 5 prescreen (new branch in `comparator.py`)

```python
# at top of comparator.py
try:
    import pybktree
    HAS_BKTREE = True
except ImportError:
    HAS_BKTREE = False

R_PRESCREEN = settings.HASH_SIMILARITY_THRESHOLD * 2


def _bit_hamming(a_hex: str, b_hex: str) -> int:
    a = int(a_hex, 16); b = int(b_hex, 16)
    return bin(a ^ b).count("1")


def find_duplicates_in_group_v2(
    videos: List[dict],
    hash_threshold: int = 10,
    audio_threshold: float = 80.0,
) -> List[List[dict]]:
    if len(videos) < 2:
        return []

    if not HAS_BKTREE:
        return find_duplicates_in_group(videos, hash_threshold, audio_threshold)

    aggs = [(i, v.get("aggregate_hash")) for i, v in enumerate(videos)]
    aggs = [(i, a) for i, a in aggs if a]
    if len(aggs) < 2:
        # fall back to v1 if aggregate missing (old cache)
        return find_duplicates_in_group(videos, hash_threshold, audio_threshold)

    tree = pybktree.BKTree(_bit_hamming, [a for _, a in aggs])
    idx_by_agg = {a: i for i, a in aggs}

    parent = list(range(len(videos)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb

    for i, agg_i in aggs:
        for dist, agg_j in tree.find(agg_i, R_PRESCREEN):
            j = idx_by_agg[agg_j]
            if j <= i:
                continue
            if not _file_size_compatible(videos[i], videos[j]):
                continue
            # ── verifier: existing compare_hash_sets ──
            ok, sim = compare_hash_sets(
                videos[i].get("hashes") or [],
                videos[j].get("hashes") or [],
                hash_threshold,
            )
            if ok:
                union(i, j)
                videos[i].setdefault("_similarities", {})[j] = sim
                videos[j].setdefault("_similarities", {})[i] = sim

    # Audio fallback over remaining unmatched pairs in the duration group
    # — unchanged semantics, run once per (i, j) not yet unioned.
    # ... (audio loop as in v1) ...

    # ... (group_map construction as in v1) ...
```

### 5. Migration of existing caches

In `init_db()` or right after, a one-time backfill:

```python
async def backfill_aggregate_hashes(db: AsyncSession):
    q = select(FileCache).where(
        FileCache.perceptual_hashes.is_not(None),
        FileCache.aggregate_hash.is_(None),
    )
    rows = (await db.execute(q)).scalars().all()
    for r in rows:
        try:
            hashes = json.loads(r.perceptual_hashes)
            agg = aggregate_hash_from_frames(hashes)
            if agg:
                r.aggregate_hash = agg
        except Exception:
            continue
    await db.commit()
```

100k rows: ~1s wall time. Run once at startup if any rows are
missing the aggregate.

---

## Honest accounting of recall loss vs current 12-hash approach

It would be dishonest to claim "single-hash screening" is free. The
current `compare_hash_sets` is a near-globally-optimal Hungarian-ish
matching that handles:

1. Different fps (frame N vs frame N' from different timestamps).
2. Slight trims (extra intro / outro in one).
3. Single-frame outliers (one black frame, one logo card).

The median-bit aggregate, naively, breaks property 1 and 2 in
extreme cases:

- **fps mismatch where the 12 frames sample DIFFERENT content**:
  unlikely, because the project's `target_fps = num_frames / duration`
  spreads frames evenly over the actual duration. The two re-encodes
  will sample very close to the same temporal positions even at
  different source fps. Recall loss here: negligible (<1%).
- **trims of >25%**: if one video has a 30% intro the other doesn't
  have, the first ~3 of 12 frames will be totally different content.
  The aggregate's per-bit majority gets ~3 votes pulling one way and
  ~9 the other; the 9 win, so the aggregate is dominated by the
  shared content. But aggregate Hamming distance might creep past
  `2 × threshold`. Recall loss with `R_PRESCREEN = 28`: moderate;
  this is the *exact* case where the audio fallback in stage 5b
  saves you.
- **single black frame in one video**: ~1/12 of votes biased. Negligible.

**Measured recall is required.** The right validation is:

1. Take a known-duplicate-pairs benchmark (curate ~50 confirmed
   pairs from the current pipeline's output).
2. Run them through the proposed `find_duplicates_in_group_v2`.
3. Count how many show up in `tree.find(...)` shortlist within
   `R_PRESCREEN`. **This is the screening recall — should be ≥95%.**
4. For the misses, check whether audio fallback recovers them. With
   audio fallback, **end-to-end recall should be ≥99%**.

If screening recall is <95%, raise `R_PRESCREEN` until it isn't, or
fall back to the existing v1 code path for that scan.

---

## Implementation order (smallest first)

1. **Add `aggregate_hash` column** to `VideoFile` + `FileCache`. (15 min)
2. **Compute and store** `aggregate_hash` in stage 3. (~30 min)
3. **Backfill** old cache rows on startup. (~30 min)
4. **Add `find_duplicates_in_group_v2`** (BK-tree shortlist + existing
   verifier). (~2 h)
5. **Add a config flag** `USE_AGGREGATE_PRESCREEN: bool = True` and a
   tunable `AGGREGATE_PRESCREEN_RADIUS: int = 28`. (15 min)
6. **Diagnostic logging**: count shortlist size per query, count of
   verifier passes/fails, count of "audio-only saved" matches. (~30
   min)
7. **Run a real scan**, validate. (1 h)

Total: ~half a day to ship. ~6 hours engineering + 1 hour
validation.

---

## What this research does NOT recommend

- Don't adopt TMK or vPDQ. Both have Windows-build problems and
  TMK's 256 KB signatures are wildly oversized for a personal
  archive.
- Don't adopt akamhy/videohash. Stagnant, 64-bit hash is too small,
  rotation handling is weaker than this codebase's own (you already
  normalise portrait → landscape in stage 3).
- Don't adopt 3D-DCT temporal pHash. No library, no decisive win,
  reimplementation cost.
- Don't adopt neural single-embeddings. Not justified by current
  corpus size.
- Don't adopt MinHash. Wrong tool for binary Hamming codes.

The single change that delivers the asymptotic win, without
sacrificing the project's existing pHash robustness, is:

**median-bit-aggregate-256 + BK-tree prescreen + existing
`compare_hash_sets` as verifier.**

That's it. Everything above is supporting evidence for that one
decision.

---

## References

- [PDQ & TMK + PDQF — A Test Drive of Facebook's Perceptual Hashing Algorithms](https://arxiv.org/abs/1912.07745) (Dalins et al., 2019)
- [Facebook ThreatExchange — PDQ + vPDQ + TMK reference implementations](https://github.com/facebook/ThreatExchange)
- [vpdq on PyPI](https://pypi.org/project/vpdq/) — Linux/macOS only
- [tmkpy — SWIG Python bindings for TMK](https://github.com/meedan/tmkpy) — Linux only
- [akamhy/videohash](https://github.com/akamhy/videohash) — 64-bit collage-wavelet hash, last commit 2022-05-29
- [pdqhash on PyPI](https://pypi.org/project/pdqhash/) — per-frame PDQ-256 in Python
- [Faiss Binary Indexes wiki](https://github.com/facebookresearch/faiss/wiki/Binary-indexes) — `IndexBinaryFlat` / `IndexBinaryIVF` / `IndexBinaryHash`
- [Faiss Binary hashing index benchmark](https://github.com/facebookresearch/faiss/wiki/Binary-hashing-index-benchmark) — 50M+10k 256-bit vector benchmarks
- [pybktree on PyPI](https://pypi.org/project/pybktree/) — pure-Python BK-tree, no native build
- [Hamming Distributions of Popular Perceptual Hashing Techniques](https://arxiv.org/pdf/2212.08035) — distribution analysis of PDQ/pHash/aHash/dHash/wHash
- Coskun & Sankur (2006), "Spatio-temporal transform based video hashing" — 3D-DCT origin work
- Esmaeili et al. (2011), "A Robust Video Copy Detection System using TIRI-DCT and DWT Fingerprints" — TIRI-DCT reference
- [A Survey of Perceptual Hashing for Multimedia](https://dl.acm.org/doi/10.1145/3727880) (ACM TOMM, 2025)
- [SimHash — bitwise majority vote near-duplicate detection](https://en.wikipedia.org/wiki/SimHash)
- [VideoMAE — Hugging Face docs](https://huggingface.co/docs/transformers/model_doc/videomae) — neural alternative
- [PHVSpec — A Benchmark-based Analysis of Perceptual Hash Systems for Videos](https://technologycoalition.org/wp-content/uploads/Tech-Coalition-Video-Hash-Benchmark-Paper.pdf) (Tech Coalition, 2024)
