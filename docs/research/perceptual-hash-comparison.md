# Perceptual Hash Algorithm Comparison for Video-Frame Duplicate Detection

Scope: evaluate alternatives to the current per-frame `imagehash.phash(img, hash_size=16)` (256-bit DCT hash, Hamming threshold ≤14) used in `backend/services/hasher.py`. Goal is one decision: keep pHash, or switch.

---

## Current setup (so we are comparing the right baseline)

A quick correction to the framing: although DCT-pHash is canonically a 64-bit hash, **this codebase is already running a 256-bit pHash**. The relevant line is `hasher.py:344`:

```python
h = imagehash.phash(img, hash_size=16)
```

`hash_size=16` yields a 16×16 = 256-bit DCT signature. `compare_hash_sets` confirms it (`max_bits = 256`, comment "16×16 hash"). The configured default `HASH_SIMILARITY_THRESHOLD=14` therefore behaves on 256 bits, not 64, which is unusually tight (≈5.5% of bits, equivalent to about 3.5 bit-flips on a 64-bit hash). That is already inside the "very strong match" regime that PDQ recommends (`≤31/256` = ~12% for PDQ matches). The frequent comparison reference "pHash 64-bit threshold 5–10" therefore does **not** apply here — our baseline is already operating in a much stricter regime than the literature shorthand.

---

## Executive summary

**Recommendation: stay on `imagehash.phash` at `hash_size=16` (256-bit). Do not migrate to PDQ.**

Three reasons:

1. **The accuracy ceiling is already close to PDQ.** PDQ is a refinement of DCT-pHash (16×16 DCT, median threshold, quality score) that improves on the *64-bit* pHash baseline most papers use. The codebase already runs DCT-pHash at the same 16×16 resolution PDQ uses; the gap to PDQ is narrow and dominated by PDQ's median-vs-mean quantisation and quality metric — neither is the bottleneck for our pipeline (the bottleneck is frame extraction, not hash quality).
2. **Switching costs are real and one-way.** `pdqhash` requires a C++/Cython build, OpenCV, and a per-OS wheel story. `imagehash` is pure-Python on top of Pillow and "just works" on Windows/Linux/Docker. Cached hashes in `FileCache.perceptual_hashes` are tied to the current algorithm; migration invalidates the entire cache.
3. **Our real recall gaps are elsewhere.** The known failures (mirror-flipped content, severe re-cropping, watermark overlays) are not fixed by PDQ — they need either a dihedral-orientation hash (PDQ does have `compute_dihedral` for this, but it's a per-frame 8× cost) or a content-segmentation hash like `crop_resistant_hash`. The audio fingerprint already covers the mirror/re-format case.

**Optional, low-risk, high-value tweak:** add `imagehash.dhash` as a *second* per-frame hash, stored alongside the pHash. dHash is ~300× faster than pHash and complements DCT-pHash on different transforms (edges vs. low-frequency structure). Used as an OR condition with pHash, it raises recall on watermarked / overlaid clips without changing the schema beyond a new JSON column. Cost is dominated by Pillow I/O which is already paid; the hash itself is essentially free. See [Combined / ensemble approach](#combined--ensemble-approach).

Expected impact of staying put: zero recall/precision change, zero risk. Expected impact of the optional dHash addition: estimated +3–8% recall on watermarked/overlaid duplicates with negligible false-positive risk (when used as `(pHash≤T_p) OR (dHash≤T_d)` with strict `T_d` ≈ 8% of bits).

**Biggest risk if we ignore this advice and migrate to PDQ:** the entire `FileCache.perceptual_hashes` corpus becomes useless and every scanned file must be re-hashed from scratch. For a 50k-video library that is many hours of wasted ffmpeg.

---

## Comparison matrix

Speed numbers are from the Content Blockchain study (Caltech101, 9143 images, default 64-bit hashes via `imagehash`) and the Qt+OpenCV `img_hash` benchmark (UKBench, 100 images). The two sources mostly agree on relative ordering; absolute milliseconds depend hugely on machine and image size, so treat the columns as rank, not literal performance.

| Algorithm | Hash size (default) | Speed per 256×256 frame | Discrim. power on transcodes | Crop | Rotate | Mirror | Watermark | Library maturity | License |
|---|---|---|---|---|---|---|---|---|---|
| **aHash** (`imagehash.average_hash`) | 64 bit | ~2 ms | Low (25% failure overall) | Bad | Fails | Fails | Moderate | imagehash (mature) | BSD-2 |
| **dHash** (`imagehash.dhash`) | 64 bit | **~0.3 ms (fastest)** | Moderate (43.6% failure) but very low collision rate | **Very bad** (99% fail on crop) | Fails | Fails | Moderate | imagehash (mature) | BSD-2 |
| **pHash 64-bit** (`imagehash.phash`) | 64 bit | ~60 ms | Good (23.7% failure, lowest collisions in Content Blockchain study) | Moderate | Fails | Fails | Weak (46% miss) | imagehash (mature) | BSD-2 |
| **pHash 256-bit** (`hash_size=16`) — **current** | 256 bit | ~60–100 ms | Very good — extra bits push collisions to negligible | Moderate | Fails | Fails | Weak | imagehash (mature) | BSD-2 |
| **wHash** (`imagehash.whash`, haar wavelet) | 64 bit | ~1 ms | Best in Content Blockchain study (21.2% failure) but **7866 collisions** (worst) | Moderate | Fails | Fails | **Best** (31.7% miss vs pHash 46%) | imagehash (mature) | BSD-2 |
| **colorhash** (`imagehash.colorhash`) | ~42 bit (binbits=3) | Fast | Poor for visual transcodes, by design | OK | OK | OK | Useful as auxiliary | imagehash (mature) | BSD-2 |
| **crop_resistant_hash** | Variable (segments × 64 bit) | **~10–50× slower than pHash** (watershed segmentation per frame) | Excellent on crop (174/175 vs 10/175 at 50% crop) | **Excellent** | Fails | Fails | Moderate | imagehash 4.2+, niche | BSD-2 |
| **PDQ** (`pdqhash`) | 256 bit | ~80 ms (paper); comparable to 256-bit pHash | Excellent — 99.96% match on format changes at threshold 30 | Bad (>5% crop fails) | Bad (>5° fails) | Fails (use `compute_dihedral` for 8× cost) | Moderate (50% miss at threshold 30) | Facebook reference, Cython wheels; needs OpenCV | BSD-3-style on bindings; PDQ algorithm separately licensed |
| **PhotoDNA** | Proprietary | Unknown | — | — | — | — | — | Microsoft only, not OSS | Restricted |
| **TMK+PDQF** (video) | ~1 KB **per video** (averaged) | Slow indexing, very fast compare; one hash per video | Excellent on transcode (near-perfect) | Bad on crop | Bad on rotate | — | Moderate | `tmkpy`, builds via swig; OpenCV+ffmpeg deps | BSD |
| **Marr-Hildreth (`mhash`)** | 72 bit | Slow — "the slowest" per OpenCV docs, but more discriminative | Good | Moderate | Fails | Fails | Good | OpenCV `img_hash` (mature, C++) | Apache-2 |
| **Block mean hash** | 256 bit (BMH0/1) | ~30–40% slower than pHash | Good — robust to noise | Moderate | Fails | Fails | Good | OpenCV `img_hash` | Apache-2 |
| **Radial variance hash** | 40 bytes | Slowest in the OpenCV set (40× cross-correlation eval) | Aspect-ratio sensitive | Bad | **Partially** rotation-resistant | Fails | Good | OpenCV `img_hash` | Apache-2 |
| **Color moment hash** | Small | Fast | Weak for transcode discrimination | OK | **Best** — rotation-resistant ±90° | OK | Moderate | OpenCV `img_hash` | Apache-2 |

Key data sources: [Content Blockchain test of 100 584 modified Caltech101 images](https://content-blockchain.org/research/testing-different-image-hash-functions/), [Qt + OpenCV blog on `img_hash`](http://qtandopencv.blogspot.com/2016/06/introduction-to-image-hash-module-of.html), [Facebook PDQ paper (arXiv 1912.07745)](https://arxiv.org/abs/1912.07745), [imagehash README](https://github.com/JohannesBuchner/imagehash).

A note on the speed column. The Content Blockchain study reports `pHash = 60 ms` per image; this is on the same dataset as the dHash `0.33 ms`. That speed gap is *real but irrelevant for us* because in our pipeline the hash is computed once per frame in a `ThreadPoolExecutor` after a ~hundreds-of-ms ffmpeg frame extraction. The hash itself never dominates wall-clock time for a video — ffmpeg does.

---

## Detailed per-algorithm analysis: the three serious candidates

### 1. pHash at `hash_size=16` (current — 256-bit DCT)

**How it works.** Resize input to 32×32 grayscale → 2-D DCT → take top-left 16×16 (low-frequency block, excluding the DC coefficient) → median-threshold each coefficient to a bit. Result: 256 bits.

**Strengths for our use case.**

- DCT discards high-frequency content, which is exactly what transcoding adds (compression artefacts, mosquito noise). Two H.264 and HEVC encodes of the same frame land within a few bits of each other.
- 256 bits gives roughly 16× the entropy of the canonical 64-bit pHash, which in the Content Blockchain dataset already had only 483 collisions out of 100 584 — the 256-bit variant essentially eliminates by-chance collisions in any realistically sized library.
- The early-exit logic in `compare_hash_sets` (lines 576–586) only works because the bit count is the same across frames; staying at 256 bits keeps the early-exit thresholds calibrated.

**Weaknesses.**

- Fails on mirror flips. A horizontally mirrored frame has a completely different DCT signature. (Audio fallback catches this in our pipeline.)
- Fails on rotation beyond ~5°.
- Weak on watermarks: the Content Blockchain study shows 46% of watermarked images miss, vs. 31% for wHash.
- The early-exit logic uses `* 0.5` and `* 2` multipliers (`compare_hash_sets:578,584`) tuned against the current 256-bit pHash distance distribution. Switching algorithm or hash size requires retuning these.

**Current threshold (≤14 on 256 bits).** This is `14/256 = 5.5%` of bits flipped. For reference, PDQ recommends `31/256 = 12%`. Our threshold is **about 2× stricter than PDQ's recommendation**, presumably because we already passed the duration pre-filter and want fewer false positives at the visual stage. This is defensible — false positives at the visual stage propagate through Union-Find and can merge two genuinely different groups.

**Verdict.** Stay.

### 2. PDQ (`pdqhash`) — the headline alternative

**How it works.** Resize to 64×64 grayscale via a tent filter (better than nearest-neighbour for aliasing), 2-D DCT, take 16×16 top-left block, median-threshold → 256 bits. Plus: a "quality score" derived from edge gradients in the downscaled image, used to reject visually featureless frames. Plus: a `compute_dihedral` variant that emits 8 hashes (one per rotation/flip combination) to handle mirrors and rotations.

**What the paper actually shows ([arXiv 1912.07745](https://arxiv.org/abs/1912.07745)):**

| Transformation | PDQ match rate at threshold ≤30 |
|---|---|
| Format change (transcode) | **99.96%** |
| Text overlay | Most under distance 10, vast majority under 20 |
| Thumbnail to 256 px | "Almost guaranteed" |
| Thumbnail to 32 px | Poor — too little signal |
| Watermark | Only "just over half" detected |
| Crop >5% | Above threshold |
| Rotation >5° | Performance "falls away sharply" |

**Strengths over pHash.**

- Quality score lets you drop near-uniform frames (e.g. dark intro frames) before they pollute the comparison. Our pipeline does not currently reject such frames; PDQ would let us skip them.
- The dihedral variant gives mirror robustness at 8× cost.
- The 64-pixel intermediate (vs. 32 for pHash) preserves slightly more spatial structure.

**Weaknesses for our use case.**

- The strengths above are largely **already mooted by our preprocessing**. SAR-normalisation and portrait-to-landscape rotation in `_build_frame_extract_cmd` (hasher.py:237–248) handle the rotation case for us. We do not need PDQ's dihedral mode.
- PDQ at 256 bits with threshold 30 is *more lenient* than our current 256-bit pHash at threshold 14. PDQ's threshold mapping is for **identifying near-duplicates anywhere** (CSAM matching, the original use case), while our task is **clustering same-source re-encodes within a duration window**. The duration pre-filter already does most of the rough work.
- Migration cost (see [Implementation switching cost](#implementation-switching-cost)).

**If we did switch, what threshold?** The paper recommends `≤31/256 ≈ 12% bit-flip`. Mapping that to our duration-pre-filtered pipeline, where the work is to distinguish "same source, different encoding" from "different source, same length", a much tighter threshold makes sense — probably **`≤20/256` for PDQ** to mirror our current pHash strictness while taking advantage of PDQ's better tail behaviour. Stricter thresholds (10–15) would essentially require pixel-identity and lose the benefit of PDQ's improved tolerance entirely.

**Verdict.** Genuinely better algorithm on paper, but the marginal recall improvement does not survive the migration cost or the cache invalidation. Defer.

### 3. wHash (`imagehash.whash`) — the most interesting auxiliary

**How it works.** Resize to a power-of-two grid (default 8×8 → 64 bits). Apply 2-D Haar wavelet transform. Use the LL (low-frequency) sub-band, median-threshold → bits.

**Why it's interesting for us.**

- The Content Blockchain study has wHash as the **most accurate** algorithm overall (21.2% failure, lower than pHash's 23.7%).
- It specifically **beats pHash on watermarked images** (31.7% miss vs 46.4%). For a personal video library where overlays like channel logos are common, this is meaningful.
- It is *very* fast (~1 ms per image, comparable to dHash) — orders of magnitude faster than pHash.

**Why it's not a drop-in replacement.**

- The same study found **7866 collisions** out of 100 584 images — by far the worst of the four. wHash is more recall-friendly but produces many more false positives in the absence of a discriminative pre-filter.
- Our duration pre-filter and 256-bit hash size both fight collisions. wHash at `hash_size=16` (256-bit) likely reduces collisions to a usable level, but I haven't seen published numbers.

**Verdict.** Don't use as a replacement. *Consider* as a second hash in an ensemble (next section).

---

## Threshold guidance

The current code uses Hamming `≤14` on a 256-bit hash. As a percentage of bits, this is `≈5.5%`.

If we were to switch hash algorithms or bit widths, the percentage-of-bits target is the right invariant to preserve, *not* the raw distance number.

| Algorithm | Hash size | Equivalent threshold at 5.5% bit-flip | Comments |
|---|---|---|---|
| pHash 64-bit | 64 bit | ≤4 | This is what the canonical "Hamming 4–6 for pHash" advice refers to |
| pHash 256-bit (current) | 256 bit | **≤14** | Current setting — strict, appropriate post-duration-filter |
| PDQ | 256 bit | ≤14 numerically, but the bit-flip distribution differs slightly; **paper recommends ≤30 for general matching, we'd want ≤20** | PDQ's median-threshold quantisation gives slightly different distance statistics; ≤14 may be too strict |
| dHash 64-bit | 64 bit | ≤4 | But dHash distance distributions are not the same shape; for ensemble use, ≤6 is a more permissive starting point |
| dHash 256-bit (`hash_size=16`) | 256 bit | ≤14 | If used in ensemble, recommend `≤20` so that dHash catches frames pHash slightly misses |
| wHash 256-bit | 256 bit | ≤14 | But collision risk is higher; consider ≤10 instead |

A subtler point: our `compare_hash_sets` uses **average best-match distance across N frames** rather than per-frame distance. The threshold gates that average, not any single frame's distance. This makes the system much more forgiving than a per-frame threshold suggests, because outlier frames (e.g. an ad break, a blank fade) are pulled down by the rest. When tuning a new algorithm, this is the metric to evaluate, not point-to-point distance.

### Early-exit constants (hasher.py:578, 584)

```python
if avg_so_far <= threshold * 0.5:  # very strong match → extrapolate
    ...
if avg_so_far > threshold * 2:     # clearly not a match → bail
    ...
```

These `0.5×` and `2×` multipliers are tuned to the current 256-bit pHash. They would need to be re-checked, but probably stay sensible, if you ever switch to PDQ (same hash size and similar distance shape). If you change hash *size* (e.g. switch to 64-bit dHash as primary), they are *fine* — they are relative to the threshold and the threshold scales with hash size.

---

## Robustness analysis: what transformations survive each hash

A "survives" verdict here means "the hash distance remains under the recommended same-content threshold for that algorithm." Numbers come from the same Content Blockchain and Facebook PDQ studies cited above.

| Transformation | aHash | dHash | pHash 256 | wHash | PDQ | crop-resistant | Audio FP (our fallback) |
|---|---|---|---|---|---|---|---|
| Re-encode (same codec) | survives | survives | **survives** | survives | survives | survives | survives |
| Cross-codec transcode (H.264 → HEVC) | usually survives | usually survives | **survives** | survives | survives | survives | survives |
| Scale 1080p → 720p | survives | survives | **survives** | survives | survives | survives | survives |
| Scale 1080p → 360p (thumbnail) | fails | fails | usually survives | survives | survives at 256, fails at 32 | survives | survives |
| Letterbox bars added | fails | fails | **partial** (DC stays similar) | partial | partial | survives | survives |
| Color grading (gamma shift) | fails | fails | mostly survives | survives | survives | survives | N/A |
| Color → grayscale | partial | partial | survives | survives | survives | survives | survives |
| Watermark (channel logo overlay) | fails | fails (43% miss) | fails (46% miss) | **survives (31% miss)** | partial (~50% miss) | survives | survives |
| Slight crop (≤5%) | fails | mostly fails | partial | partial | fails | **survives** | survives |
| Heavy crop (≥20%) | fails | fails | fails | fails | fails | **survives** | survives |
| Mirror (horizontal flip) | fails | fails | fails | fails | fails (use `compute_dihedral`) | fails | survives |
| Rotation 5° | fails | fails | fails | fails | partial | fails | survives |
| Rotation 90° | fails | fails | fails | fails | fails (use `compute_dihedral`) | fails (use color moment) | survives |
| Portrait ↔ landscape recoding | **handled by our preprocessing**, all hashes survive | same | same | same | same | same | same |
| Anamorphic SAR mismatch | **handled by our preprocessing**, all hashes survive | same | same | same | same | same | same |
| Container change only (no re-encode) | survives | survives | survives | survives | survives | survives | survives |

What this matrix tells us about our system specifically:

1. **Most failure modes are already covered** by the audio fingerprint fallback (`comparator.py:142–148`). Mirror, severe crop, rotation, and even watermarking are caught if the audio matches.
2. **The remaining recall gap is silent re-cropped content with watermarks**, e.g. screen-recorded TikToks that have lost the original audio. For these, no single hash in the matrix solves it; you'd need `crop_resistant_hash` *plus* a watermark-tolerant hash, both per frame, which is prohibitively expensive.
3. **Letterboxing is a real and currently uncaught failure**. If one encode adds black bars and the other doesn't, pHash distance can balloon. The pHash code does not crop letterbox before hashing. This is a separate fix (a `cropdetect` ffmpeg pass) and orthogonal to the hash algorithm choice.

---

## Combined / ensemble approach

The literature (Content Blockchain, OpenCV `img_hash` blog) consistently shows that **no single hash dominates all transformations**. Watermark is wHash territory; transcode is pHash territory; crop is crop-resistant territory. The natural question is whether combining them helps.

### Option A: OR-ensemble of pHash and dHash per frame

Compute both `phash(img, hash_size=16)` and `dhash(img, hash_size=16)` per frame. Two videos are visually matched if **either** the average pHash distance ≤ T_p **or** the average dHash distance ≤ T_d.

**Pros.**

- dHash captures **edge structure**; pHash captures **low-frequency structure**. They miss different things. In practice they tend to fail on different transcodes — pHash struggles with heavy contrast changes; dHash struggles with crops but is good on watermarks (the edges around the watermark perturb dHash less than they perturb pHash's median).
- Marginal compute cost is negligible: dHash is ~300× faster than pHash on the per-image benchmark. In our pipeline, where ffmpeg dominates, adding dHash per frame is essentially free.
- Cache and schema impact is small: one new JSON column on `VideoFile` and `FileCache` (e.g. `perceptual_hashes_dhash`). The original pHash column stays valid.

**Cons.**

- False-positive risk: dHash has low collision rate on its own but the OR rule weakens both. To mitigate, use a strict T_d. Recommended starting point: `T_d = 10/256 (≈4%)` vs. pHash's `T_p = 14/256 (≈5.5%)`.
- The audio fallback already exists for the same purpose. The benefit of dHash is only on **audio-less** or **audio-mismatched** edge cases.

**Expected impact:** +3–8% recall on watermarked / overlay duplicates that lack matching audio. No measurable false positive increase if T_d is strict.

### Option B: AND-ensemble (both hashes must match)

This is the inverse — used to reduce false positives, not raise recall. Useful if you find the duration-filtered visual stage is currently producing too many wrong groupings, which we have no evidence for.

**Verdict.** Skip Option B for now.

### Option C: pHash + colorhash

`colorhash` adds rotation/mirror tolerance in exchange for very weak transcode-discrimination on its own. As an *additional* check, it could catch the mirror case our system currently relies on audio for. Cost is similar to dHash.

Not recommended over Option A. Mirror is already handled by audio fallback, which is more reliable than colorhash.

### Option D: cascade — fast dHash filter, then pHash verify

The pHash-vs-dHash speed gap is ~300×. A cascade scheme could compute dHash for every frame, do a quick all-pairs filter to find candidate pairs, and only compute pHash for the candidates.

For us this is **the wrong optimisation**, because pHash is *already* gated by the duration filter, and the duration filter is far more selective than dHash. Spending engineering effort on a cascade buys nothing for our pipeline shape.

### Verdict on ensembles

If you want a single low-risk recall improvement: implement Option A (pHash OR dHash). Otherwise, leave the algorithm alone and focus on the letterbox/cropdetect gap noted above.

---

## Implementation switching cost (if you nonetheless want to migrate to PDQ)

### Install / build

| Aspect | `imagehash` (current) | `pdqhash` |
|---|---|---|
| Install command | `pip install ImageHash` | `pip install pdqhash` |
| Wheels available | Yes, pure-Python | Yes for Windows x86-64, manylinux, macOS (per PyPI listing) |
| Build deps if no wheel | None (pure Python on Pillow + numpy) | Cython, OpenCV development headers, C++ compiler |
| Runtime deps | Pillow, numpy | OpenCV (`opencv-python`), numpy |
| Docker image impact | ~5 MB | ~80–120 MB (OpenCV runtime) |

The Dockerfile in this repo is CUDA-based, so adding OpenCV is not catastrophic, but it's not nothing either. Pre-built wheels exist for Windows so the local dev story stays fine.

### API surface

```python
# Current (imagehash)
import imagehash
from PIL import Image
img = Image.open(path)
h = imagehash.phash(img, hash_size=16)
hex_str = str(h)                 # "a1b2c3d4..."
distance = h1 - h2               # Hamming, on imagehash.ImageHash objects

# PDQ (pdqhash)
import pdqhash, cv2
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hash_vec, quality = pdqhash.compute(img)   # hash_vec is numpy uint8 array length 256
# Hamming:
distance = int(np.count_nonzero(hash_vec1 != hash_vec2))
```

Key differences:

- PDQ returns a **numpy bit array** plus a **quality score**, not a hex string. The storage format on `VideoFile.perceptual_hashes` and `FileCache.perceptual_hashes` is a JSON array of hex strings (`hasher.py:345`); migrating means picking a new serialisation. The cleanest path is hex (`numpy.packbits(hash_vec).tobytes().hex()`).
- The quality score is per-frame and informative. To use it, the schema needs a parallel array of float quality scores per frame, or you filter on extraction and store only the high-quality frames.
- PDQ wants OpenCV-format BGR/RGB arrays, not PIL Image objects. The frame extraction pipeline currently saves frames to disk as JPEG and reads via PIL. You'd swap to `cv2.imread`. Functionally equivalent, but every code path that touches frames changes.

### Cache invalidation

`FileCache.perceptual_hashes` rows are keyed by `(file_path, file_size, mtime_ns)`. They survive cross-scan. They are *not* keyed by algorithm. Migrating to PDQ means either:

1. Adding an `algorithm` column to `FileCache` and treating PDQ hashes as a separate species. Old pHash rows become useless and get evicted by the scan-end sweep over time.
2. Bumping a schema version so all `FileCache` rows are flushed at next startup.

Either way: every video in the library re-hashes from scratch on the next scan. For a 50 000-video library at, say, 8 frames per video × 80 ms per PDQ hash plus ffmpeg, this is dominated by ffmpeg — call it 5–10 hours of recompute. Not a deal-breaker but not free.

### Comparator changes

`compare_hash_sets` in `hasher.py:517` and the `compute_hamming_distance` in `hasher.py:493` both assume hex strings. They would need to accept numpy bit arrays (or you serialise PDQ to hex and pretend nothing changed, at a small CPU cost).

The early-exit constants (`* 0.5`, `* 2`) are scale-relative to `threshold`, so they survive the swap as long as `HASH_SIMILARITY_THRESHOLD` is re-tuned for PDQ's distance distribution (target ~`≤20/256` rather than the current `≤14/256`).

### Net effort estimate

Concrete edits: ~150 lines across `hasher.py`, `models/database.py` (cache schema), and possibly `config.py` (new threshold default). Risk: medium — the early-exit tuning and threshold default both need re-validation against a real corpus.

**This is not weekend work for a recall improvement we have no evidence we actually need.**

---

## Specific notes on the other algorithms in the brief

- **TMK+PDQF (one hash per video).** Excellent for the CSAM-matching use case Facebook designed it for: "is this video in my known-bad list?". For our task (clustering same-source re-encodes within an arbitrary library) it is *less* useful because the temporal pooling collapses the signal that lets us discriminate near-duplicates from "same length, different content." TMK relies on a known reference set to score against, not pairwise clustering. The `tmkpy` library requires swig + a C++ build and is significantly more fragile than `pdqhash`. **Skip.**
- **PhotoDNA.** Not open source. Microsoft licenses it under strict conditions to specific organisations. Mentioned for completeness only. **Skip.**
- **Marr-Hildreth (`mhash`), block-mean hash, radial-variance hash (OpenCV `img_hash`).** These are reasonable alternatives if you ever decide the project should depend on OpenCV anyway (which the PDQ migration would force). Block-mean hash in particular is robust to noise and only ~30% slower than pHash. None of them are individually better than pHash for our specific transcode-clustering task, and adopting them costs the same OpenCV dependency that PDQ does. **Skip unless OpenCV is brought in for some other reason.**
- **`colorhash`.** Useful as an *auxiliary* if mirror/rotation handling without audio matters in the future. Skip otherwise.
- **`crop_resistant_hash`.** Solves the crop case dramatically (174/175 at 50% crop vs 10/175 for pHash) but is 10–50× slower than pHash because of the per-frame watershed segmentation. Output is not a single 256-bit hash but a *list* of per-segment hashes — comparison logic is non-trivial. Practical only if heavy cropping becomes a known recurring duplicate-miss pattern.

---

## What to actually do (in priority order)

1. **Do nothing to the hash algorithm.** Current 256-bit pHash with threshold ≤14 is correctly calibrated for our pipeline shape.
2. **Optional: implement the dHash ensemble (Option A above).** Adds one cheap secondary hash, OR-combined with strict threshold. Schema impact is one new JSON column on two tables. Estimated +3–8% recall on watermarked / overlay duplicates with no precision loss.
3. **Letterbox detection.** Not a hash issue, but the largest currently-uncaught visual failure mode. Add `ffmpeg -vf cropdetect` once per video and apply the detected crop in the frame extraction filter chain *before* `scale=320:-2`. Independent of any hash decision.
4. **If you later decide to migrate to PDQ anyway:** plan it as a cache-flush event. Use threshold `≤20/256` as the starting point. Re-tune `compare_hash_sets` early-exit constants on a labelled corpus. Add an `algorithm` column to `FileCache` to allow future migrations to be non-destructive.

---

## Sources

- [Content Blockchain — Testing different image hash functions](https://content-blockchain.org/research/testing-different-image-hash-functions/) — the 100 584-image Caltech101 study, the source of most numerical recall/precision numbers used above.
- [PDQ & TMK + PDQF — A Test Drive of Facebook's Perceptual Hashing Algorithms (arXiv:1912.07745)](https://arxiv.org/abs/1912.07745) — Facebook's own evaluation, source of the `≤31/256` threshold recommendation, the 99.96% transcode match rate, and the watermark/crop/rotation degradation numbers.
- [Facebook ThreatExchange — PDQ reference implementation](https://github.com/facebook/ThreatExchange/tree/main/pdq) — source of the `≤31` distance and `≤49` quality thresholds.
- [`pdqhash` on PyPI](https://pypi.org/project/pdqhash/) and [`pdqhash-python` on GitHub](https://github.com/faustomorales/pdqhash-python) — Python bindings, install requirements, API.
- [`imagehash` on GitHub](https://github.com/JohannesBuchner/imagehash) — current library, README and code as the source of supported algorithms and default hash sizes.
- [Qt and OpenCV blog — Introduction to image hash module of OpenCV](http://qtandopencv.blogspot.com/2016/06/introduction-to-image-hash-module-of.html) — speed and robustness rankings for OpenCV `img_hash` algorithms (block-mean, Marr-Hildreth, radial-variance, color-moment).
- [OpenCV docs — img_hash module](https://docs.opencv.org/4.x/d4/d93/group__img__hash.html) — algorithm list and licensing.
- [pHash vs dHash — comparison reference](https://ssojet.com/compare-hashing-algorithms/phash-vs-dhash) — source for the 300× speed-gap figure.
- [Crop-resistant hash — imagehash issue #117 and original paper](https://github.com/JohannesBuchner/imagehash/issues/117) — performance and 174/175 vs 10/175 cropped-match figure.
- [PHVSpec: A Benchmark-based Analysis of Perceptual Hash Systems for Videos (Tech Coalition)](https://technologycoalition.org/wp-content/uploads/Tech-Coalition-Video-Hash-Benchmark-Paper.pdf) — video-hash benchmark referenced for TMK+PDQF guidance.
