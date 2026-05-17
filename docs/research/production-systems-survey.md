# Production Systems Survey: Video Duplicate Detection

How real production systems handle video-duplicate detection, with a relentless
focus on what is and isn't applicable to a single-machine desktop app scanning
10k-50k local files (occasionally up to 1M).

This document deliberately does **not** repeat the algorithmic, caching, or
pipeline-internal recommendations already covered in:

- `docs/research/algorithmic-improvements.md` — BK-tree, CLIP/DINOv2, Chromaprint.
- `docs/research/caching-incremental.md` — `(file_path, file_size, mtime_ns)` cache, SHA-256 fast path.
- `docs/research/pipeline-optimizations.md` — audio FP sampling, tiered extraction, parallel stages, separate semaphores.

Scope here is **what production systems publish, what their open-source spinoffs
look like in 2026, and which of their techniques we should consider lifting**.

---

## Executive summary

The single most important thing you can lift from production systems is **one
hash per video, not twelve**. Every system at YouTube/Meta/TikTok scale does
this. Twelve frame-hashes per video plus a within-bucket O(n²) best-match
verifier is what you build for tens of thousands of files; one short
fixed-length descriptor per video plus a vector index is what you build when
you have to win. The desktop app is at the boundary where one-hash-per-video
becomes cheaper *and* more accurate than the current 12-frame approach. The
existing 12-frame extraction can remain — it just feeds a pooling step rather
than a 144-cell pairwise verifier.

Top 3 ideas borrowed from production systems that fit this app:

1. **vPDQ-style "video as a bag of frame PDQ hashes" with hash-count matching.**
   Facebook's open-source vPDQ produces a variable-length list of 256-bit PDQ
   hashes per video and declares a match when ≥ N hashes match within a
   per-hash threshold. This is *exactly* the current pipeline's shape (12
   hashes × 256 bits) with two changes: (a) replace `imagehash.phash`
   (`hash_size=16`) with `pdqhash.compute()` (also 256 bits, better
   calibration), and (b) replace best-match-greedy with PDQ's "count of
   matching frames" rule. This is a 1-day port, drop-in compatible with the
   existing cache schema, and provides the published Meta calibration tables
   for false-positive rate as a function of Hamming distance — something the
   current implementation has *guessed* with `HASH_SIMILARITY_THRESHOLD=14`.
2. **TMK+PDQF-style single-fixed-length-vector-per-video for the index step.**
   This is the technique that lets Meta search billions of videos. Single
   ~256 KB float vector per video, cosine-similarity comparison (one dot
   product per pair). For a desktop with 50k–1M files this collapses the
   compare stage from "duration-grouped O(n²) Hamming" to "one FAISS Flat-IP
   query per video". Worth doing **only** if the catalogue passes ~200k files
   or the comparator becomes the wall-clock bottleneck.
3. **Chromaprint for audio (already covered in algorithmic-improvements.md
   §4), wired here for completeness.** Every audio dedup product on the
   internet — MusicBrainz Picard, AcoustID, beets — uses it. The current
   64-point RMS profile is roughly what you'd build in an afternoon; this is
   roughly what 15 years of open-source audio-fingerprinting work converged on.

What we should **not** copy from production: spectro-temporal CNNs trained on
1M-video corpora (YouTube), distributed inverted indexes over 600 M SIFT
features (the 3rd-place ISC2021 solution), GPU-FAISS clusters of 100+ machines
(ByteDance). These are correct for billion-scale and overkill for 50k.

Biggest risk in adopting any of this: PDQ and TMK have published calibration
curves *for image-policy enforcement*, not *consumer dedup*. Threshold has to
be re-tuned for "I want to free up 200 GB of duplicate vacation videos" vs "is
this a known CSAM image?" — the operating point on a precision-recall curve is
in a different place.

---

## Per-system writeups

### 1. Facebook / Meta — the only company that fully open-sourced

Meta is the gold standard for "we built it for our T&S team and put the code
on GitHub." Three relevant open-source releases:

- **PDQ** — image perceptual hash, 2019. 256-bit, DCT-derived (broadly
  pHash-lineage), accompanied by a published quality score per hash (0-100)
  for "is this hash informative or just gradient-noise on a flat background".
  License: BSD.
- **TMK+PDQF** — video-level hash, 2019. Computes a per-frame "PDQF" feature
  (floating-point analogue of PDQ, ~256-d per frame), then applies the
  **Temporal Match Kernel** to compress an entire video into a *single*
  fixed-length vector regardless of video length. Match by inner product.
  Recommended threshold: 0.7.
- **vPDQ** — video as a bag of per-frame PDQ hashes, ~2020. Annotates each
  hash with `(frame_number, quality, timestamp)`. Match by counting how many
  hashes in video A have a Hamming-near-match in video B (and vice versa).
  Variable-length; not friendly to a single ANN index, but trivial to slot
  into a duration-grouped O(n²) comparator like the current one.

All three live in **`facebook/ThreatExchange`** alongside the **Hasher-Matcher-
Actioner (HMA)** reference deployment, which is the AWS deployable for content
moderation pipelines.

**Python libraries (status May 2026):**

- `pdqhash` (PyPI): Python bindings for PDQ, maintained by faustomorales,
  latest release May 2025. Supports Python 3.9-3.12. `pip install pdqhash`.
  Compatible API: `pdqhash.compute(img)` returns a 256-bit hash + quality.
- `vpdq` (PyPI): Python bindings for vPDQ, requires FFmpeg installed for
  frame extraction. Same authorship lineage.
- `darwinium-com/pdqhash`: a Rust-port of the Python PDQ binding that is
  used by Darwinium in production fraud-detection. Useful as a reference for
  what a hardened PDQ implementation looks like.

**Calibration data Meta has published:**

| Hash type | Recommended match threshold | Source |
|---|---|---|
| PDQ (image) | Hamming distance ≤ 31 / 256 (= 88% similarity) | ThreatExchange docs |
| TMK+PDQF (video) | Cosine similarity ≥ 0.7 | Dalins et al. 2019 |
| vPDQ (video frames) | Per-frame PDQ ≤ 31, plus a count threshold | ThreatExchange docs |

The PDQ threshold of 31/256 is **much more lenient** than this project's
current `HASH_SIMILARITY_THRESHOLD=14`. This is because Meta's operating point
optimises for "catch as many copies as possible, false positives are reviewed
by humans" while this app optimises for "never delete a non-duplicate."
Translation: borrowing PDQ as the hash function is fine, but keep the
existing threshold of 14 (or tighter) for the consumer-dedup use case.

**The hashing.pdf in ThreatExchange** (raw URL:
`https://raw.githubusercontent.com/facebook/ThreatExchange/main/hashing/hashing.pdf`)
is the canonical engineering write-up. It explains:

- PDQ frame resampling normalisation (256×256 grey → 16×16 DCT → median-bit
  hash, same idea as pHash but the DCT block selection and median-bit
  derivation differs slightly). Net result: more discriminative than
  imagehash's phash for natural images.
- PDQF: the same pipeline up to the DCT, but keeping the *floating-point*
  coefficients instead of binarising. Output is a 256-d float vector, used
  as input to TMK.
- TMK: a temporal kernel that takes a sequence of PDQF frame-vectors and
  outputs a single fixed-length vector capturing the video's spectro-temporal
  structure. Hardcoded to **15 fps resampling** before TMK runs.
- TMK comparison: dot product of two TMK vectors; values ≥ 0.7 are matches.

**The AiLECS test drive paper** (Dalins, Wilson, Boudry 2019, arXiv 1912.07745)
is the only published independent evaluation of these algorithms. Headline
results on real CSAM investigation data: PDQ achieved high recall on
recompression / resize, lower recall on heavy cropping. TMK was effective for
"same video, different encoding" but degraded on partial-copy scenarios where
only 30s of a 5-min source appears in a longer derivative.

### 2. YouTube / Content ID — almost everything is proprietary

Public details about Content ID architecture are minimal and largely from
journalism, not engineering blogs. What is known:

- **Both audio and video fingerprinting are run.** Audio matching is the
  primary signal for music-copyright cases; video fingerprinting catches
  visually-modified re-uploads where audio has been replaced.
- **Video fingerprinting slices each upload into thousands of frames** and
  compares each frame's "fingerprint" against the reference library
  (FastCompany 2017, Streaming Media 2024). This is consistent with a
  bag-of-frame-hashes architecture like vPDQ. The exact algorithm is not
  published.
- **Reference library is curated by rightsholders.** Content owners upload
  reference files; user uploads are matched against this index, not against
  each other. This is a critical architectural difference — *YouTube does not
  solve "find all duplicates in a corpus", it solves "find matches against a
  small curated reference set"*. Sub-second matching against billions of
  references is plausible because the *reference* set is "billions" but each
  upload only triggers an ANN lookup against the references, not all-pairs.
- **Audio fingerprint approach is undocumented**, but Content ID's behaviour
  (catches pitched-up, sped-up, partial covers) suggests something far more
  sophisticated than Chromaprint — likely chroma-vector + temporal
  alignment + a learned model. Outside the budget of any open-source project.

**Practical take-away:** Content ID's "match against curated references" is
not the same problem as "find all dupes in a folder." Don't try to copy it.
The one borrowable idea is that **audio and video fingerprints are computed
independently and OR'd at match time** — which is already what this project's
comparator does.

### 3. TikTok / ByteDance — published less, scale similar to YouTube

ByteDance has published a number of recommendation-system papers but
relatively few specifically on dedup. Closest published work:

- **Fast Video Deduplication and Localization With Temporal Consistence Re-
  Ranking** (IEEE TCSVT 2024, M. Yuan et al.) — describes a two-stage
  pipeline: (1) coarse retrieval with a compact per-video descriptor, (2)
  fine-grained re-ranking with temporal consistency. F1 of ~77% on VCDB.
  Uses a learned descriptor.
- **SVD: A Large-Scale Short Video Dataset for Near-Duplicate Video Retrieval**
  (Jiang et al., ICCV 2019) — Beijing Univ + Alibaba, but TikTok-scale-relevant.
  Contains 500k short videos with ground-truth near-duplicate labels.
- **Fast and Robust Video Deduplication** (ACM Mile-High Video, 2023) — a
  Tencent-affiliated paper but representative of the wider Chinese
  industry approach: short-video-focused, MobileNet-class CNN per frame, ANN.

ByteDance's actual *production* approach is not published. Best-guess from
hiring posts and conference talks: they use a custom CNN-derived embedding
(probably MobileNetV3-class for the on-device side, ResNet50-class on the
server) with FAISS or a proprietary equivalent. Their recommendation system
infrastructure handles the dedup as a downstream consumer of the embedding,
not as a separate pipeline.

**Practical take-away:** the SVD benchmark is small enough (~500k items) that
papers reporting numbers on it are directly comparable. A learned descriptor
is the right answer at TikTok scale; at 50k-file desktop scale, a
PDQ-or-DINOv2 embedding hits diminishing returns on the embedding side and
moves the bottleneck back to frame extraction.

### 4. Dropbox — file-level only, but the chunking idea is borrowable

Dropbox is the canonical implementation of **content-defined chunking with
rolling hashes**, originally Rabin fingerprints, now various successors
(buzhash, FastCDC). The mechanic:

1. Sliding window of ~64 bytes over the file.
2. Rolling hash of the window content.
3. When `hash mod 2^N == 0` (for some N around 13-22), cut a chunk boundary.
4. Each chunk gets a SHA-256.
5. Server stores chunks by hash; the file is a list of chunk hashes.

This deduplicates **at the byte level** across files and resists insertion/
deletion shifts. Inserting one byte at the start of a file does *not*
re-hash every subsequent chunk, because the rolling-hash cut-points are
content-defined: the same byte patterns produce the same cuts. Only the
chunk containing the inserted byte changes.

**Why this is interesting for video dedup:**

- Re-encoded videos do **not** share byte-level chunks with their sources
  (entirely different bit streams). CDC offers nothing here.
- *Identical files copied to multiple locations* do share every chunk, but
  so does a full-file SHA-256, more cheaply.
- *Identical files with slightly-modified metadata* (mp4 atom shuffles,
  changed title) **do** share most chunks. A 1-byte metadata edit causes one
  chunk-mismatch out of hundreds. This is a real edge case in user libraries.

**Practical take-away:** content-defined chunking is the wrong tool for
*duplicate-detection*. It is the right tool for *change detection / delta
sync*. For this app, a full-file SHA-256 (already in `caching-incremental.md`
§6) catches the exact-duplicate case cheaper. CDC would only matter if the
app grew a "sync a deduplicated library to the cloud" feature, which is well
outside scope.

References:

- FastCDC paper: USENIX ATC 2016 (Xia et al.). Faster than Rabin, same
  recall.
- Restic uses content-defined chunking for backup dedup.

### 5. Google Photos — the silence is the answer

Google publishes very little on its consumer dedup pipeline. From the support
docs and field reports:

- **Exact-byte duplicates are detected at upload time** (file hash).
- **Near-duplicates are not detected at all** (no automatic merging of a
  re-encoded copy, no auto-cluster of HDR/SDR pairs).
- The user-facing "Library → Trash" flow has no concept of duplicate groups
  for video.

This is a deliberate product decision: Google has near-infinite storage and
no incentive to dedup user-visible items. The dedup probably happens *at the
storage layer* (Colossus block-store has block-level dedup for free) but is
invisible to the user.

**Practical take-away:** Google's choice tells you that *consumer* near-
duplicate-video detection is a "nice-to-have" feature, not a solved problem
even at Google scale. The product space this app inhabits is small,
underserved, and has no established UX standard. Free reign on the UX side;
nothing to copy on the algorithm side.

### 6. Apple Photos — the HDR/SDR + Live Photo special case

Apple Photos detects:

- **Exact duplicates by file data** (Library → Utilities → Duplicates album,
  introduced iOS 16).
- **Live Photo's still+video pair** is detected at *import* time, not at
  scan time: when an iPhone uploads a `IMG_1234.HEIC` and `IMG_1234.MOV`
  together, Photos clusters them by filename pattern. If you import them
  separately, you get a still and a video, not a Live Photo. **This is the
  only "near-duplicate" matching Apple does, and it's filename-based.**
- **HDR/SDR pairs** (when the camera has "Keep Normal Photo" enabled) are
  *not* detected as duplicates by Apple's own UI; users complain about
  this on the forums. Apple presumably treats the SDR version as the
  "duplicate" via filename + EXIF capture-time, but does not surface it.

**Practical take-away:** Apple Photos doesn't try harder than exact-byte
matching for the same reason Google doesn't. The HDR/SDR pair detection is
an interesting *product* case but **technically trivial**: filename pattern
(`IMG_xxxx` + `IMG_Exxxx`) plus EXIF capture-time match. Not worth a special
case for this app, which targets a different problem (re-encodes of the
same content across messy filesystems, not paired captures from the same
device).

### 7. Open-source competitors — which are alive in 2026

| Project | Stack | Status May 2026 | Algorithm | Worth porting? |
|---|---|---|---|---|
| **Czkawka** | Rust | Active, v11.0.1 Feb 2026 | pHash-style per-frame, hash-based comparison | Mature but no SOTA |
| **VideoDuplicateFinder (0x90d)** | C# / .NET | Active | "Spatial+temporal" pHash on first-60s frames | Direct competitor; algorithm published in repo |
| **vid_dup_finder / vid_dup_finder_lib** | Rust | Active (Farmadupe) | Spatial + temporal pHash, configurable tolerances | Interesting Rust library to reference |
| **videohash (akamhy)** | Python | Last release 2022, low activity in 2025 | Collage of all 1-fps frames + wavelet hash → 64-bit | Worth comparing for "single short hash per video" approach |
| **video-simili-duplicate-cleaner** | Qt/C++ | Maintenance | pHash on sampled frames | Niche |
| **dupeGuru** | Python | Active for files/music/images, not video | Filename + content for files, audio fingerprint for music, no video-specific logic | Skip for video |
| **PhotoStructure** | Node | Active (v2026.1, big rewrite) | Photos focused; multi-perceptual-hash combo for fuzzy | Different problem |

#### Czkawka deep-dive

Czkawka is the most-watched dedup project in this space (>15k stars). Its
**similar videos** scanner uses the same pHash-on-frames approach as this
app. The maintainer (qarmin) has acknowledged on the issue tracker that
similar-videos detection has higher false-positive rates than similar-images,
which mirrors this project's choices around audio-fallback. Recent (2025-26)
work has focused on parallelism and cache caching, not algorithm changes.

Borrowable lessons:

- They use the **`img_hash`** Rust crate for hashing. Less battle-tested than
  PDQ but vectorised in Rust.
- Their "similar videos" performance issue thread (#1749) cites the same
  pain points: noisy results from duration-only grouping, hash collisions on
  black-frame intros. This validates the priorities in
  `algorithmic-improvements.md` §5.
- Krokiet (the modern Czkawka frontend) is Slint-based; not a useful
  reference for a React frontend.

#### VideoDuplicateFinder (0x90d) deep-dive

The .NET app that most people Google when they want a "video duplicate
finder." Its README documents the algorithm explicitly:

1. Extract several frames from the first 60 seconds.
2. Convert each frame to a 32×32 grayscale array.
3. Use a "spatial + temporal" combined hash:
   - **Spatial** = pHash of each frame (bright/dark spatial pattern).
   - **Temporal** = bright/dark *changes* between consecutive frames.
4. ScanEngine performs parallel pairwise comparison with a configurable
   algorithm.

**Critically**, the temporal channel adds robustness against re-encodes
where the spatial pHashes drift due to compression artefacts but the
inter-frame *delta* patterns remain stable. This is a free improvement over
spatial-only pHash. Cost: one extra hash per frame-pair stored.

Borrowable: **adopt the temporal-delta hash as a second column alongside the
existing per-frame pHashes**. Doubles the per-video hash storage (already
trivial — 12 hashes × 256 bits = 384 bytes; doubling is still under 1 KB)
and adds a numpy frame-subtraction per pair (negligible). The match rule
becomes "spatial OR temporal pHash within threshold", strict-OR semantics
matching the existing OR rule between video and audio.

Risk: increases false-positive rate. Mitigation: require **both** spatial
and temporal to pass for a high-confidence match, OR-fall-back only to a
stricter threshold.

Effort: ~half a day after the existing 12-frame extraction lands its
frames. The temporal channel is `frame[i+1] - frame[i]` in greyscale,
re-hashed.

#### vid_dup_finder_lib deep-dive

This is the Rust *library* that implements the same spatial+temporal idea
(distinct project from VideoDuplicateFinder; uses the same conceptual
algorithm). What's interesting:

- Public API exposes **independent tolerances** for spatial vs temporal.
  Useful for diagnostic / debugging — the desktop app already has a
  diagnose-pair tool; the analogue would be "this pair matched on spatial
  but not temporal: probably a different recording of the same scene."
- The library *ships its own ffmpeg dispatcher* in Rust, which makes it
  cross-platform with less work than Python. Not directly portable but
  validates that ffmpeg subprocess wrangling is the universal pain point.

#### videohash (akamhy) deep-dive

Different philosophy: produces a **single 64-bit hash per video**, period.
Pipeline:

1. Extract 1 frame per second.
2. Resize each to 144×144.
3. Build a **collage** of all frames in a grid.
4. Compute a **wavelet hash** (`whash`) of the collage.

Result: 64-bit hash per video. Two same-content videos hash to identical or
near-identical 64-bit strings; Hamming distance on those is the comparison.

Pros: O(n) comparison after hashing, naturally indexable in any kv store.
Cons: 64 bits is too few for a corpus over ~10k items (birthday-paradox
collisions become routine), and the wavelet-of-collage step is non-standard
— no published calibration data exists. Last meaningful release in 2022.

**Not recommended** for adoption, but useful as a reference point for "what
does the simplest possible single-hash-per-video pipeline look like."

### 8. Hash-based content addressing in general

Quick survey because the prompt asked.

- **IPFS** uses SHA-256 + Merkle DAG (`multihash`) for content addressing.
  Every file gets a `CID` (content identifier). Two files with the same CID
  are byte-identical; no near-dup logic.
- **BitTorrent BEP-30** introduces "Merkle hashes" for piece verification.
  BEP-19 (not 19; BEP-19 is WebSeed) is unrelated to dedup.
- **Git uses SHA-1 / SHA-256 for blobs** with no near-dup logic; just exact
  match for content addressing.

**Practical take-away:** content-addressing hashes are exactly equivalent to
the SHA-256 fast path already proposed in `caching-incremental.md` §6. The
"same file under different name" case is solved by a full-file SHA-256, period.
No production system uses a fancier hash for this case because none is
needed.

### 9. Academic benchmarks

| Benchmark | Year | Scale | What it tests | Top score / method |
|---|---|---|---|---|
| **VCDB** (Fudan/Columbia) | 2014 | 100k web videos + 9k copy-pairs | Partial copy detection | F1 ~77% (TCSVT 2024 paper) |
| **VCSL** (segment-level VCDB) | 2022 | Larger, segment-level | Segment-level localisation | F1 ~67% (segment) |
| **FIVR-200K** | 2019 | 225k videos | Fine-grained instance retrieval (DSVR/CSVR/ISVR) | Various |
| **SVD** | 2019 | 500k short videos | Near-duplicate retrieval | Various |
| **CC_WEB_VIDEO** | 2007 | 13k videos | Classic near-duplicate | mAP > 0.95 with modern methods |
| **DISC2021** (ISC2021) | 2021 | Image-level (not video) | Image copy detection | SSCD: μAP 0.66 |
| **VSC2022** (Meta) | 2022 | Video, billion-scale derived | Video similarity descriptor + matching | 1st place: contrastive self-supervised |

**Key signal:** modern (2022-2024) SOTA on real benchmarks like VCDB sits at
**F1 ≈ 75-80%** for partial-copy detection. The *near-duplicate* problem this
app targets (full re-encodes, same content) is **strictly easier** than what
those benchmarks measure (partial copies, mashups). On the easier sub-problem,
existing pHash-based methods are likely already at F1 > 90%, and the SOTA
benchmarks aren't the right yardstick.

The **VSC2022** winning solutions all converged on:

- Self-supervised contrastive learning (SimCLR / DINO-style training).
- **Per-frame image embeddings**, sampled at 1 fps from videos.
- Pooled into a per-video descriptor.
- FAISS for retrieval.

This is **exactly the architecture proposed in
`algorithmic-improvements.md` §3** (DINOv2 image embeddings + mean-pool +
FAISS). The desktop-app analogue is more conservative on model size but
identical in shape.

---

## Algorithm comparison table

| Algorithm | Library | License | Used in production by | Per-video output | Time complexity (compare) | Scale tested | VCDB-class score | Notes |
|---|---|---|---|---|---|---|---|---|
| **pHash (DCT)** | `imagehash` (PIL/py) | MIT | This app, Czkawka, VideoDuplicateFinder | 12 × 256 bits | O(n² × 144) within bucket | 10k-100k | F1 ~60-70% on partial-copy benchmarks | Current baseline |
| **PDQ** | `pdqhash` (py) | BSD | Meta, Darwinium | 1 × 256 bits per frame | Same shape as pHash | 100M+ (Meta) | Slightly better than pHash on cropped/recoloured | Drop-in replacement |
| **vPDQ** | `vpdq` (py) | BSD | Meta | Variable: bag of (PDQ, frame#, quality, timestamp) | O(n²) within bucket | 100M+ | Same as PDQ at frame level + temporal count | Best fit for variable-length video |
| **TMK+PDQF** | `ThreatExchange/hashing` (C++) | BSD | Meta | 1 × ~256-d float vector per video | O(n) per query w/ index | Billions (Meta) | Strong on full-copy, weaker on partial | Single fixed-length hash per video |
| **videohash (collage+whash)** | `videohash` (py) | MIT | None notable | 1 × 64 bits | O(n) | Low | Not benchmarked publicly | Too short for >10k items |
| **Spatial+temporal pHash** | VideoDuplicateFinder source | MIT | Hobbyist projects | 2 × 256 bits per frame | O(n² × 144) within bucket | 10k-100k | Better than pHash alone | Adds delta-frame channel |
| **DINOv2 mean-pool** | `dinov2` (pytorch hub) | Apache 2 | Research, growing | 1 × 384-d float | O(n) per query w/ FAISS | 100M+ in research | F1 ≈ SOTA on near-dup | Needs GPU; 5ms/image |
| **SSCD ResNet50** | `sscd-copy-detection` (py) | MIT | Meta (T&S) | 1 × 512-d float | O(n) per query | Image: 1M, video: research | DISC2021 μAP 0.66 (image) | Image-only; needs pooling for video |
| **VSC2022 winner** | `vsc2022` (py) | MIT | None outside Meta | 1 × 512-d float per video | O(n) | VSC2022 benchmark | VSC2022 winner | Heavy training; production-grade |
| **Chromaprint** | `pyacoustid` (py) | LGPL | MusicBrainz, beets, AcoustID | Variable int32 array | O(n × m) per pair, vectorisable | Millions (AcoustID) | Audio-only; very high recall | Replaces 64-pt RMS |
| **TF-IDF on Chromaprint** | `pyacoustid` + custom | LGPL | Audio research | Bag-of-tokens | O(log n) w/ index | Millions | Higher precision for sub-clip matching | Optional refinement |

---

## Applicability to a 50k-file desktop app

This is where most of the production-systems literature breaks down. The
50k-file local scan has constraints that web-scale systems don't:

1. **Single machine.** No distributed inverted indexes. A 100-node Spark
   cluster is not on the table. Everything has to fit in 16 GB RAM and run on
   a consumer GPU.
2. **No reference set.** Unlike Content ID, the system doesn't have a curated
   reference library to match against. It does **all-pairs** within the
   user's library. This is a different problem.
3. **Low false-positive tolerance.** If the system says "duplicate," the user
   deletes a file. Meta T&S has a human reviewer downstream; this app does
   not. Operating point must be on the high-precision side of the curve.
4. **No training data.** A learned descriptor needs hundreds of thousands of
   labeled (positive, negative) pairs to train from scratch. **Pretrained
   models can be used off-the-shelf**; training a custom model is out of
   scope.
5. **FFmpeg dominates wall-clock.** Per-file FFmpeg subprocess startup is
   ~100ms; the actual hash compute is microseconds. Any algorithm that
   reduces frame-extraction count is high-leverage; any algorithm that just
   makes the compare step faster is irrelevant up to N ≈ 50k.

### Production techniques that *are* directly usable

| Technique | Effort | Why it fits |
|---|---|---|
| **PDQ instead of pHash** (Meta) | 1 day | Drop-in 256-bit hash with published calibration; replaces a guessed threshold with a measured one |
| **Spatial+temporal pHash** (VideoDuplicateFinder) | 0.5 day | One extra hash per frame; complements pHash on heavily-compressed content |
| **vPDQ-style hash-count matching** (Meta) | 1 day | Better-defined match rule than greedy best-match; published threshold |
| **Chromaprint for audio** (industry standard) | 1 day (already covered in algorithmic-improvements.md §4) | Replaces an obviously-weak RMS profile |
| **DINOv2 mean-pool + FAISS** (VSC2022 winners) | 3-4 days | Only at >100k catalogue; gates on visible compare-stage bottleneck |
| **TMK+PDQF single-vector-per-video** (Meta) | 5-7 days (C++ binding) | Only at >200k catalogue; questionable ROI on a single machine |

### Production techniques that are *over-engineered* for 50k files

| Technique | Why it's overkill |
|---|---|
| Self-supervised contrastive training (SSCD, VSC2022 winners) | Off-the-shelf pretrained model is fine; training is a 6-month research project |
| Distributed FAISS / GPU-FAISS clusters | Single-machine FAISS handles 10M+ vectors |
| 600M-feature SIFT local-retrieval (ISC2021 3rd) | Image-level technique; video-frame analogue is overkill at 50k |
| Content-Defined Chunking (Dropbox / FastCDC) | Solves delta-sync, not dedup. SHA-256 fast path is simpler |
| Custom CNN per dataset (ByteDance-style) | Pretrained is sufficient; custom needs labeled data |
| Spectro-temporal CNN for audio (Content ID) | Chromaprint covers 95% of consumer needs |
| Re-ranking with temporal-consistency models (TCSVT 2024) | Only useful for partial-copy / segment-level, which the duration filter already excludes |

### The boundary cases

Two techniques are at the edge — usable at this scale, but with a real ROI
question:

1. **TMK+PDQF.** Replaces the current 12-hash-per-video representation with
   one 256-d float per video. The win is index-able comparison: O(n) per
   query against FAISS, vs the current "duration-bucket + O(n²) verifier."
   For 50k files, the current approach is probably fast *enough* after the
   `algorithmic-improvements.md` BK-tree change; for 1M files, TMK+PDQF is
   the right answer. **Defer until catalogue scale > 200k**.

2. **DINOv2 mean-pool.** Strictly more accurate than any hash-based approach.
   But requires GPU at scan time (already true for this app), 5ms/image
   inference (already in budget), and a FAISS index. The blocker is **threshold
   calibration** — what cosine value separates "same content" from "same
   genre"? This needs a labeled held-out set to choose, which the app does
   not currently have. **Defer until a held-out set exists or false-negative
   complaints surface**.

---

## TMK+PDQF deep dive

The most-asked question in this research: **could a single hash per video
replace the current 12 hashes?**

### What TMK does

Reads a video, resamples to 15 fps (hardcoded), computes a PDQF (256-d
float, not 256-bit binary) per frame, and applies the Temporal Match Kernel:

```
For period P in {fast, slow}:
    For phase φ in {sin, cos}:
        Σ_t PDQF(t) · trig(2π t / P + φ)
```

The output is a single vector of length `n_periods × 2 × 256` (the
*coefficient* matrix), plus a length-256 mean vector. In Meta's reference,
two periods are used (4 frame and 16 frame), giving a per-video vector of
shape `2 × 2 × 256 + 256 = 1280` floats. At fp32 that's 5 KB per video; at
fp16 it's 2.5 KB.

Two TMK descriptors are compared by inner product of the mean vectors plus a
sum over the period/phase coefficients. The standard threshold (Dalins et al.
2019, Section 3): **cosine similarity ≥ 0.7** is a match.

### What it would replace in this codebase

Currently the per-video state is:

```python
VideoFile.perceptual_hashes: str  # JSON array of 12 hex hash strings, ~12 × 64 = 768 bytes
```

A TMK adoption would replace this with:

```python
VideoFile.tmk_descriptor: bytes  # 1280 × 4 bytes = 5 KB at fp32, or 2560 at fp16
```

Frame extraction stays the same (still need frames to compute PDQF per
frame). What changes:

- Stage 3 outputs one descriptor per video instead of 12 hashes.
- Stage 5 compare becomes a single FAISS-Flat-IP query per video against
  the rest of the catalogue.
- Within-duration-bucket O(n²) verifier is gone.

### Library status May 2026

- **`vpdq`** on PyPI provides Python bindings for vPDQ (the frame-bag
  variant), not TMK. **The TMK+PDQF Python binding does not exist as a
  PyPI package** as of this writing. The C++ implementation is in
  `facebook/ThreatExchange/tmk/`; binding it to Python requires either
  `pybind11` work or shelling out to the C++ CLI.
- The 2019 paper notes that TMK's compute is dominated by the per-frame PDQF
  extraction (essentially the same cost as the existing per-frame pHash).
  The TMK kernel itself is microseconds.

### Realistic adoption cost

- **2-3 days** to write a `pybind11` wrapper for the C++ TMK code, or shell
  out per video. Shelling out is much simpler and acceptable since stage 3
  already shells out for FFmpeg.
- **0.5 day** to add a `tmk_descriptor` BLOB column to `file_cache` and
  `VideoFile` and serialise.
- **1 day** to swap stage 5's compare logic to a FAISS-IndexFlatIP query.
- **0.5-1 day** to calibrate the threshold against the existing
  `HASH_SIMILARITY_THRESHOLD=14` operating point on a held-out test set.

Total: ~5-7 days for a working TMK-based pipeline. Net wall-clock benefit at
50k catalogue: probably 0 — the compare stage isn't the bottleneck. Net
wall-clock benefit at 500k: significant — moves the compare stage from
"duration-bucket O(n²)" to "FAISS O(log n)" globally.

### Recommendation

**Don't adopt TMK at the current scale.** The 12-hash representation is
finer-grained than TMK (which loses frame-level information through the
kernel) and gives equal or better recall on full-copy detection. TMK shines
when you need a single index-able vector for billion-scale ANN; the desktop
app's bottleneck is frame extraction, not comparison.

**Adopt PDQ (the per-frame variant) as a drop-in replacement for
`imagehash.phash`.** This is the simple win: same code shape, better
calibration, published thresholds. The 12-per-video shape stays.

**Reserve TMK as a Phase 4 option** if the catalogue grows past ~200k or if
a cross-catalogue duplicate-detection feature is added (e.g. "did this new
file already exist in a different library?").

---

## Open-source competitor comparison

The two best open-source competitors are Czkawka and VideoDuplicateFinder
(0x90d). Both ship; both are widely-used; both have engineering write-ups in
their repos.

### What Czkawka does well

- **Polished UI** (Krokiet) with cache management baked in.
- **Aggressive parallelism** in the discovery + hashing phases.
- **Robust file-system traversal** including network share handling.
- Slot-based scan caching (similar concept to this app's `file_cache`).

### What Czkawka does poorly

- Hash-based matching only; no audio fallback. Mirror-flipped duplicates,
  audio-only duplicates, severe re-encodes all miss.
- No quality-scoring step — Czkawka shows the user a list and lets them
  pick which to delete.
- No GPU acceleration for frame extraction.

### Worth porting from Czkawka

- **Cache management UI patterns** (this app's frontend Settings page could
  borrow the "cache statistics + clear cache" pattern from Krokiet).
- **Network-share-aware orphan handling** — covered in
  `caching-incremental.md` §7 already.

### What VideoDuplicateFinder does well

- **Spatial + temporal pHash** combo (described above, §7 above).
- **Configurable similarity threshold** exposed in the UI.
- **Filter-by-duration / by-codec / by-resolution** features in the comparison view.

### What VideoDuplicateFinder does poorly

- Cross-platform but Windows-first; the Linux/Mac builds lag.
- C#/.NET runtime is a packaging burden for a Python-side decision-maker
  (irrelevant for this app, since the comparison would be at the algorithm
  level).
- No audio fingerprinting.

### Worth porting from VideoDuplicateFinder

- **Temporal-delta hash** as a second column. Half a day of work; pure
  upside for re-encoded content.
- **Per-pair "spatial vs temporal" diagnostic** in `diagnose_pair.py` — gives
  the user a clearer "why did this match?" answer.
- The 1st-minute-only sampling. The current app samples frames uniformly
  across the whole duration; VideoDuplicateFinder samples only the first
  60s. For movies and TV that's a bad trade-off (intros dominate); for
  short-form (TikTok-class) it's fine and dramatically cheaper. **Don't
  port this** — the existing uniform sampling is more general.

### What neither does

- Audio-based duplicate detection (this app's audio fallback is a genuine
  differentiator).
- Quality-ranking of duplicates (best-quality auto-selected).
- Cross-scan caching of perceptual hashes (this app's `file_cache` is
  novel in the open-source space).

### Net assessment

The two closest competitors are *behind* this app's pipeline on the algorithm
front: they don't have audio fallback, they don't quality-rank, and they
don't cache across scans. They're *ahead* on UI polish (Czkawka) and on the
temporal-channel idea (VideoDuplicateFinder). The right strategic move is to
borrow the temporal-channel hash and keep this app's algorithmic
differentiation in audio + quality ranking + caching.

---

## Realistic adoption path

What to steal, ordered by ROI for this app's specific scale (10k-50k typical,
1M occasional):

### Steal now (≤ 1 week of work, immediate wins)

1. **PDQ instead of imagehash.phash.** `pip install pdqhash`. Replace
   `imagehash.phash(img, hash_size=16)` with `pdqhash.compute(img)[0]`. Same
   256-bit output, better calibration, published thresholds. Existing cache
   schema fits with a 1-line change to how hashes are serialised (PDQ
   outputs `numpy.ndarray` of 256 ints; current code expects `imagehash`
   objects).
2. **Temporal-delta hash channel.** Compute `phash(frame[i+1] - frame[i])`
   alongside `phash(frame[i])` during stage 3. Doubles the cache size for
   hashes (still trivial) and enables a strict-AND high-confidence match
   path. Borrowed from VideoDuplicateFinder.
3. **Chromaprint for audio.** Already covered in
   `algorithmic-improvements.md` §4; this survey only adds the production-
   pedigree justification.

### Consider when the existing pipeline hits a real wall (1-2 weeks each)

4. **vPDQ-style frame-count match rule.** Replace "average best-match Hamming
   distance ≤ threshold" with "N or more frames have a near-match." Provides
   a more interpretable threshold ("match if 7/12 frames match"). Borrowed
   from Meta vPDQ.
5. **BK-tree on aggregate PDQ hash** (already covered in
   `algorithmic-improvements.md` §1, listed here for completeness). PDQ gives
   a single 256-bit hash per frame; an aggregate (median-bit) over the 12
   frames gives a per-video bucket key for BK-tree indexing.

### Defer to Phase 4 (only at >200k catalogue)

6. **DINOv2 mean-pool + FAISS HNSW.** Already covered in
   `algorithmic-improvements.md` §3. The VSC2022-winner architecture, but
   smaller. Adopt only when the comparator becomes the wall-clock
   bottleneck.
7. **TMK+PDQF single-vector-per-video.** Adopt only if cross-library /
   cross-catalogue features become a priority. Not at this scale.

### Never (production-only, doesn't fit this app)

- Distributed FAISS clusters.
- Custom-trained CNN descriptors.
- Content-Defined Chunking (different problem).
- Reference-library Content-ID-style architecture.
- Apple/Google-style "we have infinite storage, just don't surface dupes."

---

## What to leave on the table

A short list of things that look attractive in the literature but don't fit
the desktop-app use case:

- **Audio CNNs for fingerprinting** (Content ID, ShazamID descendants). The
  Chromaprint baseline is good enough; CNN-based audio fingerprinting has
  no off-the-shelf Python library with comparable maturity.
- **Segment-level / partial-copy detection** (FIVR-200K DSVR, VCSL). Useful
  if a user uploads a 90-second clip cut from a 90-minute source; this app
  rejects those at duration-filter anyway. Adopting partial-copy detection
  would *widen* recall in a way the user didn't ask for and probably
  doesn't want ("the system thinks my home video is a duplicate of *Cars 3*
  because there's a 90-second clip of Cars 3 in it").
- **Watermark-robustness training** (SSCD, ISC2021 augmentations). Real but
  exotic edge case for a personal library; not a priority.
- **Mirror-flip detection.** Production systems handle this via
  `pdqhash.compute_dihedral()` (returns 8 hashes per image: 4 rotations × 2
  flips). Cheap to add but a known false-positive amplifier on stock-footage
  libraries. Defer until users actually complain.
- **CSAM-detection-grade calibration tables.** Meta's published PDQ
  thresholds are CSAM-calibrated. Don't naively copy them; the desktop app's
  operating point is different.

---

## Sources

Engineering write-ups and papers referenced above.

### Facebook / Meta
- [PDQ & TMK + PDQF — A Test Drive of Facebook's Perceptual Hashing Algorithms (arXiv:1912.07745)](https://arxiv.org/abs/1912.07745)
- [Facebook ThreatExchange GitHub](https://github.com/facebook/ThreatExchange)
- [ThreatExchange hashing.pdf (canonical engineering write-up)](https://raw.githubusercontent.com/facebook/ThreatExchange/main/hashing/hashing.pdf)
- [Hasher-Matcher-Actioner (HMA) reference deployment](https://github.com/facebook/ThreatExchange/tree/main/hasher-matcher-actioner)
- [pdqhash on PyPI (Python binding)](https://pypi.org/project/pdqhash/)
- [vpdq on PyPI (video PDQ binding)](https://pypi.org/project/vpdq/0.0.4/)
- [darwinium-com/pdqhash (Rust port)](https://github.com/darwinium-com/pdqhash)
- [SSCD: A Self-Supervised Descriptor for Image Copy Detection (arXiv:2202.10261)](https://arxiv.org/abs/2202.10261)
- [SSCD GitHub repository](https://github.com/facebookresearch/sscd-copy-detection)
- [ISC2021 image similarity challenge](https://sites.google.com/view/isc2021)
- [VSC2022 GitHub repo](https://github.com/facebookresearch/vsc2022)
- [Meta AI Video Similarity Challenge winners](https://drivendata.co/blog/meta-vsc-winners)
- [VSC2022 1st place solution](https://github.com/FeipengMa6/VSC22-Submission)
- [3rd Place VSC2022 solution (arXiv:2304.11964)](https://arxiv.org/abs/2304.11964)
- [The Faiss Library (arXiv:2401.08281)](https://arxiv.org/pdf/2401.08281)
- [FAISS Binary indexes wiki](https://github.com/facebookresearch/faiss/wiki/Binary-indexes)

### YouTube / Content ID
- [How Does The YouTube Content ID System Work (jdhao blog)](https://jdhao.github.io/2021/08/02/the_youtube_content_id_system/)
- [Wikipedia: Content ID](https://en.wikipedia.org/wiki/Content_ID)
- [YouTube's Digital Fingerprints (Streaming Media)](https://www.streamingmedia.com/Articles/ReadArticle.aspx?ArticleID=78358)
- [Fast Company: How YouTube Is Fixing Its Most Controversial Feature](https://www.fastcompany.com/3062494/how-youtube-is-fixing-its-most-controversial-feature)

### TikTok / ByteDance
- [Fast Video Deduplication and Localization (IEEE TCSVT 2024)](https://ieeexplore.ieee.org/iel8/76/4358651/10577179.pdf)
- [SVD: A Large-Scale Short Video Dataset (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Jiang_SVD_A_Large-Scale_Short_Video_Dataset_for_Near-Duplicate_Video_Retrieval_ICCV_2019_paper.pdf)
- [Fast and Robust Video Deduplication (ACM Mile-High Video 2023)](https://dl.acm.org/doi/10.1145/3588444.3591050)
- [Digital Fingerprinting on Multimedia: A Survey (arXiv:2408.14155)](https://arxiv.org/html/2408.14155v1)
- [A Cost-Efficient Video Deduplication System at Web-scale](https://par.nsf.gov/servlets/purl/10418826)

### Dropbox / Content-Defined Chunking
- [How Dropbox Syncs Files Without Re-Uploading Them](https://akshayghalme.com/blogs/how-dropbox-delta-sync-works/)
- [FastCDC paper (USENIX ATC 2016)](https://www.usenix.org/system/files/conference/atc16/atc16-paper-xia.pdf)
- [A Thorough Investigation of Content-Defined Chunking (arXiv:2409.06066)](https://arxiv.org/pdf/2409.06066)
- [restic — Content Defined Chunking](https://restic.net/blog/2015-09-12/restic-foundation1-cdc/)
- [Inside Dropbox's Brain: The Chunking Trick](https://new2026.medium.com/inside-dropboxs-brain-the-chunking-trick-that-lets-you-sync-gigabytes-in-seconds-e62a866bb407)

### Google Photos / Apple Photos
- [How to detect duplicate photos/videos in Google Photos (Quora)](https://www.quora.com/How-can-I-detect-duplicate-photos-videos-in-Google-Photos)
- [Apple Photos: cloud photo storage duplicate HDR](https://discussions.apple.com/thread/8026259)
- [Apple Photos: How to identify HDR and non-HDR photos](https://discussions.apple.com/thread/253332444)
- [Apple Photos: Live Photo import behaviour](https://discussions.apple.com/thread/253298510)

### Open-source competitors
- [qarmin/czkawka — main repo](https://github.com/qarmin/czkawka)
- [Czkawka: How Does It Detect Duplicate Files](https://czkawka.com/how-does-czkawka-detect-duplicate-files/)
- [Czkawka similar-videos quality issue thread](https://github.com/qarmin/czkawka/issues/1749)
- [0x90d/videoduplicatefinder (.NET)](https://github.com/0x90d/videoduplicatefinder)
- [0x90d/videoduplicatefinder on DeepWiki](https://deepwiki.com/0x90d/videoduplicatefinder)
- [Farmadupe/vid_dup_finder_lib (Rust)](https://github.com/Farmadupe/vid_dup_finder_lib)
- [vid_dup_finder on lib.rs](https://lib.rs/crates/vid_dup_finder)
- [akamhy/videohash — collage+wavelet single-hash approach](https://github.com/akamhy/videohash)
- [arsenetar/dupeguru — file/music/image but not video](https://github.com/arsenetar/dupeguru/issues/1227)
- [video-simili-duplicate-cleaner (Qt/C++)](https://theophanemayaud.github.io/video-simili-duplicate-cleaner/)
- [PhotoStructure v2026.1 release notes](https://photostructure.com/about/v2026.1/)
- [PhotoStructure 2025 release notes](https://photostructure.com/about/2025-release-notes/)
- [Awesome Duplication Finders list](https://github.com/github-userx/Awesome-Duplication-Finders)
- [idealo/imagededup library](https://github.com/idealo/imagededup)

### Audio fingerprinting
- [pyacoustid PyPI](https://pypi.org/project/pyacoustid/)
- [beetbox/pyacoustid GitHub](https://github.com/beetbox/pyacoustid)
- [Chromaprint reference (AcoustID)](https://acoustid.org/chromaprint)
- [acoustid/chromaprint C library](https://github.com/acoustid/chromaprint)

### Benchmarks
- [VCDB: A Large-Scale Database for Partial Copy Detection in Videos](https://link.springer.com/chapter/10.1007/978-3-319-10593-2_24)
- [VCSL: Segment-level video copy detection (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/He_A_Large-Scale_Comprehensive_Dataset_and_Copy-Overlap_Aware_Evaluation_Protocol_for_CVPR_2022_paper.pdf)
- [The 2023 video similarity dataset and challenge (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S107731422400078X)
- [FIVR-200K dataset](https://fvl.fudan.edu.cn/dataset/vcdb/list.htm)
- [PHVSpec: Benchmark-based Analysis of Perceptual Hash Systems for Videos (Tech Coalition)](https://technologycoalition.org/wp-content/uploads/Tech-Coalition-Video-Hash-Benchmark-Paper.pdf)

### Misc
- [pHash.org open source perceptual hash library](https://phash.org/docs/howto.html)
- [FB TMK PDQ WTF (Hacker Factor Blog: independent analysis)](https://www.hackerfactor.com/blog/index.php?/archives/971-FB-TMK-PDQ-WTF.html)
- [Similarity Detection in Online Integrity (FOSDEM 2023 slides)](https://archive.fosdem.org/2023/schedule/event/similarity_detection/attachments/slides/5771/export/events/attachments/similarity_detection/slides/5771/Similarity_Detection_in_Online_Integrity.pdf)
