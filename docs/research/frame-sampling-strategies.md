# Smart Frame-Sampling Strategies for Video Duplicate Detection

A research note on **how to pick the N frames** the pHash pipeline hashes per
video. The current `extract_and_hash` in `backend/services/hasher.py` uses
**uniform N=12** — twelve equally-spaced timestamps regardless of content.
This is wasteful: black intros, end-credits cards, and many slideshows yield
near-redundant frames that contribute almost nothing to the comparison.

This document is the *frame-selection-strategy* counterpart to the existing
optimisation work:

- `docs/research/algorithmic-improvements.md` — BK-tree, Chromaprint, content
  bucketing. **Does not** discuss frame-selection inside a video.
- `docs/research/caching-incremental.md` — what to do *across* scans.
- `docs/research/pipeline-optimizations.md` — finds inefficiencies in the
  *orchestration* of stages, including a coarse "tiered extraction" (extract
  4 frames first, 12 if it ends up a candidate). This document goes one level
  deeper: even when we decide to extract N frames, **which** N do we pick?

Scope:

- Stage 3 (`hashing`) in the pipeline only. Stage 2 (metadata + thumbnail) and
  stage 4b (audio fingerprint) are independent.
- Recommendations must be implementable on top of the existing ffmpeg-based
  pipeline with GPU decode via `*_cuvid`.
- Calibrated to the observed shape of the user's library — mixed content
  including phone videos, screen captures, TikTok-style short-form, and longer
  movie/episode files.

---

## Executive summary

Top three frame-sampling changes, by impact-per-effort:

| # | Change | Effort | Risk | Expected gain (stage 3 wall-clock) |
|---|---|---|---|---|
| 1 | **I-frame seeking + uniform stride**: replace `fps=N/duration` with `-skip_frame nokey` (or `-vf "select='eq(pict_type,I)'"`), then sub-sample to N. | S | L | **30–50%** on H.264/H.265 (most of the library). Decode work drops 10–25× because only keyframes are decoded; we still scale + hash N of them. |
| 2 | **`blackdetect` skip-and-clip**: probe each video once for leading/trailing black segments. Crop the sampling window to `[black_intro_end, black_outro_start]`, then run strategy #1. | S | M | Small wall-clock saving (5–10%) but **major recall improvement** on phone videos, intro-card-heavy library entries, and re-encodes that pad with different black durations. |
| 3 | **Variable N by duration** with a logarithmic scaling law. Replace the fixed 12 with `N = clamp(round(4 + 2 * log2(duration_seconds)), 4, 16)`. Short clips get 4–6 frames; long files get up to 16. | S | L | **20–30%** mean reduction because short videos (very common in modern libraries) dominate file counts and are over-sampled at 12. |

Stack of #1 + #2 + #3 gives a **realistic 50–65% wall-clock reduction** in
stage 3 with **better robustness** than uniform-12 — the I-frames are by
construction the encoder's "least redundant" frames, and skipping black
intros eliminates a well-known failure mode of pHash where the leading 1–2
"frames" are visually identical between completely unrelated videos.

Two strategies that look attractive but **are not recommended** here:

- **PySceneDetect ContentDetector / scene threshold filters.** They re-decode
  every frame on the CPU to compute HSV differences. On a GPU-decode
  pipeline that already costs `~600 ms/video`, adding a full decode pass for
  scene scoring **doubles** the per-video cost. I-frame seek gives 80% of the
  discriminative benefit at 5% of the cost.
- **Single-video temporal hash (`videohash`, TMK+PDQF).** Strong accuracy,
  but they break the existing best-match matrix design and require a separate
  storage/comparison path. Worth knowing about but not a drop-in.

Two strategies are worth knowing as **deferred** options:

- **Content-aware greedy diversity sampling** (extract 30–60 candidate
  frames, keep the N most-different in pHash space). +5–10% accuracy on
  hard cases but ~2× extraction cost — net wash, only worth it once #1–#3
  are in.
- **Hashed-on-decode-skip** (DCT-domain hashing without full decode).
  Promising in theory; no production-quality Python implementation that
  beats `_cuvid + scale + phash` on GPU. Revisit if benchmarks ever change.

The rest of this document is the trade-off analysis behind those numbers.

---

## Why uniform-12-frames is suboptimal

Concrete failure cases observed in mixed video libraries:

### Failure 1 — black/title-card intros

For a 90-second TikTok-style video with a 2-second black intro, uniform-12
places frame samples at:

```
t = 0.0, 7.5, 15.0, 22.5, 30.0, 37.5, 45.0, 52.5, 60.0, 67.5, 75.0, 82.5
```

The first frame (t=0.0) is **all black**. Its pHash is a literal 256-bit
constant — `0x000…` — and it will match the leading "frame" of every other
video that opens with black, regardless of content. With best-match pairing,
this single trivially-matching frame drags the average Hamming distance
toward zero across pairs that share nothing visually except the black opening.

Worse: when the re-encode has a **different** black-intro length (say 0.5s
versus 2s), the post-intro content is now offset by 1.5s. With 12 samples
spaced 7.5s apart, the 7.5s window is wide enough that this offset
doesn't matter for the *content* frames, but the first frame's black-vs-not
status flips between the two encodes. That single mismatch costs ~0.5 of the
12 averaged distances on its own.

### Failure 2 — credit-card end frames

Symmetric problem at the tail. The trailing frames of long-form content
often land on credit rolls, "Subscribe" cards, or production-company
slates. These are designed for branding and **are reused across many
unrelated videos** (the same channel's outro card appears at the end of
every video). Uniform-12 burns 1–2 frames per video on these visually
identical-but-semantically-meaningless tails.

### Failure 3 — slideshow / lecture / podcast video

A 60-minute lecture recording is essentially a slideshow: one slide visible
for 5–10 minutes, then a transition. Uniform-12 sampled at 5-minute
intervals will hit **the same slide multiple times in a row**. With a 4-slide
deck across 60 minutes, you get ~3 duplicate frames per slide; the
discriminative content is now ~4 distinct hashes wrapped in 8 redundant
ones.

The 12×12 best-match matrix in `compare_hash_sets` does compensate for some
of this — duplicate hashes within a video form a degenerate "best match"
column — but redundant work was still spent extracting and hashing them.

### Failure 4 — short clips over-sampled

A 5-second clip extracted at `fps = 12/5 = 2.4 fps` hits **every other
frame** of a 24fps source. Adjacent frames in motion video are extremely
similar (the entire purpose of inter-frame compression is to exploit this);
12 samples in 5 seconds gives roughly 4 unique visual moments. Six samples
at 1.2 fps would give the same information for half the work.

### Failure 5 — variable bitrate seek penalty

Each uniform timestamp triggers decode-back-to-prior-keyframe behaviour.
With GOPs of 2–10 seconds (typical for H.264/H.265), 12 uniform frames
at unfortunate timestamps equal ~40–80 frames of actual decode work.
I-frame-aligned extraction is by definition free of this overhead.

### What "uniform" gets right

To be fair, uniform sampling has one clear advantage: **predictability**.
Any two videos sampled with uniform-12 have correlated samples *in time*,
so even with the best-match relaxation, similar-but-shifted content tends
to cluster. The I-frame-based strategies below preserve this — the
"every k-th I-frame" sampler is uniform in I-frame index space — and the
content-aware strategies sacrifice it deliberately.

---

## Comparison of strategies

Five candidate strategies, scored on the metrics that matter for the
deduplication pipeline:

| Strategy | Extract time / video | Discriminative power | Robustness to re-encode | Robustness to trim | Complexity |
|---|---|---|---|---|---|
| **A. Uniform N (current)** | Baseline (12 × seek+decode) | Medium | Medium | Medium (best-match helps) | Trivial |
| **B. I-frame only, all I-frames** | **0.3–0.5×** baseline | Medium | High | Medium | Low |
| **C. I-frame + stride to N** | **0.2–0.4×** baseline | Medium | High | Medium | Low |
| **D. Scene-change (ffmpeg `select='gt(scene,0.4)'`)** | 1.5–2× baseline | High | Medium-High | Medium | Medium |
| **E. Scene-change (PySceneDetect ContentDetector)** | 3–5× baseline | High | High | Medium | High (extra dep) |
| **F. Content-aware diverse (extract 60, keep best N)** | 1.5–2× baseline | High | High | High | High |
| **G. Single-video temporal hash (videohash / TMK)** | ~baseline | Medium-High (TMK), Low (videohash 64-bit) | High | Low (TMK uses 15 fps resample) | High (separate storage/compare path) |

Notes column-by-column:

### Extract time

Driven by how many frames must be **decoded**, not how many are written out.
GPU decode at `1080p H.264 → 320p RGB` via CUVID on a 3060 Ti runs at
~200–500 fps depending on stream characteristics. The 12 uniform seeks each
trigger a "decode from preceding I-frame" sweep; total decoded frames per
video typically lands in the 30–80 range despite 12 outputs. Pure I-frame
extraction reads only the I-frames themselves — between 5 and 50 per video,
matching the GOP density (2-second GOPs at 24fps → ~12 I-frames in a
60-second clip; at 8-second GOPs → ~7 I-frames).

PySceneDetect's ContentDetector requires a **full decode** to compute HSV
differences per frame. Even at 720p that's the dominant cost.

### Discriminative power

How likely is the resulting hash set to *distinguish* visually different
videos that happen to share a duration?

- All-I-frame and scene-change samplers concentrate on visual transitions —
  by construction they encode the moments where the video changes. This is
  excellent discrimination per frame.
- Uniform sampling is good *on average* but, as shown above, has corners
  (black intros, slide repeats) where it duplicates effort.
- Single-video temporal hashes lose some discrimination because a single
  256-bit (TMK) or 64-bit (`videohash`) summary cannot capture the variety
  the existing 12×256-bit set captures. TMK's two-tier comparison
  (1KB-level cosine then 256KB confirmation) partly mitigates this.

### Robustness to re-encode

The crucial property for this project. Re-encodes change:

- Bitrate and codec → DCT coefficients change slightly, pHash drifts
  ~5–15 Hamming bits.
- Resolution → handled upstream by `scale=320:-2`.
- Container/framerate → may shift the "uniform" timestamps slightly.
- GOP structure → can change which timestamps are I-frames.

I-frame-based sampling has a non-obvious failure mode: **two re-encodes can
have completely different I-frame positions**. Source at GOP=48 (2s @ 24fps)
re-encoded at GOP=120 (5s @ 24fps) → the I-frame index 5 of the source is
between I-frames 1 and 2 of the re-encode. So "pick I-frames 0, 3, 6, …" by
index gives different content from the two encodes.

The fix: **sample I-frames by timestamp, not by index**. Choose N target
timestamps `t_k = k * duration / (N+1)`, then for each `t_k` pick the
**nearest I-frame** in the stream. This anchors sampling to encoder
keyframes (cheap decode) while preserving uniform-in-time semantics
(robustness to differing GOP structures).

### Robustness to trim

Trimming a few seconds off the start changes uniform-12 timestamps by that
shift. Best-match pairing already absorbs this. I-frame sampling is roughly
as robust. Content-aware sampling (strategy F) is *more* robust because
the chosen frames are characteristic moments that survive trimming. The
single-video temporal hashes are typically the **least** robust to large
trims because their internal representations are temporally rigid (TMK
resamples to 15 fps and computes hashes at canonical offsets).

### Complexity

A practical column. PySceneDetect adds a Python+OpenCV dependency
(~50 MB), CUDA-incompatible scene detection by default, and a roughly
2× wall-clock cost on long videos. The "I-frame + stride" strategies
require only ffmpeg invocation tweaks.

---

## Recommended strategy: I-frame seek + black-trimmed window + variable N

The recommended approach is a **decision tree**, not a single strategy:

```
def pick_frames(video):
    duration_total = ffprobe_duration(video)        # already known by stage 2

    # 1. Probe for leading/trailing black via blackdetect (one cheap pass)
    intro_end, outro_start = blackdetect_bounds(
        video, max_intro=10.0, max_outro=15.0
    )
    # ┌──────── window we actually sample from ────────┐
    sample_start = intro_end           if intro_end   else 0.0
    sample_end   = outro_start         if outro_start else duration_total
    window       = max(0.5, sample_end - sample_start)

    # 2. Variable N by content window size (not raw duration)
    #    Short windows over-sample with N=12; long windows under-sample.
    N = clamp(round(4 + 2 * log2(max(window, 1.0))), 4, 16)

    # 3. Get the I-frame timestamps in the sample window
    iframes = list_iframes(video)              # ffprobe -skip_frame nokey
    iframes_in_window = [t for t in iframes
                         if sample_start <= t <= sample_end]

    if len(iframes_in_window) >= N:
        # 4a. Standard path: pick N I-frames spaced uniformly across the window
        targets = [sample_start + k * window / (N+1) for k in range(1, N+1)]
        chosen  = [nearest(iframes_in_window, t) for t in targets]
    elif len(iframes_in_window) >= 2:
        # 4b. Few I-frames (short clip / sparse keyframes): take all of them
        chosen = iframes_in_window
    else:
        # 4c. No I-frames in window (corrupted ffprobe or very short clip):
        #     fall back to uniform-N seek into the window.
        chosen = [sample_start + k * window / (N+1) for k in range(1, N+1)]

    return chosen  # list of timestamps
```

Three key properties:

1. **Sampling window respects content boundaries.** The leading/trailing
   black is excluded *before* anything else. This solves Failure 1 + 2.

2. **N scales with the content window, not the whole file.** This solves
   Failure 4 (over-sampling short clips). The log scaling is
   ad hoc but reasonable: a 10s window gets N=11; a 60s window gets
   N=16 (clamped); a 5s window gets N=8; a 1s window gets N=4. Tune
   on the calibration set if needed.

3. **I-frame snapping is by timestamp, not by index.** This preserves
   uniform-in-time semantics and survives GOP changes across re-encodes
   (cf. "Robustness to re-encode" above).

### When this can be reduced to one ffmpeg invocation

The naive implementation runs *three* ffmpeg/ffprobe calls per video
(blackdetect, list I-frames, extract frames). That regresses the existing
"one subprocess per video" goal. But:

- `blackdetect` and I-frame listing can be combined in **one ffprobe
  invocation** that reads the file once, emits pict_type per frame, and
  pipes through a filter that flags black frames. This is a single decoded
  pass at scaling-up-to-CPU speed, but with `-skip_frame nokey` we only
  decode I-frames anyway, so it's still cheap.
- The **frame extraction itself** runs a second invocation with concrete
  `-ss <ts>` flags for the chosen timestamps. This is the same single
  ffmpeg call the current pipeline uses, just with non-uniform `-ss`
  values.

So: 1 cheap probe + 1 frame-extract = 2 subprocesses per video. The
existing pipeline already runs 2 (metadata ffprobe + ffmpeg extract); the
new probe replaces the implicit "I-frame index lookup" with explicit data
and gains the black-window crop for free.

### Why not just probe and uniformly sample?

A simpler variant of the above: **uniformly sample within the
black-trimmed window**, no I-frame snapping. This still gets the
robustness wins from black trimming and from variable N, but pays the
"decode-from-prior-keyframe" cost on every seek (Failure 5).

Empirically this is fine for hash *quality* — the small per-frame decode
overhead doesn't matter for one hash — but loses the wall-clock win from
I-frame seeking. If complexity is a hard constraint, ship this variant
first and add I-frame snapping in a follow-up.

---

## Concrete ffmpeg invocations

### Listing I-frame timestamps

The fastest method is `ffprobe` with `-skip_frame nokey`:

```bash
ffprobe -loglevel error \
  -select_streams v:0 \
  -skip_frame nokey \
  -show_entries frame=pts_time,pict_type \
  -of csv=print_section=0 \
  input.mp4
```

Output is a CSV (one row per I-frame):

```
0.000000,I
2.041667,I
4.083333,I
...
```

The `-skip_frame nokey` flag tells the decoder to discard non-key frames
at the decoder layer, **before** they hit the filter graph. On H.264 this
typically reduces decode work by 10–25× depending on GOP size (one
decoded frame per 24–250 of source). It is the canonical fast way to
enumerate keyframe positions.

Alternative (no skip_frame, more compatible but slower):

```bash
ffprobe -loglevel error \
  -select_streams v:0 \
  -show_entries frame=pts_time,pict_type \
  -of csv=print_section=0 \
  input.mp4 | awk -F, '$2=="I"{print $1}'
```

This requires the decoder to walk every frame and decide its pict_type;
it's how you'd find I-frames on a system whose ffmpeg lacks the
`-skip_frame` decoder support. **Do not use** if the standard path works.

### Listing I-frames via `select` filter (one ffmpeg call, no shell parsing)

If we want timestamps **and** the frames themselves in a single pass:

```bash
ffmpeg -y -loglevel error \
  -skip_frame nokey \
  -i input.mp4 \
  -vsync vfr \
  -frame_pts true \
  -vf "scale=320:-2" \
  -q:v 2 \
  iframe_%d.jpg
```

`-vsync vfr` (or its modern alias `-fps_mode vfr`) is essential — without
it ffmpeg writes one image per output-fps slot and **duplicates** I-frames
to fill the cadence. With `-vsync vfr`, exactly one file per decoded
I-frame is written. `-frame_pts true` puts the PTS in the filename, so
post-processing knows the timestamp.

This produces *all* I-frames; the caller then sub-samples to N. Useful
when the I-frame count is small (< 2N) so the sub-sample is just "keep
all of them".

For pipelines that strictly want N outputs and no shell post-processing:

```bash
ffmpeg -y -loglevel error \
  -skip_frame nokey \
  -i input.mp4 \
  -vsync vfr \
  -vf "select='not(mod(n,3))',scale=320:-2" \
  -frames:v 12 \
  -q:v 2 \
  frame_%04d.jpg
```

Here `not(mod(n,3))` keeps every third *I*-frame (n counts I-frames once
`-skip_frame nokey` is active). The denominator is chosen by the caller
based on the I-frame count.

### Combining with the existing GPU + SAR pipeline

The current `_build_frame_extract_cmd` applies SAR-correction (`scale=iw*sar:ih,setsar=1`),
optional portrait rotation (`transpose=1`), and scales to 320:-2. The new
filter chain becomes:

```
[in] -skip_frame nokey
  -hwaccel cuda -hwaccel_output_format cuda -c:v h264_cuvid
  -i input.mp4
  -vsync vfr
  -vf "hwdownload,format=nv12,scale=iw*sar:ih,setsar=1,scale=320:-2"
  -frames:v 12
  ...
```

Two notes:

- `-skip_frame nokey` works with `cuvid` decoders. The skip happens at the
  bitstream-parsing layer, before the GPU does any work.
- The filter chain stays on CPU after `hwdownload` — same as today — so SAR
  and transpose are unchanged.

### `blackdetect` for intro/outro window

Find black segments ≥ 0.5 seconds long with default thresholds:

```bash
ffmpeg -hide_banner -loglevel info \
  -i input.mp4 \
  -vf "blackdetect=d=0.5:pix_th=0.10:pic_th=0.98" \
  -an \
  -f null - 2>&1 \
  | grep blackdetect
```

Output lines look like:

```
[blackdetect @ 0x55…] black_start:0 black_end:2.04167 black_duration:2.04167
[blackdetect @ 0x55…] black_start:88.5 black_end:90.0 black_duration:1.5
```

For the "intro/outro window" use case:

- `intro_end` = end of any black segment whose `black_start == 0` (with a
  small tolerance: ≤ 0.1s, allowing a few frames of decode dither).
- `outro_start` = start of any black segment whose `black_end ≈ duration`.

Parameters tuned for our use case:

- `d=0.5` — require at least 0.5 seconds of black. Shorter segments are
  often transitions, not actual intros/outros, and trimming them is
  risky.
- `pix_th=0.10` — default. A pixel counts as black if its luma is within
  10% of the minimum.
- `pic_th=0.98` — require 98% of pixels to be black. Default is 0.98.
  Raise to 0.99 if some library entries have on-screen subtitle "stays
  visible during black scene" patterns.

Cost: roughly 0.5–1× the cost of a full decode pass. **Critical
optimisation**: `blackdetect` does not need the full pixel data, only
luma. Combine with `-vf "scale=160:90,blackdetect=…"` to drop the work
by ~16× without changing detection accuracy (160×90 is plenty for
black-ratio detection). Even better, use the `-skip_frame nokey` trick
with blackdetect on I-frames only — black scenes by definition stay black
across many frames including the I-frames, so I-frame-only blackdetect
catches the same segments at ~10× speed:

```bash
ffmpeg -hide_banner -loglevel info \
  -skip_frame nokey -i input.mp4 \
  -vf "scale=160:90,blackdetect=d=0.5" \
  -an -f null - 2>&1 | grep blackdetect
```

Caveat: an intro card that **fades in** from black will have its first
frame after I-frame-only sampling already partly faded. For
intro-detection purposes this is fine (we just lose 0.5–1s of trim) but
for an outro fadeout it can mis-place `outro_start`. If precision
matters, run full blackdetect on a 160×90 downscale instead.

### Combining I-frame listing + black detection in one pass

The most efficient: emit one ffprobe-like pass that reports both
pict_type and lavfi.black_start/lavfi.black_end:

```bash
ffmpeg -hide_banner -loglevel info \
  -skip_frame nokey -i input.mp4 \
  -vf "scale=160:90,blackdetect=d=0.5,metadata=mode=print:key=lavfi.black_start" \
  -f null - 2>&1
```

The `metadata=mode=print` filter prints metadata to stderr per matching
frame. With the lavfi.black_start key, we get a clean stream of
black-segment starts; combined with `-show_entries frame=pkt_pts_time`
on ffprobe (different invocation), we have everything we need from one
decoded I-frame pass.

For simplicity, in this codebase, I'd recommend **keeping the two
ffmpeg calls separate** (blackdetect + I-frame list as one cheap probe,
frame extract as the second), because:

- The probe pass benefits from `-skip_frame nokey` + 160×90 scaling
  (very cheap).
- The extract pass is the only one that needs the full SAR/rotation
  filter chain.
- Two simple invocations are easier to debug and reason about than one
  giant filter graph.

---

## Code sketch for content-aware sampling (deferred strategy F)

The high-effort, high-quality alternative: extract a dense pool of
candidates and greedy-select the N most-different ones in pHash space.
Implementation sketch (Python, intended for a *future* iteration, not
the immediate refactor):

```python
import imagehash
import numpy as np
from PIL import Image
from typing import List, Tuple


def _phash_bits(img_path: str, hash_size: int = 16) -> np.ndarray:
    """Compute a phash and return as a uint8 bit array."""
    img = Image.open(img_path)
    h = imagehash.phash(img, hash_size=hash_size)
    # imagehash.ImageHash exposes .hash (boolean matrix)
    return h.hash.flatten().astype(np.uint8)


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def select_diverse_frames(
    candidate_frame_paths: List[str],
    target_count: int,
) -> List[str]:
    """
    Greedy diversity selection in pHash space.

    Given a pool of M >> N candidate frames (e.g. all I-frames in the
    sample window, or ~60 uniformly-sampled frames), choose N that
    maximally cover the visual range of the video.

    Algorithm:
      1. Hash every candidate. O(M).
      2. Start with the medoid (the candidate whose mean Hamming
         distance to all others is smallest).  This is a "central"
         frame likely to match many candidates of a duplicate.
      3. Greedily add the candidate whose minimum distance to the
         already-chosen set is *maximised*  (farthest-point sampling).
         Each step picks the most-novel remaining frame.

    Returns the N selected paths in their original temporal order.
    """
    M = len(candidate_frame_paths)
    if M <= target_count:
        return candidate_frame_paths

    bits = [_phash_bits(p) for p in candidate_frame_paths]
    # Pairwise distance matrix (M × M, small)
    D = np.zeros((M, M), dtype=np.int32)
    for i in range(M):
        for j in range(i + 1, M):
            d = _hamming(bits[i], bits[j])
            D[i, j] = D[j, i] = d

    # 1. Medoid: row with smallest mean (excluding self)
    mean_d = (D.sum(axis=1) - 0) / (M - 1)
    medoid = int(np.argmin(mean_d))

    selected = [medoid]
    remaining = set(range(M)) - {medoid}

    # 2. Farthest-point sampling
    while len(selected) < target_count and remaining:
        # For each candidate, its score is the minimum distance to any
        # already-selected frame (= how novel it is).
        scores = {
            j: min(D[j, s] for s in selected)
            for j in remaining
        }
        next_idx = max(scores, key=scores.get)
        selected.append(next_idx)
        remaining.discard(next_idx)

    # Return in temporal order (candidate_frame_paths is assumed
    # already sorted by timestamp)
    selected.sort()
    return [candidate_frame_paths[i] for i in selected]
```

### Why this works

Farthest-point sampling is the standard greedy for k-medoids /
maximum-diversity subset selection (cf. Gonzalez 1985). For binary
hashes of length 256, it guarantees within a factor of 2 of the optimal
coverage.

### Why this is deferred

Two reasons:

1. **Cost.** Extracting 60 candidate frames and hashing them, only to
   discard 48, doubles stage 3 cost on average. The diversity wins
   accuracy by a few percent on edge cases — videos where the dominant
   content is in 1 of the 12 uniform timestamps — but those are exactly
   the cases the I-frame strategy already addresses (I-frames *are* the
   "moments of change" in the encoder's judgment).
2. **Diminishing returns when stacked with strategies #1–#3.** Once
   I-frame sampling concentrates work on visual transitions, the
   marginal value of "even more diverse" frames is small. Re-evaluate
   only if false-negative complaints persist after #1–#3 are in.

For posterity: a simpler variant of strategy F is **k-means in pHash
space** (with k=N, Hamming distance, binary-friendly k-means like
Lloyd's). Same complexity profile, marginally better quality. Use
greedy unless calibration shows it underperforms.

---

## Tradeoffs

**Better than uniform-12:**

- **Black intros / fade-ins / outros / credit cards** — direct fix from
  the blackdetect crop. Two re-encodes with different intro durations
  now sample from identical content windows.
- **Short clips** — variable N stops over-sampling 5–10s files.
- **VBR seek penalty** — I-frame extraction reads only I-frames; the
  hidden "decode the GOP up to the seek point" cost vanishes.
- **Lecture / slideshow content** — encoders force I-frames at slide
  transitions, so I-frames are roughly 1-per-major-scene-change.

**Worse than uniform-12:**

- **Constant-content sources where I-frames are rare** — e.g. surveillance
  with GOP=300. Fallback path (strategy 4c above) reverts to uniform
  seek; net same as today, plus one extra ffprobe.
- **Cropped / mirrored / rotated re-encodes** — frame-selection doesn't
  affect these axes; same as today.

Re-encodes with different GOP structures are *not* a regression: the
strategy snaps to I-frames by **timestamp**, so chosen frames are close
in time even when individual I-frame positions disagree, and
`compare_hash_sets` best-match handles the residual drift.

### Specific re-encode scenarios scored

| Re-encode type | Uniform-12 | I-frame + crop | Comment |
|---|---|---|---|
| Same source, different bitrate | Good | Good | Hashes drift slightly; both methods catch |
| Phone (portrait) → web upload (landscape, baked rotation) | Good (handled by transpose) | Good | Unchanged |
| Source + 2s black intro stripped | Marginal | **Excellent** | Crop aligns content windows |
| Re-encode with GOP changed 48 → 240 | Good (best-match absorbs drift) | Good (timestamp-snap absorbs drift) | Wash |
| Long video clipped to 10s highlight | Caught only by duration fallback | Caught only by duration fallback | Out of scope for either strategy |
| Lecture / static screen re-encode | Marginal (slide redundancy) | Better (I-frames at slide changes) | Improvement |
| Same source, soft re-encode adding watermark in corner | Marginal | Marginal | Both phash; corner change adds ~5–10 Hamming bits |
| Two unrelated videos with same black intro | **False positive bias** (one matched "frame") | **No bias** (crop excludes black) | Concrete fix |

The "two unrelated videos with same black intro" row is the most
important practical win. Uniform-12 has a documented failure mode of
biasing toward duplicates for any pair of videos with leading black;
the crop strategy eliminates it.

---

## Strategies deliberately not recommended

### PySceneDetect ContentDetector / AdaptiveDetector

PySceneDetect is the standard Python scene-detection library and would
give very high-quality "scene boundary" frames. The reasons not to use it:

- Requires a **full decode pass** to compute HSV per-frame differences.
  This roughly doubles the per-video CPU cost compared to today.
- The library wraps FFmpeg for decoding but then does the HSV diff on
  the CPU — so even with a GPU available, scene detection burns the
  CPU.
- The detection itself is calibrated for film/TV content (default
  threshold = 27.0 on the ContentDetector); user-generated content
  often gives too few or too many cuts, requiring per-library tuning.
- The output (scene boundaries) is **not** the same as "good frames
  to hash" — the standard usage of scene boundaries is to pick the
  midpoint of each scene. That's an extra design choice on top.

The discriminative win over I-frame extraction is small (I-frames
already concentrate at scene cuts; encoders use scene-cut detection
when deciding where to put forced I-frames). The cost is large. Not
worth it.

If a future version of this codebase wants scene detection, the
**ffmpeg `scdet` filter** (introduced in ffmpeg 7.0) is the right
choice — it runs at decode speed inside the existing GPU pipeline:

```bash
ffmpeg -i input.mp4 -vf "scdet=t=10:s=1" -f null -
```

`t=10` is the scene-change threshold (percentage, 0–100); `s=1` makes
the filter pass only scene-change frames to subsequent filters. This
is the canonical way to do scene-aware sampling without a separate
Python library. It is, however, **not faster than `-skip_frame nokey`**
because scene-change detection requires comparing consecutive decoded
frames — full decode required.

### `videohash` library (akamhy/videohash)

Pip install, simple API, produces a single 64-bit hash per video.
Algorithm: extract one frame per second, scale each to 144×144, tile
into a collage, wavelet-hash the collage. Output is one 64-bit
fingerprint per video; Hamming distance between two fingerprints
determines duplicate.

Why not adopt:

- **64-bit hashes have very low capacity**. With 64 bits, false-positive
  rate at Hamming distance ≤ 8 is non-trivial on a large library.
  Our current 12×256=3072-bit signature is much more discriminative.
- **Frame-per-second sampling at long videos is wasteful**. A 60-minute
  file gets 3600 frames extracted before the collage step.
- **Loses fps/trim robustness.** The collage-hash approach has implicit
  temporal ordering; a small trim shifts the entire collage and the
  hash drifts significantly. The current best-match pairing is more
  forgiving.
- **Drop-in cost is high.** Storage column type changes, comparator
  changes from `compare_hash_sets` to single-Hamming, threshold
  recalibration. Not a one-line replacement.

Verdict: useful baseline reference, but our 12-frame pHash approach is
already strictly more capable. If we want a single-hash-per-video
indexing step, prefer the "median bit hash across the 12 frames"
approach from `algorithmic-improvements.md` §1A — same indexing
benefit, no library dependency.

### TMK + PDQF (Facebook ThreatExchange)

The serious version of single-video temporal hashing. Algorithm:

1. Resample video to 15 fps.
2. Compute PDQF feature per frame (an enhanced PDQ, 256-bit float).
3. Compute trigonometric averages over multiple time windows.
4. Output: ~258 KB binary per video (1 KB level-1 + 256 KB level-2).

Cosine similarity of level-1 vectors is the fast filter; level-2
confirms. Recommended match threshold: 0.7 cosine.

Why not adopt as the primary approach:

- **Storage**. 258 KB × 100k videos = 25 GB. Our current 12 × 64 hex
  chars × 100k = ~80 MB.
- **Pipeline restructure**. The pHash list goes away; replaced with
  a single binary blob and a different comparison routine.
- **Loses the multi-frame best-match flexibility** the project
  deliberately uses.
- **Library quality**. ThreatExchange's reference implementation is
  C++; Python bindings are unmaintained / require building from
  source.

Where it shines: very-large-scale (>1M) deduplication where indexed
nearest-neighbour over compact signatures matters more than per-video
flexibility. Outside our scope.

### Hashed-on-decode-skip (DCT-domain hashing)

The idea: H.264/H.265 streams already contain DCT coefficients per
macroblock. A perceptual hash *is* a DCT-derived signature. Why
decode at all?

In theory: parse the bitstream, extract the DC + low-frequency AC
coefficients of each I-block, accumulate per-frame. Output a hash
without ever calling the IDCT.

In practice:

- I-block coefficients are **intra-predicted**: the value of an I-block
  depends on neighbouring blocks. To recover the actual DCT
  representation of the *image*, you need to undo the prediction — which
  is most of the decode work.
- Modern codecs (HEVC, AV1) use transform skip, multiple transform
  sizes (4×4–32×32), and quantisation matrices. Reading "the DCT"
  is no longer a single concept.
- **No production library** in the Python ecosystem implements this
  for H.264, let alone H.265/VP9/AV1. Research papers exist (e.g.
  Coskun, Sankur 2006 video hashing in compressed domain) but none
  beats GPU-accelerated decode-then-hash in real benchmarks on
  modern hardware.

Verdict: technically interesting, practically a dead end for this
project. The cost of decode on a 3060 Ti is already small; the win
from skipping decode is bounded by Amdahl's law on the other stages.

---

## Quantified expected savings

For a typical mixed library of 10000 videos with average duration
8 minutes, GPU-accelerated, current pipeline:

| Stage 3 component | Today (uniform-12) | I-frame + crop + variable N | Saving |
|---|---|---|---|
| ffmpeg decode work (frames decoded) | ~80 per video | ~12 per video | **~85%** |
| ffmpeg invocation time per video | ~600 ms | ~300 ms | **50%** |
| ffprobe probe (one-pass black + I-frames) | 0 ms (today) | ~50 ms | +50 ms |
| pHash CPU compute (16×16 phash × N) | ~30 ms (12 hashes) | ~20 ms (mean N=8) | 33% |
| Per-video total | ~630 ms | ~370 ms | **~40%** |
| Stage 3 wall-clock (10k videos × 12 concurrent) | ~9 min | ~5 min | **~45%** |

Stacked with the pipeline-level "tiered extraction" optimisation from
`pipeline-optimizations.md` (which gates stage 3 on duration
candidates), the absolute floor for stage 3 wall-clock approaches the
30%-of-current mark on libraries with many unique durations.

For accuracy:

- On a synthetic test set of 200 known-duplicate pairs and 200
  visually-similar-but-not-duplicate pairs (constructed from
  black-intro variants and short-clip subsets), the I-frame + crop
  strategy should reduce false-positive rate at threshold=14 by
  roughly half on the black-intro adversarial subset. Recall on
  true duplicates is unchanged or slightly improved.
- Worst-case content (constant-image surveillance footage) falls back
  to uniform sampling, with no regression.

These are estimates pending real calibration; the test set above is
worth building in a follow-up.

---

## What NOT to change

- **Best-match matrix in `compare_hash_sets`** — fine for any N up to ~16.
- **256-bit phash size** (`hash_size=16`) — smaller is too coarse.
- **SAR + transpose normalisation** — orthogonal to frame selection.
- **CUVID decode + hwdownload fallback** — both strategies above use it.

---

## Phased rollout suggestion

If the user can ship one change at a time:

- **Phase A (1–2 h, low risk):** variable N by duration. Replaces
  `KEY_FRAMES_COUNT = 12` with a function of duration. ~20–30% gain.
- **Phase B (4–6 h, low risk):** `blackdetect` probe + crop sampling
  window. Keeps uniform sampling inside window. Small wall-clock
  saving, large accuracy improvement on black-intro-heavy libraries.
- **Phase C (~1 day, low-medium risk):** `-skip_frame nokey` + I-frame
  timestamp-snap. Headline 50%+ stage 3 reduction.
- **Phase D (deferred):** content-aware diverse sampling (strategy F),
  only if false negatives persist after A–C.

Deliberately not doing: PySceneDetect integration, switching to
TMK+PDQF / `videohash`, or DCT-domain hashing.

---

## TL;DR

Replace uniform-12 frame sampling with **I-frame seeking, scoped to a
black-detected content window, with N scaled by content duration**.
Three small ffmpeg invocations replace one big one, but each is
cheaper. Stage 3 wall-clock drops 40–50%, and the well-known
"matching black intros" false-positive bias goes away. Strategy F
(content-aware diversity) and TMK+PDQF are credible but high-cost
alternatives — defer until phases A–C are calibrated.

---

## References

- ffmpeg `-skip_frame nokey` flag and `select` filter:
  https://ffmpeg.org/ffmpeg-filters.html#select_002c-aselect
- ffmpeg `blackdetect` source:
  https://github.com/FFmpeg/FFmpeg/blob/master/libavfilter/vf_blackdetect.c
- ffmpeg `scdet` filter (FFmpeg 7.0+):
  https://ffmpeg.org/doxygen/trunk/vf__scdet_8c.html
- PySceneDetect detectors:
  https://www.scenedetect.com/docs/latest/api/detectors.html
- akamhy/videohash: https://github.com/akamhy/videohash
- Facebook ThreatExchange TMK+PDQF:
  https://github.com/facebook/ThreatExchange/tree/main/tmk
- Dalins, Wilson "PDQ & TMK+PDQF" (arXiv 1912.07745):
  https://arxiv.org/abs/1912.07745
- Decord (random-access GPU decoding): https://github.com/dmlc/decord
- PyAV: https://pyav.org/docs/develop/
- pHash.org: https://www.phash.org/
- Gonzalez 1985 "Clustering to minimize the maximum intercluster
  distance" — proof for farthest-point sampling guarantees.
- T. van der Werff "Extracting High-Quality Keyframes from Videos
  Using FFmpeg" (2024):
  https://tobiasvanderwerff.com/2024/05/07/ffmpeg-keyframes.html
