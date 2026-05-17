# Pipeline Optimization Audit

This is an audit of `run_scan_pipeline()` in `backend/api/scan.py` and the services it
orchestrates (`scanner`, `metadata`, `hasher`, `audio_fingerprint`, `comparator`,
`gpu_detector`). The goal is to surface concrete, surgical inefficiencies — not
algorithmic rewrites — that explain why scans burn CPU and wall-clock time on large
libraries.

The findings are listed in approximate **impact-per-effort** order; the executive
summary picks the top five.

---

## Executive summary

| # | Finding | File:line | Effort | Risk | Impact |
|---|---------|-----------|--------|------|--------|
| 1 | **Audio fingerprint decodes the FULL track** instead of sampling | `audio_fingerprint.py:53–73` | S | L | Saves ~80–95% of stage 4b decode time on long files |
| 2 | **Tiered frame extraction**: extract 4 frames first, only extract 12 if a candidate | `hasher.py:259–326`, `scan.py:308–369` | M | M | Saves 40–60% of stage 3 time for "lonely" videos |
| 3 | **Stage 3 and stage 4b run sequentially** even though stage 3 is GPU-bound and stage 4b is CPU-bound | `scan.py:294–449` | M | L | Overlapping the two halves wall-clock for the candidate set |
| 4 | **Single shared semaphore** for GPU-bound and CPU-bound work; audio FP also reuses `max_concurrent` (often 12, GPU-tuned) which over-saturates the CPU | `scan.py:181, 419` | S | L | Removes contention, speeds up audio FP by avoiding context-switch storm |
| 5 | **`video_records` holds every ORM object for the whole scan** — on 50k files this is hundreds of MB; also blocks GC of finished items | `scan.py:176, 273, 430, 450–468` | M | M | Memory ceiling raised; minor wall-clock improvement on huge scans |

Other findings (#6–#11) are smaller wins individually but cheap to apply together.

---

## 1. Audio fingerprint decodes the FULL audio track (single biggest win)

**Where:** `backend/services/audio_fingerprint.py:53–73`

```python
cmd = [
    "ffmpeg",
    "-i", str(file_path),
    "-vn",
    "-ac", "1",
    "-ar", "8000",
    "-f", "s16le",
    "-acodec", "pcm_s16le",
    "pipe:1",
]
result = subprocess.run(cmd, capture_output=True, timeout=120, ...)
```

There is no `-ss` or `-t`. For a 90-minute file, FFmpeg decodes 90 × 60 = **5400 s of
audio at 8 kHz** = 43.2 M samples = 86 MB of PCM piped into Python only to be sliced
into **64 segments**, each computing a single `np.sqrt(np.mean(seg**2))`. Each
segment is therefore (5400 / 64) ≈ 84 seconds long; the per-segment RMS is
statistically identical whether you use the whole 84 s or any reasonably-long
sub-window (say, the first 1–2 s).

### What's wasted

Per-file CPU cost scales **linearly with duration**. On a typical mixed library:

| Avg duration | Naïve cost | Sampled cost (64 × 1 s) | Saved |
|--------------|-----------|--------------------------|-------|
| 5 min        | 300 s decoded | 64 s decoded            | ~80% |
| 30 min       | 1800 s     | 64 s                     | ~96% |
| 90 min       | 5400 s     | 64 s                     | ~99% |

Decode is fast (FFmpeg PCM extract is roughly 50–100× realtime on a modern CPU), but
this stage is gated by `_executor = ThreadPoolExecutor(max_workers=8)` and the
shared `audio_sem` (also 8–12), so wall-clock time is roughly
`total_audio_seconds / (8 × 50) = total_audio_seconds / 400`. On a 5000-candidate
library averaging 30 min, that's 5000 × 1800 / 400 ≈ **22500 s** = 6.25 hours of
wall-clock. Sampling cuts this to roughly 15 minutes.

### Proposed fix

Use FFmpeg's `aselect` filter (or repeated `-ss` calls) to grab 64 short windows
spread across the duration, OR — simpler — keep the 64-segment design but extract
each segment as a 0.5–1 s slice:

```python
# Pseudo: build a single ffmpeg invocation with one -i and an aselect filter
# expression that keeps 64 evenly-spaced 1-second windows.
filter_expr = (
    "aselect='" +
    "+".join(f"between(t,{i*duration/64},{i*duration/64 + 1})" for i in range(64)) +
    "',asetpts=N/SR/TB"
)
```

Even simpler approach: probe duration first (we already have it from stage 2 — just
pass it in!) and run **64 tiny `-ss + -t` decodes**. Subprocess overhead dominates
at that scale, so a single-pass `aselect` is preferable. Either yields ≥ 80%
savings.

The function signature should accept `duration: Optional[float] = None` exactly as
`extract_and_hash` does, and `scan.py:421` should pass `v.duration` so we don't
re-probe.

- **Difficulty**: S (one filter string, one new param).
- **Risk**: L if calibrated well — the energy profile is statistically equivalent
  for window sizes ≥ 0.25 s.

---

## 2. Tiered frame extraction: extract 4 frames first, 12 only if candidate

**Where:** `backend/services/hasher.py:259–326`, called from `backend/api/scan.py:308–369`

Stage 3 currently extracts `KEY_FRAMES_COUNT = 12` frames for **every** video:

```python
num_frames = options.get("key_frames_count", settings.KEY_FRAMES_COUNT)  # 12
...
return await extract_and_hash(
    video.file_path, num_frames,
    duration=video.duration, codec=video.video_codec, video_info=vinfo,
)
```

But the duration pre-grouping at `scan.py:393–404` already tells us which videos are
candidates. **A video alone in its duration bucket can never be a duplicate** — yet
we already paid for 12 frames + 12 pHash computations for it.

For the same reason audio FP is gated to candidates only (great optimisation —
`scan.py:389–404`), frame hashing should be tiered:

### Proposed two-tier scheme

1. **Tier 1 (all videos)**: extract 3–4 frames + compute their pHashes. Cheap
   enough we can run it everywhere.
2. **Group by duration** as today.
3. **Tier 2 (candidates only)**: re-extract 12 frames for items that are in a
   non-singleton duration bucket. These are the ones whose hashes will actually be
   compared pairwise — they need the full 12-frame fidelity.

Since FFmpeg's `fps=N/duration` extraction time is dominated by **decode**, not
output, going from 4 → 12 frames adds maybe 5–10% overhead. Skipping the second
extraction entirely for non-candidates is the win.

Empirically (see Stage 4a docs: 50–95% of files have unique durations on noisy
libraries):

- 70%-unique-durations library, 10000 files: tier-1 only on 7000, tier-1+tier-2 on
  3000. Old work: 10000 × 12 = 120000 frames extracted. New work: 7000 × 4 + 3000 ×
  16 = 76000 frames. **~37% reduction**, but the saving is on the slowest items
  (full decode pipeline runs once per video).

Even better: the **tier-1 hashes are still useful**. A videofile alone in its
duration bucket gets its 4 hashes stored too, so future incremental scans that add
new files of the same duration can compare them against existing tier-1 hashes
without re-extracting tier-1 frames.

### Alternative fusion: extract frames during stage 2 metadata pass

The metadata pass already runs ffprobe per file. We could fuse stage 2 + stage 3
into a single FFmpeg invocation per file: instead of `ffprobe + ffmpeg(thumbnail) +
ffmpeg(12 frames)` (3 subprocesses), run `ffmpeg -vf select+scale -frames N+1` once
that emits both the thumbnail and the N hash-frames. The metadata still needs
ffprobe (different output format), but we'd cut the **thumbnail subprocess** which
currently runs once per video alongside ffprobe (`scan.py:215`). This is finding
#10.

- **Difficulty**: M (need to emit 4-frame hashes, then re-extract 12 for
  candidates, and merge JSON).
- **Risk**: M — ensure tier-1 hashes are stored in the same column format so
  comparator can read them; ensure the "best-match" comparison still works with
  4-vs-12 mismatched lengths (it does — `compare_hash_sets` already handles
  `min(n1, n2)`).

---

## 3. Stage 3 and stage 4b run sequentially despite different bottlenecks

**Where:** `backend/api/scan.py:294–449`

```
Stage 3 (hashing)         : 45% → 75%   (GPU-bound, GPU semaphore)
Stage 4a (pre-grouping)   : (instant CPU)
Stage 4b (audio FP)       : 76%         (CPU-bound, same semaphore)
```

These are run strictly serially. But the bottleneck of stage 3 is **GPU decode +
small CPU filter chain**; the bottleneck of stage 4b is **CPU PCM decode**. A GPU
running CUVID at 30% utilisation while waiting for FFmpeg + filter on the host
doesn't help the audio decoder.

If we move stage 4a to **right after stage 2** (we already have all the durations
we need at end of stage 2!), then stage 4b can be kicked off as a separate
asyncio.Task in parallel with stage 3. The two tasks finish around the same wall
clock, then we converge at the comparator.

### Proposed fix

```python
# After stage 2:
candidate_paths = compute_candidates_from_records(video_records)

# Kick off audio FP in parallel with hashing:
audio_fp_task = asyncio.create_task(
    _run_audio_fp_stage(candidate_paths, ...)
)

# Stage 3 (hashing) runs as today
await _run_hashing_stage(...)

# Now collect the audio results that ran in parallel
audio_fps = await audio_fp_task
```

With separate semaphores (finding #4) the two stages don't fight for FFmpeg slots.

- **Difficulty**: M.
- **Risk**: L — the `_pipeline_check` discipline still works as long as both tasks
  use the same `scan_control` events.

---

## 4. Single shared semaphore for GPU-bound and CPU-bound work

**Where:** `backend/api/scan.py:181, 315, 419`

```python
sem = asyncio.Semaphore(max_concurrent)         # used in stages 2 + 3
audio_sem = asyncio.Semaphore(max_concurrent)   # also = max_concurrent
```

When GPU is active, `max_concurrent = GPU_MAX_CONCURRENT = 12` (`config.py:23`).
That's tuned for **GPU decode + 12 lightweight CPU filter chains**, which fits an
RTX 3060 Ti with ~8 GB VRAM. But:

- For stage 4b (audio FP), 12 concurrent FFmpeg PCM-decode subprocesses on an 8-core
  CPU is **oversubscribed** — each ffmpeg process spawns its own threads, leading
  to context-switch thrash and net slowdown.
- `_executor = ThreadPoolExecutor(max_workers=8)` in `audio_fingerprint.py:33` caps
  it at 8, so the actual concurrency is `min(8, 12) = 8` — *but only because of
  this hidden cap*. The semaphore says 12; the pool says 8. They disagree.

### Proposed fix

```python
# Different limits per stage:
HASH_CONCURRENCY  = max_concurrent              # 12 GPU streams OK
AUDIO_CONCURRENCY = min(os.cpu_count(), 8)      # CPU-only, conservative

hash_sem  = asyncio.Semaphore(HASH_CONCURRENCY)
audio_sem = asyncio.Semaphore(AUDIO_CONCURRENCY)
```

Also: the `_executor` in `hasher.py:38` is `MAX_CONCURRENT_FFMPEG * 3 = 24`
threads, but only `max_concurrent = 12` jobs ever queue at once. The factor-of-3 is
unused.

- **Difficulty**: S (4 lines).
- **Risk**: L.

---

## 5. `video_records` holds every ORM object for the whole scan

**Where:** `backend/api/scan.py:176, 273, 430, 450–468`

```python
video_records = []
...
video_records.append(r)        # appends a VideoFile ORM object
...
# at stage 3:
batch_videos = video_records[batch_start:batch_end]
...
# at stage 4b:
candidate_list = [v for v in video_records if v.file_path in _candidate_paths]
...
# at compare:
for v in video_records:
    hashes = json.loads(v.perceptual_hashes) if v.perceptual_hashes else []
    ...
    video_data.append(vd)        # AND a parallel dict copy
```

For a 50,000-file scan, this is two lists of 50k items each, plus SQLAlchemy
identity-map references. Each `VideoFile` is on the order of a kilobyte; with the
parallel `video_data` list that's ~100 MB sitting in memory the entire scan, and
the python objects pin SA's identity map preventing the session from compacting.

The `_meta_video_info` dict stored on each `VideoFile` (`scan.py:240`) also persists
even after stage 3 has consumed it.

### Proposed fix

The pipeline already commits each `VideoFile` to the DB at `scan.py:272–278`. After
stage 3 commits hashes (`scan.py:355`), we don't need the python ORM object — we
need:
- `id`, `file_path`, `duration`, `perceptual_hashes`, `width`, `height`, `bitrate`,
  `video_codec`, `file_size`, `fps` — for the comparator.

We could re-load only those columns with a single `SELECT` after stage 3 ends:

```python
result = await db.execute(
    select(VideoFile.id, VideoFile.file_path, VideoFile.duration,
           VideoFile.perceptual_hashes, VideoFile.width, VideoFile.height,
           VideoFile.bitrate, VideoFile.video_codec, VideoFile.file_size,
           VideoFile.fps)
    .where(VideoFile.scan_job_id == scan_id)
)
rows = result.all()
```

Then drop `video_records` entirely between stages. SA's session expires on commit
(set off via `expire_on_commit=False` at `database.py:16` — exactly the wrong
choice if memory matters), so we'd also flip that to `True` for the scan session.

- **Difficulty**: M.
- **Risk**: M — need to clean up `_meta_video_info` ferrying, restructure stage 3
  to read its inputs from DB rows.

---

## 6. Comparator inner loop: no early-exit on duration delta or quick-reject

**Where:** `backend/services/comparator.py:117–155`

Within a duration group, the inner loop walks every (i, j) pair:

```python
for i in range(n):
    hashes_i = videos[i].get("hashes") or []
    audio_i  = videos[i].get("audio_fp") or []
    for j in range(i + 1, n):
        if not _file_size_compatible(videos[i], videos[j]):
            continue
        hashes_j = videos[j].get("hashes") or []
        audio_j  = videos[j].get("audio_fp") or []
        ...
```

Two issues:

### 6a. No Union-Find skip in the inner loop

The function uses Union-Find but **only to merge transitive matches**, not to skip
pairs that are already in the same component. If `find(i) == find(j)`, comparing
them adds nothing. On a duration group of 30 items where the first 15 are all the
same content, that's ~100 redundant `compare_hash_sets` calls each costing
O(num_frames²) = 144 hash comparisons.

```python
for j in range(i + 1, n):
    if find(i) == find(j):
        continue                            # ← cheap skip, currently missing
    if not _file_size_compatible(videos[i], videos[j]):
        continue
    ...
```

### 6b. `compare_hash_sets` builds full distance matrix even for clearly mismatched pairs

`hasher.py:540–551`:

```python
dist = np.full((n1, n2), 999, dtype=np.int32)
for i, b1 in enumerate(bits1):
    if b1 is None:
        continue
    for j, b2 in enumerate(bits2):
        if b2 is not None and len(b1) == len(b2):
            dist[i, j] = int(np.count_nonzero(b1 != b2))
```

For 12×12 hashes that's a 144-entry matrix built with a Python double loop. Better
done as a single vectorised XOR:

```python
B1 = np.stack(bits1)              # (n1, 256)
B2 = np.stack(bits2)              # (n2, 256)
# Pairwise distance = popcount(b1 XOR b2)
dist = (B1[:, None, :] != B2[None, :, :]).sum(axis=-1)
```

For 12×12 it's a small win, but with `key_frames_count=24` (or larger) it scales
better.

### 6c. Duration-sort the comparison order

`group_by_duration` already sorts by duration, but `find_duplicates_in_group` then
walks `videos[i]` in input order. If we **re-sort by `file_size` descending** within
the group, large files compare with their nearest size-neighbours first, and the
file-size sanity check (`hi <= lo * 20`) eliminates outliers earlier.

- **Difficulty**: S for 6a/6c, M for 6b.
- **Risk**: L (6a/6c are pure additions); M for 6b (numpy version must match
  semantics exactly).

---

## 7. `KEY_FRAMES_COUNT = 12` is high for 16×16 pHash

**Where:** `backend/config.py:19`

A `phash` with `hash_size=16` gives 256-bit hashes. With 12 of them, the
"signature" of a video is 12 × 256 = 3072 bits. For visual content, 4–6 hashes
capture the full key-scene distribution; the marginal value of frames 7–12 is small
and they get matched greedily via best-match anyway.

**This is finding #2 in disguise** — but worth listing separately. Even without
tiered extraction, dropping to 8 frames cuts stage 3 ffmpeg seek/decode work by
~33%. The hash-comparison cost in stage 5 also drops by 144 → 64 entries per pair.

- **Difficulty**: S (one config value).
- **Risk**: M (need to validate accuracy doesn't regress on test set).

---

## 8. Pre-grouping for audio FP looks correct, but candidate set could be tighter

**Where:** `backend/api/scan.py:393–404`

```python
_pre_video_data = [
    {"file_path": v.file_path, "duration": v.duration} for v in video_records
]
_pre_groups = _pre_group(_pre_video_data, duration_tolerance)
_candidate_paths: set = set()
for g in _pre_groups:
    for vd in g:
        _candidate_paths.add(vd["file_path"])
```

This works correctly: only files in a duration group of size ≥ 2 get fingerprinted.
But within the comparator at stage 5, the order of operations is:

1. Try video pHash. If match → done.
2. Otherwise try audio FP.

For pairs that match on **video pHash alone**, the audio FP was wasted work. We
fingerprinted both to enable the audio-fallback branch, but in practice if the
visual hashes match cleanly, audio is never consulted.

### Possible refinement

Skip audio FP for pairs where video pHash is already a strong match. But this
requires running the comparator partially before audio FP, which inverts the
pipeline. Probably **not worth the complexity** — the current pre-group already
filters 50–95%, and the remaining audio FP is the right insurance against
visual-only re-encodes.

The only minor win: ensure the candidate set excludes videos that **failed** pHash
extraction (`hash_computed = False`) — if a video has no usable hashes, it cannot
match by audio either, because the audio fallback is per-pair: both must have audio
AND at least one must have failed visual matching against another item in the same
group. As-is, an unhashed candidate sitting in a duration group with one other
member just gets fingerprinted but has nothing to compare to that won't already be
caught.

- **Difficulty**: S.
- **Risk**: L (it's a strict subset).

---

## 9. Small `db.commit()` calls in tight loops

**Where:** `backend/api/scan.py:272–278, 348–355, 478–510`

The metadata batch loop calls `db.commit()` once per **batch** (every 32–48 files).
For a 10000-file library that's ~250–300 commits. Each commit on SQLite involves a
write transaction + WAL flush.

The hash batch loop also commits every batch (`scan.py:355`).

The duplicate-group save loop commits **once at the end** (`scan.py:510`), which is
fine.

### Proposed fix

Commit only every **N batches** (say, every 10 batches) for metadata and hashing,
unless a pause/stop signal arrives. Pre-emptive commits at signal time ensure
durability for resumable scans, but mid-scan the data isn't user-visible until the
WS notification anyway.

Alternative: switch SQLite to WAL with `synchronous=NORMAL` (likely already the
case for `aiosqlite`) and let the OS coalesce writes. But fewer Python-level
commits is still a win in CPU & async-loop contention.

- **Difficulty**: S.
- **Risk**: M — on crash, more work to redo; resumed scans need to handle the
  smaller window of un-persisted state.

---

## 10. Thumbnail extraction in stage 2 is on the critical path

**Where:** `backend/api/scan.py:215–219`, `backend/services/hasher.py:393–479`

```python
await extract_thumbnail(
    vpath, thumb_path,
    duration=meta.get("duration"),
    codec=meta.get("video_codec"),
)
```

The thumbnail is a UX nicety used only by the frontend's comparison view. Generating
it during stage 2 forces every file to pay for an ffmpeg subprocess (single frame +
scale + jpeg encode) before the scan can advance.

For a duplicate-detection scan this is wasted work for files that turn out to be
unique (and even for many of the duplicates — only the ones the user actually opens
in the viewer need a thumbnail).

### Proposed fix

Defer thumbnail extraction to **on-demand**:

1. Stage 2 stores `thumbnail_path = None`.
2. Frontend comparison endpoint checks `if not thumbnail_path: extract and cache`.
3. Optionally a background task (low priority, single-threaded) generates thumbnails
   for entries in completed duplicate groups *after* the scan finishes — so they're
   ready by the time the user opens the UI.

Order-of-magnitude: `extract_thumbnail` adds roughly the same wall-clock as a
4-frame extract. Removing it from stage 2 **roughly halves stage 2's per-file
cost**.

- **Difficulty**: M (need a small endpoint + frontend hook).
- **Risk**: M (UX may stutter the first time a comparison view opens).

---

## 11. Subprocess-call accounting per video (sanity check)

The audit prompt asks how many ffprobe/ffmpeg invocations each video receives. After
walking the code:

| Stage                      | Subprocess  | When                                           |
|----------------------------|-------------|------------------------------------------------|
| 2: metadata                | 1 ffprobe   | always                                         |
| 2: thumbnail               | 1 ffmpeg    | always (1 fallback ffmpeg if GPU thumbnail fails) |
| 3: extract_and_hash        | 1 ffmpeg    | always (1 fallback ffmpeg if GPU yields 0 frames) |
| 4b: audio fingerprint      | 1 ffmpeg    | only for candidates                            |

So per video: **~3 subprocesses (non-candidate)** or **~4 (candidate)**, with
GPU-fallback adding a 5th in rare cases.

The hasher's helpers `_get_video_duration` and `_get_video_codec` and
`_get_video_info` are **only called when their respective param is None** —
verified at `hasher.py:279–286`. The pipeline does pass them all
(`scan.py:319–326`), so for the happy path there are no redundant ffprobes.

**However**: `scan.py:215–219` in the thumbnail call passes `duration` and `codec`,
but **not `video_info`**. Looking at `_extract_thumbnail_sync` (`hasher.py:393`),
the thumbnail doesn't need SAR/rotation, so this is fine. ✓

**One real redundancy**: when GPU thumbnail extraction fails (`hasher.py:459–474`),
the CPU fallback **does not pass `duration` or `codec`** because they're already
captured in closure — but the CPU fallback also re-spawns a fresh subprocess. That's
correct. ✓

**Another small one**: the GPU detector runs `nvidia-smi`, `ffmpeg -hwaccels`,
`ffmpeg -decoders`, `ffmpeg -filters`, `ffmpeg -encoders` — five subprocesses on
first call. It's cached (`gpu_detector.py:55–63`), so only first scan pays. ✓

**Actually-redundant ffprobe calls per video: 0** (good!). The optimisation
groundwork is already in place; the wins now are at the **stage** level, not the
**per-call** level.

---

## 12. Minor: progress percent calculation runs even when WS is throttled

**Where:** `backend/api/scan.py:275–290, 352–367`

```python
scan.scanned_files = batch_end
scan.progress_percent = round((batch_end / total_files) * 40 + 5, 1)
scan.current_file = os.path.basename(batch_paths[-1])
await db.commit()                 # always

if _should_send_ws(scan.progress_percent):
    await _send_status_ws(...)    # throttled
```

The DB commit happens every batch even when the WS update is throttled. Combined
with finding #9, this is the biggest source of small commits.

- **Difficulty**: S.
- **Risk**: L.

---

## 13. Extra: pre-caching `_meta_video_info` is good — but the dict is re-created in stage 3

**Where:** `backend/api/scan.py:319, hasher.py:213`

```python
vinfo = getattr(video, "_meta_video_info", None)
return await extract_and_hash(
    video.file_path, num_frames,
    duration=video.duration, codec=video.video_codec, video_info=vinfo,
)
```

```python
# in hasher.py
vinfo = video_info if video_info else _get_video_info(file_path)
```

Good — when `video_info` is provided, no extra ffprobe runs. But the dict-stash
pattern (`scan.py:240`) attaches a mutable Python dict to a SQLAlchemy ORM object.
SA may issue spurious "instance has been modified" detections during flush.

Cleaner: keep a parallel `dict[file_path, video_info]` map alongside `video_records`
that gets discarded after stage 3.

- **Difficulty**: S.
- **Risk**: L.

---

## Cross-references

- Pre-grouping (Stage 4a) is correctly implemented and saves substantial work
  (`scan.py:393–404`). ✓
- The hasher correctly accepts and respects `duration`, `codec`, `video_info`
  pre-supplied params (`hasher.py:259–286`). ✓
- The thumbnail correctly accepts `duration`, `codec` (`hasher.py:393`). ✓
- The comparator correctly filters by file-size ratio before doing the heavy hash
  comparison (`comparator.py:122–123`). ✓
- The `_executor` in `hasher.py:38` is shared, so concurrent scans don't each spawn
  their own thread pool. ✓

---

## Recommended rollout order

If the user can only afford to ship one change at a time, do them in this order:

1. **Audio FP sampling** (#1) — biggest single win, smallest patch.
2. **Separate semaphores** (#4) — required to actually realise wins from #1 and #3.
3. **Parallel stage 3 + 4b** (#3) — wall-clock improvement once #1 + #4 land.
4. **Tiered extraction** (#2) — accuracy validation needed but big CPU saver.
5. **Defer thumbnails** (#10) — UX-visible change, do last.

The rest (#5, #6, #7, #9, #11, #12, #13) are micro-optimisations: bundle them into
a "polish" PR after the big-five wins.
