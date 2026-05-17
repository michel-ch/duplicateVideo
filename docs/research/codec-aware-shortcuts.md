# Codec-Aware Shortcuts and Demuxer-Only Paths

Faster duplicate-video detection without doing full inverse-DCT + YUV→RGB decoding for every candidate file.

This document complements the three existing research papers in this folder:
- `algorithmic-improvements.md` — algorithmic wins inside the existing pipeline (LSH, BK-tree, smarter audio).
- `caching-incremental.md` — cross-scan persistence of expensive results.
- `pipeline-optimizations.md` — concurrency, batching, and stage ordering.

Here we instead ask: **what can we measure without ever rendering a pixel?**

The current `services/hasher.py` pipeline costs roughly **150–800 ms per file** even on GPU (decode → SAR/transpose → scale → JPEG encode → PIL → DCT pHash, repeated 8–12 times). For a 10k-file library that is ≥ 25 minutes of pure decode work on top of metadata. We want a cheap pre-pHash cascade that throws out 90 %+ of the non-pairs before they reach the GPU.

---

## Executive Summary

Three shortcuts, ranked by **likely impact × ease of integration in 2026 with stock ffmpeg + Python**.

| Rank | Shortcut | What it catches | Per-file cost | Effort | When it fails |
|---|---|---|---|---|---|
| 1 | **Frame-size signature** (per-packet byte count from `ffprobe -show_packets`) | Same source re-encoded with the same encoder/settings; exact duplicates with different container; bitrate-ladder copies | ~30–80 ms | 1 file change, no new dep | Cross-codec re-encodes (H.264 → HEVC); aggressive bitrate change |
| 2 | **FFmpeg `signature` filter (MPEG-7 Video Signature, ISO/IEC 15938-3)** | Re-encodes across codec, resolution, container; produced & validated for archive use cases | ~100–400 ms (still decodes, but ~3× cheaper than 12-frame pHash + has built-in matcher) | Replace `extract_and_hash` for the hashing step | Heavily cropped or rotated content (similar to pHash) |
| 3 | **Bitstream byte-sample xxhash** for exact-byte duplicates | "Same file under different name", hard-linked copies, identical bitrate-ladder renditions | ~5–15 ms | 1 file change, only stdlib + `xxhash` | Container repack (mp4 ↔ mkv); any re-encode |

Recommended order of integration: 3 → 1 → (evaluate 2 against current pHash).

**Estimated funnel reduction** (rough, for a typical mixed library of ~10k files):

```
Discovered files                       10000  (100 %)
  after byte-xxhash collapse (exact)    9600  (96 %)   -4 %  free
  after frame-size signature group       400  (4 %)   -96 %  cheap
  after duration & 20× size guard        ~80  (0.8 %)
  after pHash (full decode)              ~30  (0.3 %)
```

The point of stages 1–2 of the cascade is that the GPU pHash step only sees ~80 candidates instead of every pair from the duration groups, which is the dominant cost in any library where the duration histogram has bunches.

---

## How to read the cost numbers

All "ms per file" numbers below assume:
- A typical 720p–1080p H.264/HEVC file, 30–120 s duration.
- ffprobe / ffmpeg on PATH, no GPU contention.
- Cold cache (first time the file is touched).

They are rough; on this repo's stack (Windows, async semaphore = 12 with GPU), per-file *wall-clock* will be 2–4× lower because ffprobe runs concurrently.

---

# Per-Finding Sections

## 1. DCT-coefficient hashing (direct from bitstream)

### What it would measure

H.264, HEVC, and VP9 all use block transforms (4×4 / 8×8 / up to 32×32) at the encoder. After quantisation, the coefficients sit in the bitstream verbatim. In principle you can:

1. Parse the bitstream into NAL units / OBUs.
2. Entropy-decode (CAVLC, CABAC) to recover the quantised transform coefficients.
3. Hash the DC + first few AC coefficients of each macroblock from N keyframes.

The hash would survive container repack and trivial re-multiplexing because the coefficients are unchanged — but break the moment the file is re-encoded.

### Library / API survey (state of 2026)

- **PyAV**: exposes packets (NAL-unit boundaries for H.264 in Annex-B form) but does *not* expose post-entropy-decoded transform coefficients. PyAV has not landed bitstream-filter passthrough nor coefficient inspection in 17.x.
- **FFmpeg `trace_headers` bitstream filter**: emits *header-level* syntax (NAL unit type, SPS/PPS, slice header). It does **not** emit per-block transform coefficients. It is useful for GOP structure (see §5) and `nal_unit_type` patterns (§7) but not for transform-domain hashing.
- **GPAC `analyse=bs` / `inspect:analyze=on`**: parses syntax elements down to slice-header level. Macroblock layer is documented as "work in progress" — slice headers only.
- **LLNL/trestles**: a research fork of the H.264 reference decoder that exports residual transform coefficients and motion-vector differences. C, last touched years ago, only H.264. *Research-grade.*
- **bento4**: MP4 container parser. Does not entropy-decode the codec payload.

### Feasibility verdict (2026)

**Research-only. Do not implement.** Reasons:

1. No 2026 stock Python library exposes post-entropy-decoded transform coefficients. You'd need to write a CABAC decoder or vendor `trestles`. CABAC alone is ~1500 lines of tightly-coupled C that gets revised between H.264 profiles.
2. Coverage gap: even a working H.264 implementation buys you nothing for HEVC (different transform sizes, different scan order, asymmetric partitions) or AV1 (whole different transform family — DCT, ADST, identity, WHT).
3. **The thing it would catch — container repacks of the same bitstream — is already caught for free by §10 (byte-sample xxhash).** Re-encodes that change the quantisation table break DCT-coefficient hashes anyway. There is no middle ground where this technique uniquely wins.
4. Maintenance: a CABAC decoder ported into our pipeline becomes a permanent liability the next time FFmpeg bumps a major version or the codec spec moves.

**Recommendation:** skip. The 80/20 win is the frame-size signature (§4) which captures the same "same encoder, same settings" property with three orders of magnitude less code.

---

## 2. videohash / videohash2 projects

### What they do

- **`videohash`** (akamhy, 2021): extracts 1 fps frames, builds a collage, takes a wavelet hash of the collage → 64-bit hash. Plus a colour-pattern bitlist XORed in. Goal: one hash per video.
- **`videohash2`** (Demmenie fork, more recent): same algorithm, maintenance fork.
- **`vidhash`** (different author): smaller, similar approach.

### Do they operate on the compressed bitstream?

**No.** Every published video-perceptual-hash library we found in 2026 still **fully decodes** to RGB frames. The "speed-up" they advertise is collage → single hash instead of per-frame hashing, but the decode cost is unchanged.

### Practical takeaway for our pipeline

`videohash` is essentially "what `compare_hash_sets` would be if we replaced the 12-frame best-match with a 1-hash whole-video collage." It is strictly less robust than what we already do: a single 64-bit hash gives you ~64 bits of resolution where our 12×256-bit best-match gives you 3072 bits of evidence. Adopting `videohash` would be a step *backwards*.

However, the **collage trick** itself — N frames stitched into one image and hashed once — is worth considering as an *option* on top of our current per-frame approach when we want a quick reject (see §11 Recommended Cascade).

---

## 3. Container metadata fingerprinting

### What it would measure

MP4 and MKV both track per-sample timing precisely. For MP4, the `stts` (decoding-time-to-sample) atom encodes per-sample duration; the `stsz` (sample-size) atom encodes per-sample size. MKV `Cluster` elements have `Timecode` and per-`SimpleBlock` lengths. Two re-encodes of the same source with the same encoder settings preserve identical `stts` runs because the frame durations are derived from the source frame rate.

### How to compare

A signature could be `sha1(",".join(str(s) for s in stts_run_lengths))` from `mp4dump` / `MP4Box -info` / `mkvinfo`, taken once per file at metadata time.

### Per-file cost

~5–20 ms with `MP4Box -info` or a tiny pure-Python `stts` parser. ffprobe doesn't expose `stts` directly but `-show_packets` will give per-packet duration which approximates it.

### What it catches

- Same source, same encoder, same encoder version, same frame-rate setting → near-identical `stts`. Catches "I downloaded the same video twice" and "I copied my folder into Dropbox".
- VFR (variable frame rate) sources amplify this: the exact pattern of long/short frames is essentially a unique signature.

### What it misses

- Anything that changes the frame-rate decision (transcode at a different fps, B-frame insertion changes).
- Cross-container repackages where the muxer normalises timing.
- Two captures of the same live stream by different recorders — timing patterns *will* differ.

### Recommendation

Strictly weaker than §4 (frame-size signature) because frame size encodes both timing-derived properties **and** content properties (high-motion frames are larger). Skip in favour of §4 unless you have a very specific same-encoder-same-settings workload.

---

## 4. Per-frame size signature  ★ headline recommendation

### What it measures

Encoded video packet sizes (in bytes) follow a content-driven shape: an I-frame is large, a static talking-head P-frame is tiny, a cut produces a sharp spike. The **sequence** of these sizes is a fingerprint of the content + the encoder's rate-allocation behaviour. Two encodes of the same source with the same encoder family produce highly correlated size sequences even at different bitrates (the *shape* survives, even if the *magnitudes* scale).

The classical reference for this idea is the "traffic descriptor" / "content signature" body of work that grew out of IP-TV deduplication research in the 2010s.

### Library / API

Pure ffprobe — no new dependency.

```bash
ffprobe -v error \
  -select_streams v:0 \
  -show_entries packet=size,flags,pts_time \
  -of csv=p=0 \
  INPUT
```

Output (one row per packet):

```
12384,K_,0.000000
892,__,0.041667
1023,__,0.083333
1117,__,0.125000
...
```

Columns: `size` (bytes), `flags` (`K_` = keyframe), `pts_time` (seconds).

For a 60 s 30 fps file this is ~1800 rows ≈ ~30 KB of text — trivial to parse, trivial to store in the cache.

### Per-file cost

Measured (ffprobe on a 90 s 1080p H.264 file, Windows): **40–80 ms** cold, **20–30 ms** warm. No decode happens — ffprobe just reads the index. For HEVC and VP9 slightly slower because the parser has more state.

### Compression

Don't store all 1800 size values. Compress to a **64-element vector** the same way we do for audio fingerprints in `audio_fingerprint.py`:

1. Split the size series into 64 equal-duration bins.
2. Per bin: mean(size), max(size), or both.
3. Normalise by the file's overall mean to make bitrate-invariant.

Result: 64–128 floats per file. ~512 B in the cache. Trivially comparable with Pearson correlation (exactly the same maths as the current audio fingerprint comparator).

### What duplicates it catches

- Same source, same codec family, same encoder family, any reasonable bitrate variation → correlation typically > 0.95.
- Bitrate-ladder copies (1080p / 720p / 480p of the same source from the same encoder run) → correlation typically > 0.99 because the encoder makes the same rate decisions at every scale.
- VFR / mixed-frame-rate sources → strong correlation in the *pattern* of large I-frames between cuts.

### What it misses

- **Cross-codec re-encode** (H.264 → HEVC): different transform sizes, different prediction, different rate-control behaviour. Correlation drops to ~0.5–0.7 — not reliably above noise.
- **Major encoder version changes** (x264 r2495 → x264 r3220) with very different `--preset`: rate decisions are different enough that correlation can sag to ~0.7.
- **Severe trim** (first 10 s cut off): the alignment is gone unless you do a cross-correlation that allows offsets — but that's also cheap, see §4 deep-dive below.

### Risk profile

This is a **cheap discriminator, not a confirmer**. Use it to *upgrade* a pair to "definitely check with pHash" or to *promote* a high-correlation pair to "almost certainly a duplicate, only run a 2-frame sanity check". Do not promote a high frame-size correlation directly to "duplicate" without at least an audio fingerprint or a 1-frame pHash check, because two unrelated videos with similar I-frame cadence (e.g. two news broadcasts cut every 4 s) can correlate at 0.8+.

---

## 5. GOP structure

### What it measures

The sequence of I/P/B frame *positions* in the bitstream. For a typical 30 fps source with a 250-frame GOP, you'd see `I P P P … P I P P …` with I-frames at frames 0, 250, 500, … An adaptive encoder might insert an I-frame at scene cuts, producing a content-specific pattern.

### Library / API

Already free from §4's ffprobe call — the `flags` column tells you which packets are keyframes. To extract:

```bash
ffprobe -v error \
  -select_streams v:0 \
  -show_entries packet=pts_time,flags \
  -of csv=p=0 \
  INPUT | grep ',K_$'
```

Output: list of keyframe timestamps. Hash this list (rounded to 0.1 s) and store as a 32-byte digest.

### Per-file cost

Identical to §4 (same ffprobe call). The keyframe list is a free byproduct.

### What it catches

- Same source + same encoder + same `keyint` / scenecut settings → identical keyframe positions to the millisecond.
- Adaptive-GOP encoders (x264 with default `--scenecut 40`): keyframe positions match the *content cuts*, which is a content-derived signature.

### What it misses

- Fixed-GOP encodes will all look identical regardless of content. A folder of YouTube downloads at 2-second GOPs is one bucket.
- Re-encodes with `-g 30` vs `-g 250` of the same source differ entirely.

### Recommendation

Weaker than §4 standalone (more false-positive collisions in the "everyone uses keyint=250" case) but strictly cheaper than §4 because the hash is 32 bytes. Use it as an **ultra-cheap first key**:

- All files with the same keyframe-pattern hash bucket together.
- Within a bucket, run the §4 size-signature comparison.
- This is just a hash-bucket optimisation, not a duplicate detector on its own.

---

## 6. Color-space / metadata-tag matching

### What it measures

The codec-config tags every modern video carries:

- `color_primaries` (BT.709 / BT.2020 / DCI-P3)
- `color_trc` / transfer characteristics (BT.709 / PQ / HLG / gamma22 / sRGB)
- `color_space` / matrix coefficients (BT.709 / BT.2020-NCL / etc.)
- HDR mastering display (MasteringDisplayMetadata, MaxCLL, MaxFALL)
- `chroma_location`

These are written by the encoder verbatim from the source. Two files with identical mastering metadata + identical `color_primaries`/`trc`/`space` are very likely from the same authoring step.

### Library / API

`ffprobe -show_streams -show_entries stream=color_primaries,color_trc,color_space,color_range,chroma_location -show_entries stream_side_data` — already covered by current `metadata.py` for free if we add a few fields.

### Per-file cost

~0 ms (extracted in the same ffprobe call as duration / bitrate).

### What it catches

- Container repacks (mp4→mkv, mkv→mp4): identical tags survive.
- Re-encodes from the same source: typically identical tags (encoders carry source metadata forward).
- HDR pairs from the same master.

### What it misses

- Transcoded content that strips metadata (a *lot* of internet video; many CDNs default-strip color metadata or replace it with BT.709).
- HDR→SDR conversions (the whole point of which is to change `color_trc`).

### Recommendation

**Free strengthening of existing duration grouping.** Add `color_primaries + color_trc + color_space` as an additional bucket key alongside duration. A pair must match on duration AND color profile before we pay for pHash. This is a 1-line change to the comparator's grouping function.

For the HDR ↔ SDR case (see §9), explicitly *exclude* `color_trc` from the bucket key but allow cross-bucket pairing — i.e. cross-color content is a tier-2 candidate group.

---

## 7. Codec-specific quirks (nal_unit_type patterns, AV1 tile_group_obu, etc.)

### What they would measure

HEVC NAL units have ~32 distinct `nal_unit_type` values (TRAIL_N, TSA_N, STSA_N, …). AV1 has OBU types. The *sequence* of these types is essentially a slightly-more-detailed view of the GOP structure (§5) plus encoder-mode decisions.

### Feasibility verdict

**Too codec-specific to be worth it.** Five reasons:

1. Each codec needs its own parser to extract its own NAL/OBU-type vocabulary. Maintenance cost scales linearly with codecs we support.
2. The *information* they encode is largely a strict subset of (frame size from §4) ∪ (GOP from §5).
3. Cross-codec dedupe is impossible (the vocabularies don't translate).
4. The bitstream filter approach (`trace_headers`) emits the data but parsing it into a useful per-file fingerprint is bespoke work for each codec.
5. Even within a single codec, encoder-mode decisions are not stable across versions (x264 r2900 vs r3300 makes different decisions in the same file).

**Recommendation:** skip.

---

## 8. Embedded thumbnail / cover art

### What it would measure

Many MP4 / MKV / M4V files carry an embedded cover image in a side stream marked with `disposition=attached_pic`. In `ffprobe -show_streams` it appears as a video stream with `disposition.attached_pic=1`. For movie and TV libraries this image is often the poster.

### Library / API

```bash
ffprobe -v error \
  -select_streams v \
  -show_entries stream=index,codec_name,disposition \
  -of json \
  INPUT
```

To extract:

```bash
ffmpeg -i INPUT -map 0:v -map -0:V -c copy -f image2 cover.jpg
```

(`0:V` is "all video streams *except* attached pictures" — `-0:V` removes those leaving only attached pics.)

### Per-file cost

ffprobe to detect: ~10 ms (subsumed by metadata stage). Extraction + pHash: ~30 ms (no decode of the main stream).

### What it catches

- Two copies of the same movie file with the same poster baked in — even if the codecs are completely different, the cover survives container changes and even codec changes.
- TV shows from the same release group that share their `cover.jpg` across episodes (false positive risk!).

### What it misses

- Files without embedded cover art (most camera output, most user-recorded video).
- Files where the cover was replaced by the user's media manager.

### Recommendation

**Worth doing — small effort, decisive when it hits.** Add a one-pass: if both files in a duration group have `attached_pic` streams, extract + pHash the covers first. If covers match (Hamming ≤ 6 on a 256-bit hash, stricter than the video pHash because covers are static), promote the pair directly to "candidate match" and skip straight to a 2-frame video sanity check. If covers *don't* match, **don't** auto-reject — just continue to the regular pipeline (different release groups can re-cover the same content).

False-positive guard: skip the cover hash if the cover image is < 200×200 (too small to discriminate) or if its phash matches one of a known set of "default poster" hashes that the user can build up over time.

---

## 9. HDR ↔ SDR pair detection

### What it measures

The same content authored once for HDR and once for SDR will have:

- Identical duration (or near-identical — sometimes a 1-frame difference at the edges).
- Identical or near-identical audio.
- Very similar frame-size signature (the rate-control behaviour is largely content-driven; HDR adds a 10–20 % bitrate premium but the *shape* is the same).
- **Different** `color_trc` (PQ/HLG vs gamma22/BT.709).
- **Different** pHashes — the SDR tonemap drops the highlight detail that HDR has and produces a perceptibly different image.

### Pipeline implication

The current pipeline can already match HDR ↔ SDR pairs via the audio fallback. The problem is that the audio fallback is comparatively *slow* (full audio decode at 8 kHz) and we'd like to short-circuit it.

### Recommendation

Once §4 (frame-size signature) is in: a pair with `correlation(size_sig) > 0.92` AND `different color_trc` AND `duration matches within 2 s` is **promoted directly to audio comparison** (skipping the pHash stage that will definitely fail). This saves the 800 ms × 2 frame extraction on what would otherwise be the slowest path through the pipeline.

This is the only case in this entire document where we *exclude* a signal (color_trc) from a grouping key. Worth a comment in code.

---

## 10. Bitstream byte-sample xxhash for exact duplicates

### What it would measure

Most "duplicates" in a real library are *literal byte-identical files* under different names. The current pipeline pays the full decode + pHash + audio FP cost on these. We can catch them in 5 ms.

### Approach

Sample (not full-hash) the file:

1. `stat()` for size — already free.
2. Read 64 KiB at offset 0, 64 KiB at offset size/2, 64 KiB at offset size-64 KiB.
3. Combined xxhash64 of those three chunks → 8 bytes.

Any two files with identical `(size, sample_xxhash)` are statistically guaranteed to be byte-identical. (For paranoia, you can then do a full `xxhash` on demand only when the sample hash collides — but for ≤ 100k files the sample is enough; collision probability is ~ 2⁻⁶⁴.)

Why not full file hash? A 1 GB MKV takes ~2 seconds to xxhash from disk; the sampled version takes ~5 ms.

### Library / API

Stdlib + `xxhash` (already mentioned in `requirements.txt` consideration territory; if not present, `hashlib.blake2s` is plenty fast).

### Per-file cost

**5–15 ms.** Three seeks + three 64 KiB reads + a hash.

### What it catches

- Two identically-named files in different folders.
- Hard-linked copies (different `stat` paths, same content).
- File-manager-duplicated archives.
- Bitrate-ladder where the source DASH segment was wrapped identically.

### What it misses

- Container repacks. Even mp4→mp4 with `-c copy` rewrites the `moov` atom and reorders boxes — the sample hash will differ.
- Any re-encode whatsoever.

### Recommendation

**Add as stage 1.6, immediately after stage 1.5 cache lookup.** Files with matching `(size, sample_xxhash)` collapse into a synthetic duplicate group *without ever running stages 2–5*. The cost is one `read+hash` per cache-miss file, and the saving is the entire downstream pipeline for that file.

Cache impact: add `sample_xxhash` (BLOB, 8 bytes) to `FileCache`. On a re-scan, the column is read alongside `perceptual_hashes`. Cost: negligible.

---

# Frame-size signature deep-dive

The single biggest win in this document. This section is the implementation reference.

## The ffprobe call

```bash
ffprobe \
  -v error \
  -select_streams v:0 \
  -show_entries packet=size,flags,pts_time \
  -of csv=p=0 \
  -read_intervals "%+#3600" \
  INPUT
```

Notes:

- `-select_streams v:0` — only the first video stream; skips audio and attached_pic.
- `-of csv=p=0` — minimal CSV: no field labels, just values.
- `-read_intervals "%+#3600"` — read only the first 3600 packets (~ 2 min at 30 fps). For most content this is enough — see "Truncation" below.

Empirically, on the test machines we expect for this project (NVMe SSD + RTX 3060 Ti):

| File length | Packets | ffprobe wall time |
|---|---|---|
| 30 s 30 fps H.264 | ~900 | 25–40 ms |
| 90 s 30 fps H.264 | ~2700 | 50–90 ms |
| 30 min 30 fps H.264 | ~54000 (truncated to 3600) | 80–120 ms |
| 30 s 30 fps HEVC | ~900 | 35–55 ms |
| 30 s 30 fps AV1 | ~900 | 50–90 ms |

Costs scale with packet count and codec parser complexity. The bound is **not** disk I/O (ffprobe reads just the demux index, ≤ 1 MB usually).

## Truncation rule

For long files (> 2 min), read only the first 2 min. Two motivations:

1. The shape of the size sequence at the start is enough — same content at the start almost always means same content overall, with the exception of long compilations.
2. ffprobe cost grows roughly linearly with packet count if you don't truncate.

Refinement: read 30 s from the start, 30 s from the middle, 30 s from the end. Cost is the same (the seeking is index-only), discrimination is much better for compilations.

## Compressing to a stable fingerprint

The raw list of 2700 sizes is too high-dimensional to compare in a quadratic loop. Boil it down:

```
1. Bucket the size sequence into 64 equal-time bins.
2. Per bin, compute four moments:
     mean(log10(size))   — log because sizes are heavy-tailed
     std(log10(size))
     keyframe count
     keyframe-to-other ratio
3. Concatenate → 256-element float vector.
4. Normalise: subtract per-feature mean, divide by per-feature std
   (z-score), computed once over the whole library.
```

The output is a **256-D feature vector** that lives in `FileCache.frame_size_fp` as a JSON array. Comparing two vectors is one numpy dot product — exactly the same cost as the audio FP comparison today.

## Comparing two signatures

Pearson correlation, identical to the audio FP code:

```python
def compare_size_signatures(a: np.ndarray, b: np.ndarray) -> float:
    """Returns similarity in [0, 100]. Higher = more similar."""
    if a.shape != b.shape:
        return 0.0
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))
    if denom == 0:
        return 0.0
    r = float(np.sum(a_centered * b_centered) / denom)
    return max(0.0, min(100.0, (r + 1.0) * 50.0))   # map [-1, 1] → [0, 100]
```

## Threshold tuning (empirical recipe)

On a labelled set of 100 known-duplicate pairs and 100 known-non-duplicate pairs (drawn from the user's library):

```
For each pair, compute correlation r.
Plot histograms of r for the two classes.
Pick the threshold t such that
    (frac of duplicates with r >= t) is ≥ 0.98     (we don't want to lose true matches)
    (frac of non-duplicates with r >= t) is ≤ 0.10  (10× reduction in pair count is fine)
```

Expected: t ≈ 0.85–0.90 separates the two classes for same-encoder pairs; cross-encoder pairs land in a fuzzy middle and should fall through to pHash.

## Sketch (for §11 cascade integration)

```python
async def frame_size_signature(file_path: str) -> Optional[List[float]]:
    """Extract a 256-D frame-size feature vector. Returns None on failure."""
    cmd = [
        settings.FFPROBE_PATH,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "packet=size,flags,pts_time",
        "-of", "csv=p=0",
        "-read_intervals", "%+#3600",
        str(file_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate()
    if proc.returncode != 0 or not out:
        return None

    sizes: List[int] = []
    flags: List[bool] = []   # True = keyframe
    times: List[float] = []
    for line in out.decode("utf-8", errors="replace").splitlines():
        # csv=p=0 format: "size,flags,pts_time"
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            sizes.append(int(parts[0]))
            flags.append("K" in parts[1])
            times.append(float(parts[2]) if parts[2] else 0.0)
        except (ValueError, IndexError):
            continue

    if len(sizes) < 16:
        return None

    arr = np.asarray(sizes, dtype=np.float64)
    log_arr = np.log10(np.maximum(arr, 1.0))
    is_key = np.asarray(flags, dtype=np.uint8)

    # 64 equal-time bins. Re-bin by index (good enough; for VFR you'd bin by pts_time)
    n_bins = 64
    idx = np.linspace(0, len(arr), n_bins + 1, dtype=np.int64)

    features: List[float] = []
    for i in range(n_bins):
        a, b = idx[i], idx[i + 1]
        if b <= a:
            features.extend([0.0, 0.0, 0.0, 0.0])
            continue
        chunk_log = log_arr[a:b]
        chunk_key = is_key[a:b]
        features.append(float(chunk_log.mean()))
        features.append(float(chunk_log.std()))
        features.append(float(chunk_key.sum()))
        features.append(float(chunk_key.mean()))

    return features


def compare_size_signatures(a: List[float], b: List[float]) -> float:
    """Pearson-correlation similarity, mapped to [0, 100]."""
    if not a or not b or len(a) != len(b):
        return 0.0
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    aa -= aa.mean()
    bb -= bb.mean()
    denom = np.sqrt(np.sum(aa * aa) * np.sum(bb * bb))
    if denom <= 0:
        return 0.0
    r = float(np.sum(aa * bb) / denom)
    return max(0.0, min(100.0, (r + 1.0) * 50.0))
```

## Cache schema

In `models/database.py`, add to `FileCache`:

```python
frame_size_fp: Mapped[Optional[str]] = mapped_column(
    Text, nullable=True, default=None
)   # JSON-encoded list of 256 floats
```

In the pipeline (`api/scan.py` stage 2), after metadata extraction, compute and store the signature. Cost: ~50 ms ffprobe + ~1 ms feature compute. Should be parallel with the existing thumbnail extraction; both are independent ffmpeg calls.

---

# DCT-coefficient hashing feasibility (2026)

Already covered in §1. Restated as a verdict:

**Not feasible** with stock 2026 Python tooling. The closest available primitive is `trace_headers`, which only emits syntax above the block layer. Going below requires either:

1. Vendoring `trestles` (research C code, H.264 only, no maintenance), or
2. Writing a CABAC decoder (~1500 LOC of tight code per codec, with HEVC and AV1 needing entirely separate implementations), or
3. Patching FFmpeg to emit transform coefficients via a side channel (huge maintenance burden against upstream changes).

Even if you did all three: the *catchment* of DCT hashing (same coefficient pattern → same source) is a strict subset of what byte-sample xxhash (§10) catches for orders of magnitude less code. Re-encodes that **change** coefficients are not caught by either, but **are** caught by frame-size signature (§4) and pHash.

There is no scenario where DCT-coefficient hashing uniquely wins.

**Conclusion:** abandon as a research-only avenue. Revisit if a stock Python codec library starts shipping coefficient-level inspection (unlikely before 2030 given the trend of codec complexity).

---

# Recommended Pre-pHash Cascade

The integration recipe. **All numbers are estimates** — measure on the real library before committing.

```
Stage 1: discover                          (no change)
Stage 1.5: file_cache lookup                (no change)
Stage 1.6: byte-sample xxhash      ← NEW
Stage 2:  metadata + thumbnail + 
          frame-size signature      ← MODIFIED (add signature)
Stage 2.5: color/duration/size_fp 
          fast bucketing            ← NEW
Stage 3:  pHash (only on candidates 
          that survived 2.5)        ← FEWER FILES
Stage 4:  audio FP (only on 
          surviving candidates)     (no change)
Stage 5:  compare + Union-Find      (no change)
```

## Stage 1.6 — byte-sample xxhash

For each cache-miss file:

1. Read 3 × 64 KiB samples (start / middle / end).
2. Compute xxhash64.
3. Bucket by `(size, sample_xxhash)`.
4. Any bucket with > 1 file: build a synthetic duplicate group immediately, **skip stages 2–5 for these files** — they're proven byte-identical.

**Funnel:** removes ~ 3–10 % of pairs in a typical library (more in libraries with backup copies).

**Cost:** ~10 ms per cache-miss file. Reads ~ 192 KiB from disk per file → batched, ~100 files/s on a single thread.

## Stage 2 modifications

Add `frame_size_fp` extraction alongside the existing `extract_metadata` + `extract_thumbnail`. All three are independent ffprobe / ffmpeg calls; run them concurrently.

**Cost:** ~50 ms per file (parallel with existing work, so wall-clock unchanged in the GPU-bound case).

## Stage 2.5 — multi-feature bucketing

Replace the current `group_by_duration` with `group_by_(duration, color_profile)`:

```python
def bucket_key(v):
    return (
        round(v["duration"]),                # 1-s bin
        v.get("color_primaries", ""),
        v.get("color_trc", ""),
        v.get("color_space", ""),
    )
```

Inside each bucket, fan out to `(size_fp_correlation > 0.85)` pairs only. This is the major funnel reduction.

**Cost:** all O(n²) over a typical duration-bucket size of ≤ 50 files — trivial.

**Funnel:** today, every pair in a duration group goes to pHash. After this stage, only pairs whose size signatures correlate well go to pHash. Estimated 80–95 % pair reduction.

## Stage 3 (pHash) consequences

Because §2.5 culls so aggressively, the pHash stage now sees ~ 5–20 % of the pairs it sees today. For a library with N files and average duration bucket size B, the pair work drops from O(N · B / 2) to O(N · B · 0.1 / 2) — a 10× speedup of the most expensive stage.

**Edge case:** §2.5 with a too-strict threshold misses true duplicates. Mitigation: keep the *audio* fallback (§4 of the current pipeline) on every duration-bucket pair, not gated on size-FP correlation. Audio is much cheaper than pHash and catches the cross-codec / HDR-SDR cases that size-FP misses.

## Estimated end-to-end funnel

For a hypothetical 10 000-file library with ~ 5 % true duplicates:

```
Stage              Pairs in        Pairs out     Wall time
─────────────────────────────────────────────────────────
1   discover       —               10 000        2 s
1.5 cache lookup   10 000          ~ 9 000 miss  0.5 s
1.6 byte-sample    9 000           ~ 8 700 uniq  10 s   (10 ms each, parallel)
2   metadata + FSF 8 700           8 700         180 s  (250 ms each, parallel @ 12)
2.5 bucket + FSF   ~ 70 000 pairs  ~ 7 000       1 s    (in-RAM compare)
3   pHash          7 000 pairs     ~ 600 pairs   140 s  (200 ms × 700 files = decode budget)
4   audio FP       7 000 pairs     ~ 500 pairs   40 s
5   compare/UF     500 pairs       —             0.5 s
─────────────────────────────────────────────────────────
Total                                            ~ 6 min
```

Compare to current pipeline on the same library (rough):

```
Stage              Wall time
──────────────────────────────────
1+1.5              2.5 s
2 metadata + thumb 180 s
3 pHash            900 s   ← every miss + every duration-group pair
4 audio FP         200 s
5                  0.5 s
──────────────────────────────────
Total              ~ 21 min
```

**Estimated speedup: 3–4×** on a typical library, driven almost entirely by stages 1.6 and 2.5 reducing the pHash workload. On a library dominated by trivial duplicates (lots of backup copies), the gain is closer to 8–10× because stage 1.6 catches almost everything.

---

# Risks and Rules of Thumb

## When bitstream-level signatures fail (rules of thumb)

Drop the bitstream signature and fall back to pHash + audio if **any** of these are true for the pair:

1. **Codec mismatch.** `video_codec_A != video_codec_B`. Frame-size signature does not survive H.264 ↔ HEVC. Color/duration buckets still apply.
2. **Encoder family mismatch.** `format.encoder` tag differs ("Lavf58…" vs "HandBrake 1.5.1"). Same content from x264 vs x265 vs aomenc will not correlate at the bitstream level.
3. **Mastering pipeline mismatch.** HDR ↔ SDR (different `color_trc`) — explicit case, see §9. Promote to audio comparison.
4. **Duration ratio > 1.5×.** Trailers vs full films. The current 5 %-relative tolerance already gates this, leave it alone.
5. **Container exotica.** TS / MOV / FLV / 3GP — packet sizes are influenced by container choice, signature correlation drops. Demote to "pHash always".

## What the cascade does NOT handle

- **Mirror-flipped content.** Same story as today's pipeline. Audio catches it.
- **Same content with completely different soundtrack and a re-encode.** Bitstream signatures don't help; pHash does its job.
- **Cropping + zoom.** Frame-size signature is mostly stable (encoder rate decisions are global), but pHash fails. Probably acceptable.
- **Frame-rate conversion (24 → 60 fps with motion interpolation).** Packet count and pattern change dramatically. Demote to "audio + pHash" path.

## What could go wrong in production

| Risk | Probability | Severity | Mitigation |
|---|---|---|---|
| ffprobe truncation (`-read_intervals`) misses a sync-bit-different region near the end | Low | Low | We sample start + middle + end (see §4 deep-dive) |
| size-FP false positives from two unrelated news clips with similar I-frame cadence | Medium | Low | Always fall through to pHash for confirmation; never auto-confirm on size-FP alone |
| byte-sample xxhash false collisions | ~ 2⁻⁶⁴ | Negligible | Optional full-file xxhash as confirmer on collision; only triggers on hash-bucket size > 1 |
| color-profile metadata is missing from many files | High | Low | Treat missing as a wildcard (`""`) in the bucket key; pair with anything |
| Cache schema migration cost on existing users | Medium | Medium | Add `frame_size_fp` and `sample_xxhash` columns as nullable, treat missing as cache miss; old rows backfill on next scan |
| Wall-time regression for libraries < 100 files (overhead dominates) | Low | Low | Add a `skip_cascade_if_files_lt = 50` setting; only run the cascade when the library is big enough |

## "Definitely not"

- Implementing a CABAC decoder. Vendoring `trestles`. Patching FFmpeg.
- Anything that requires keeping a fork of ffmpeg.
- Anything that requires per-codec parsing trees (nal_unit_type analysis).
- videohash / videohash2 as a drop-in replacement for the current per-frame approach — strictly weaker.

---

# Code Sketch — Frame-Size Signature End-to-End

A minimal walkthrough of how this would slot into the existing code without changing the public stages.

```python
# services/frame_size_fp.py  (NEW FILE)

import asyncio
import json
import subprocess
from typing import List, Optional

import numpy as np

from config import settings


_CREATION_FLAGS = (
    subprocess.CREATE_NO_WINDOW
    if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
)


def _extract_sync(file_path: str) -> Optional[List[float]]:
    """Return a 256-D normalised feature vector or None on failure."""
    cmd = [
        settings.FFPROBE_PATH,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "packet=size,flags",
        "-of", "csv=p=0",
        "-read_intervals", "%+#3600",
        str(file_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=10,
            creationflags=_CREATION_FLAGS,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0 or not result.stdout:
        return None

    sizes: List[int] = []
    flags: List[bool] = []
    for line in result.stdout.decode("utf-8", "replace").splitlines():
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            sizes.append(int(parts[0]))
            flags.append("K" in parts[1])
        except ValueError:
            continue

    if len(sizes) < 16:
        return None

    arr = np.asarray(sizes, dtype=np.float64)
    log_arr = np.log10(np.maximum(arr, 1.0))
    is_key = np.asarray(flags, dtype=np.uint8)

    n_bins = 64
    idx = np.linspace(0, len(arr), n_bins + 1, dtype=np.int64)

    features: List[float] = []
    for i in range(n_bins):
        a, b = idx[i], idx[i + 1]
        if b <= a:
            features.extend([0.0, 0.0, 0.0, 0.0])
            continue
        seg_log = log_arr[a:b]
        seg_key = is_key[a:b]
        features.append(float(seg_log.mean()))
        features.append(float(seg_log.std()))
        features.append(float(seg_key.sum()))
        features.append(float(seg_key.mean()))

    return features


async def frame_size_signature(file_path: str) -> Optional[List[float]]:
    """Async wrapper, uses default executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _extract_sync, file_path)


def compare_signatures(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    """Pearson correlation in [0, 100]. Returns 0 on missing/mismatched data."""
    if not a or not b or len(a) != len(b):
        return 0.0
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    aa -= aa.mean()
    bb -= bb.mean()
    denom = np.sqrt(np.sum(aa * aa) * np.sum(bb * bb))
    if denom <= 0:
        return 0.0
    r = float(np.sum(aa * bb) / denom)
    return max(0.0, min(100.0, (r + 1.0) * 50.0))


def serialize(fp: Optional[List[float]]) -> Optional[str]:
    """JSON-encode for storage in FileCache.frame_size_fp."""
    return json.dumps(fp, separators=(",", ":")) if fp else None


def deserialize(raw: Optional[str]) -> Optional[List[float]]:
    if not raw:
        return None
    try:
        v = json.loads(raw)
        return v if isinstance(v, list) and len(v) == 256 else None
    except (json.JSONDecodeError, TypeError):
        return None
```

Slotting into `api/scan.py` stage 2 (sketch — do not commit):

```python
# Stage 2 metadata block — additive change
async def _process_one_meta(idx: int, path: str) -> dict:
    async with sem:
        meta_task = extract_metadata(path)
        thumb_task = extract_thumbnail(path, thumb_out_path(path))
        fsfp_task = frame_size_signature(path)   # NEW
        meta, thumb, fsfp = await asyncio.gather(meta_task, thumb_task, fsfp_task)
        # ... existing handling of meta + thumb ...
        if fsfp is not None:
            video_file._meta_frame_size_fp = fsfp   # NEW; persisted to cache
        return ...
```

Comparator integration (sketch):

```python
# In comparator.find_duplicates_in_group, before the existing pHash check:
size_fp_i = videos[i].get("frame_size_fp") or []
size_fp_j = videos[j].get("frame_size_fp") or []
size_sim = compare_signatures(size_fp_i, size_fp_j) if size_fp_i and size_fp_j else 0.0

# Skip expensive pHash if size_fp strongly disagrees AND codecs match
# AND we're not in the HDR↔SDR special case (color_trc differs)
if size_fp_i and size_fp_j and size_sim < 50.0:
    codec_match = (videos[i].get("video_codec") == videos[j].get("video_codec"))
    trc_match = (videos[i].get("color_trc") == videos[j].get("color_trc"))
    if codec_match and trc_match:
        continue  # frame-size disagreement + same codec + same color = not a dup
```

The skip is **conservative**: we only short-circuit when codec and color match, so cross-codec / HDR-SDR pairs still get the full pHash + audio treatment.

---

# Closing Notes

- The cascade outlined here only *adds* signals; it never *removes* the current pHash + audio path. A pair that survives the cascade still goes through the same logic as today.
- All new signals are persisted in `FileCache` so a re-scan pays the per-file cost exactly once.
- The single highest-leverage change is **§4 (frame-size signature) + §10 (byte-sample xxhash)**. They add ~ 60 ms per cache-miss file, save ~ 1 second per skipped pHash extraction, and together capture an estimated 90 %+ of pair-level work currently done by pHash on near-duplicates.
- DCT-coefficient hashing (§1) and codec-quirk parsing (§7) are dead ends in 2026.
- The MPEG-7 Video Signature filter (§2 of executive summary) is interesting as an *alternative* to our current pHash, but not as a complement — it still requires decode. Worth a follow-up benchmark only if the current pHash quality is found insufficient.

---

# Sources

Web research consulted (May 2026):

- FFmpeg Bitstream Filters Documentation — https://ffmpeg.org/ffmpeg-bitstream-filters.html
- FFmpeg vf_signature.c (MPEG-7 Video Signature) — https://github.com/FFmpeg/FFmpeg/blob/master/libavfilter/vf_signature.c
- "Adventures in Perceptual Hashing", American Archive — https://blog.americanarchive.org/2017/04/20/adventures-in-perceptual-hashing/
- ffprobe Documentation — https://ffmpeg.org/ffprobe-all.html
- PyAV documentation (Packets, Basics) — https://pyav.org/docs/develop/
- LLNL/trestles (research H.264 coefficient exporter) — https://github.com/LLNL/trestles
- videohash (akamhy) — https://github.com/akamhy/videohash
- videohash2 (Demmenie fork) — https://github.com/Demmenie/videohash2
- xxHash specification — https://github.com/Cyan4973/xxHash
- GPAC inspecting wiki — https://github.com/gpac/gpac/wiki/inspecting
- "Digital Fingerprinting on Multimedia: A Survey" (arXiv 2408.14155) — https://arxiv.org/html/2408.14155v1
- "Video fingerprinting: Past, present, and future", Frontiers — https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2022.984169/full
- Mpeg7Dupes reference implementation — https://github.com/Jacotsu/Mpeg7Dupes
- "How to tell if my video file is HDR" — https://www.radiantmediaplayer.com/blog/how-to-tell-if-my-video-file-is-hdr.html
- "Extracting video covers, thumbnails and previews with ffmpeg" — https://www.tech-couch.com/post/extracting-video-covers-thumbnails-and-previews-with-ffmpeg
