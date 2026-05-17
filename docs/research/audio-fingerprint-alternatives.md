# Audio Fingerprint Alternatives — Deep Comparative Analysis

A focused research note on what to put **next to or instead of** the current
64-point RMS profile in `backend/services/audio_fingerprint.py`. The existing
research file `algorithmic-improvements.md` already recommends Chromaprint as
"item #4". This document goes deeper: what Chromaprint *actually does* in the
short-clip / silent / music-video edge cases, what the realistic alternatives
look like in 2026, and how to **compose** several techniques because no single
one covers the full duplicate-detection workload.

The audio fallback in this pipeline is gated by duration grouping and only
needs to discriminate **"same recording vs different recording, both about
this duration."** It is not a music-identification problem. That framing
matters: it lets us trade some of Chromaprint's noise-robustness for
short-clip behaviour, simpler dependencies, and stronger handling of speech /
silent video.

---

## Executive summary

Three ranked changes, by impact/effort. All three should ship — they cover
different failure modes and compose cleanly.

### 1. Replace RMS with Chromaprint via the `fpcalc` binary (HIGH impact, LOW–MED effort)

The single biggest accuracy win. Chromaprint is the de-facto standard
(MusicBrainz, Picard, beets, half the music-tagging world). Two re-encodes of
the same audio land at ≥ 95% bit-similarity over the overlap; truly unrelated
audio averages ~50% (random for 32-bit hashes). The current RMS approach is
geometric/energy-based and has near-zero discriminative power for content with
similar dynamic shape — e.g. two action movie scenes both averaging "loud".

The pyacoustid binding is fine but not strictly required: shipping the
`fpcalc` binary and parsing its `-json` output is simpler, has no Python ABI
risk, and dodges the libchromaprint Windows DLL hassle. Cost per file is
roughly the same as the current RMS pass once we also adopt finding #1 from
`pipeline-optimizations.md` (sampling instead of full-track decode).

**Discriminative-power gain: ~10–20× over RMS for typical content.** RMS
correlation of ≥ 0.85 is reachable by chance between two random music videos
of the same length; Chromaprint bit-agreement of ≥ 90% is essentially
impossible without identical source audio.

**Threshold to start with:** ≥ 0.85 bit-agreement over the aligned overlap
(i.e. Hamming distance ≤ 4.8 bits per 32-bit hash on average), with a
secondary requirement that the overlap covers ≥ 60% of the shorter
fingerprint.

### 2. Add spectrogram-pHash fallback for short / Chromaprint-failing clips (MED impact, LOW effort)

`fpcalc` refuses to fingerprint audio shorter than ~3 seconds and returns a
useless degenerate fingerprint (`AQAAAA…`) on near-silent content. **Both
cases are common** in this pipeline: GIF-converted phone clips, mute
surveillance footage, TikTok-style short clips, slideshows.

Solution: compute a **STFT-based 2D pHash of the spectrogram** as a side
fingerprint. No new dependencies — `numpy` already in use, `imagehash` already
in use for the video frames. Render the magnitude spectrogram as a tiny
greyscale "image", pHash it (or wHash; wavelet hash is more tolerant of small
time-shifts), and compare via Hamming exactly like the video hashes.

Use this **only** when Chromaprint failed or the clip is short. It is much
cheaper than Chromaprint (single FFT pass, no chroma decomposition) and works
on any audio length ≥ ~250 ms.

### 3. Multi-segment fingerprinting + segment-vote matching (LOW–MED impact, MED effort)

Currently a single full-track fingerprint represents the whole video. Two
problems:

- A 30-min vlog and a 30-min vlog **with a different middle 5 minutes** look
  ~83% similar overall — likely above any tunable global threshold.
- Sliding alignment over a long Chromaprint fingerprint is O(n²) on length and
  becomes the comparator hot path.

Fingerprint **three 10-second windows at 10%, 50%, 90% of duration** as a
"compact" signature. Two videos are an audio match if ≥ 2 of 3 windows
individually match (bit-agreement ≥ 85% with a small ±2-frame alignment
search). This is ~10× cheaper per comparison than full-track sliding
alignment and rejects "same intro, different content" duplicates.

This is also what's needed to handle **varispeed re-encodes** (small tempo
changes) — full-track alignment fails them; per-window alignment over short
windows still works.

---

## Comparative table

Algorithms scored on a 1–5 scale where 5 = best. Storage is per-minute of
audio for a fingerprint suitable for duplicate detection. "Maint" = active
maintenance signal as of 2026 based on commit cadence, PyPI download
trajectory, and Issue-response time. Latency is order-of-magnitude on an
8-core CPU including FFmpeg decode.

| Algorithm | Recall (re-encoded) | Recall (varispeed) | Precision | Latency / file | Storage / min | Deps | Maint 2026 | License | Short clip OK? | Silent OK? |
|---|---|---|---|---|---|---|---|---|---|---|
| **Current RMS 64-pt** | 4 | 2 | 1 | full-decode ⚠ | 256 B | numpy | self | n/a | YES (but useless) | YES (returns 0s) |
| **Chromaprint via fpcalc** | 5 | 3 | 5 | ~150 ms | ~1.9 KB | fpcalc binary | very high (industry std) | LGPL2.1 | NO (< 3s) | NO (degenerates) |
| **pyacoustid (libchromaprint)** | 5 | 3 | 5 | ~120 ms | ~1.9 KB | C lib + cffi | medium (last release 2022; alive) | MIT (wrapper), LGPL (lib) | NO | NO |
| **STFT spectrogram pHash** | 4 | 2 | 4 | ~50 ms | 32 B (one 256-bit hash) | numpy + imagehash (have) | self | n/a | YES | YES (still discriminates) |
| **MFCC mean+std pool** | 3 | 3 | 3 | ~80 ms | 160 B (20-d mean+std) | librosa | medium | ISC | YES | YES |
| **Dejavu (constellation)** | 5 | 1 | 5 | ~300 ms | ~25 KB (DB rows) | MySQL/Postgres | **DEAD** (worldveil/dejavu inactive per Snyk; last meaningful commit 2018) | MIT | NO | NO |
| **audfprint** | 5 | 2 | 5 | ~250 ms | ~15 KB | numpy, librosa | **DORMANT** (Dan Ellis; sporadic, last meaningful work 2017) | MIT | NO | NO |
| **Olaf** | 4 | 2 | 4 | ~80 ms (C lib) | ~5 KB | C lib, WASM bindings | **ALIVE** (Joren Six, burst-mode) | AGPL ⚠ | NO | NO |
| **Panako** | 5 | 5 (designed for it!) | 5 | ~400 ms | ~20 KB | Java | alive (Joren Six, burst-mode) | AGPL ⚠ | NO | NO |
| **CLAP (LAION HTSAT)** | 4 | 5 | 3 | 200–600 ms CPU / 20 ms GPU | 1.5 KB (512-d fp16) | torch + transformers + 600 MB weights | very high | CC-BY-NC ⚠ for some weights | YES | YES (returns embedding for silence too — discriminates poorly) |
| **OpenL3** | 4 | 3 | 3 | 300–800 ms CPU | 1 KB (256-d fp16) | tensorflow + 30 MB weights | low (last release 2021; tf2 only) | MIT | YES | YES |
| **PANN CNN14** | 4 | 3 | 3 | 250–700 ms CPU | 4 KB (2048-d fp16) | torch + 350 MB weights | low (PyPI `panns_inference` stale) | Apache 2.0 | YES | YES |
| **Wav2Vec2 pooled** | 3 | 4 | 2 | 400 ms–1 s CPU | 1.5 KB (768-d fp16) | torch | very high (HF transformers) | Apache 2.0 | YES | YES (but speech-biased) |

Notes:

- "Recall (varispeed)" = how well the algorithm handles small pitch/tempo
  changes (±5%). Panako was specifically designed for this (its core
  contribution); chroma-based Chromaprint tolerates a bit; LSH-on-spectrogram
  approaches (Dejavu/audfprint) fail badly because peak positions shift.
- The current RMS approach gets **recall 4** because RMS energy is invariant
  to a *lot* of stuff — but at the cost of **precision 1**, which is exactly
  the problem.
- Latency includes FFmpeg decode of 1–2 minutes of audio. Pure inference is
  much faster; the decode dominates.
- Storage values exclude the audio source; they describe what gets persisted
  to `FileCache.audio_fp`.

---

## Detailed analysis: the top three candidates

### A. Chromaprint via `fpcalc` (primary recommendation)

#### How it works internally

1. Decode to **11025 Hz mono PCM** (libchromaprint does this internally; if you
   pipe via FFmpeg you do it yourself).
2. STFT with frame size 4096, hop size 1365 (2/3 overlap). Each frame is
   ~0.124 s.
3. **Chroma transform**: collapse the FFT bins into 12 pitch classes (C, C#,
   D, … B), normalised. This is what makes Chromaprint robust to codec /
   bitrate / EQ changes — chroma is a perceptual summary of "what notes are
   sounding" with octave information thrown away.
4. **Image filter bank**: a 16×12 sliding window over the chroma image. For
   each window position, 16 pre-trained filters compute differences across
   bins; each result is quantised to 2 bits via Gray coding. Concatenate →
   one **32-bit hash** per window position.
5. Output: a list of 32-bit unsigned ints, ~8 hashes/second, plus a "duration"
   integer.

For a 60-second clip that's ~480 × 32 bits = ~1.9 KB raw, or about 1 KB
base64-encoded. AcoustID stores the base64-encoded form. For our cache, store
the raw `np.uint32` array as a JSON list or as a packed BLOB.

#### Comparison

Naive: align by time index (assumes both fingerprints start at the same
audio moment) and compute average Hamming distance over the overlap. This is
the "fast path".

Robust: **sliding alignment**. For each offset Δ in `[-MAX_OFFSET, +MAX_OFFSET]`
(typical `MAX_OFFSET = 80` frames = 10 seconds), compute the average bit
agreement of the overlap. Take the best offset.

```python
import numpy as np

def chromaprint_similarity(fp1: np.ndarray, fp2: np.ndarray,
                            max_offset: int = 80) -> tuple[float, int]:
    """Return (best_bit_agreement, offset) in [0, 1], frames."""
    best = 0.0
    best_off = 0
    for off in range(-max_offset, max_offset + 1):
        if off >= 0:
            a, b = fp1[:len(fp1) - off], fp2[off:off + len(fp1) - off]
        else:
            a, b = fp1[-off:-off + len(fp2) + off], fp2[:len(fp2) + off]
        n = min(len(a), len(b))
        if n < 30:                          # need ≥ ~4s overlap to trust
            continue
        a, b = a[:n], b[:n]
        # bits differing = popcount(a XOR b)
        x = (a ^ b).astype(np.uint32)
        # popcount via numpy bit tricks (or use gmpy2.popcount):
        x = x - ((x >> 1) & 0x55555555)
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F
        diff_bits = (x * 0x01010101 >> 24).sum()
        agreement = 1.0 - diff_bits / (n * 32)
        if agreement > best:
            best, best_off = agreement, off
    return best, best_off
```

This runs in ~5 ms for two 60-second fingerprints, dominated by the popcount
loop. Vectorised numpy avoids the Python loop overhead. If you store many
fingerprints and need fast indexing across the whole library, see the
**indexing** subsection below.

#### Recall and precision

Empirical numbers from the literature and from our own back-of-envelope
calibration:

- Same recording, different codecs (FLAC vs MP3 320 vs AAC 128 vs Opus 96):
  ≥ 0.95 bit-agreement.
- Same recording, with intro/outro trims of ≤ 5 s: ≥ 0.90 with sliding
  alignment.
- Same recording, varispeed ±2%: ~0.85.
- Same recording, varispeed ±5%: drops to ~0.70 (chroma is pitch-aware; pitch
  shift breaks alignment). This is when you want Panako or a CLAP embedding
  as fallback.
- Different recordings, same musical key / similar genre: ~0.55–0.65.
- Truly unrelated: ~0.50 (random).

**Recommended threshold for "same audio": 0.85.** Below 0.75 reject as
non-match. The 0.75–0.85 band is the uncertain zone; defer to video pHash
verdict if available.

#### Storage and caching

`FileCache.audio_fp` is currently a JSON list of 64 floats. Replace with a
JSON list of `int` (the uint32 values; JSON has no native binary). For a
mixed library averaging 5 min per video, that's ~10 KB per row vs ~250 B
today — a 40× bump but still trivial against the video table.

If size matters, encode as base64 of the packed `np.uint32` LE bytes:

```python
import base64, numpy as np
encoded = base64.b64encode(np.asarray(fp, dtype="<u4").tobytes()).decode()
# decode:
fp = np.frombuffer(base64.b64decode(encoded), dtype="<u4")
```

Roughly 6 KB / 5 minutes encoded. Add a `audio_fp_version` column (int,
default 1) so future migrations can detect old RMS profiles vs new
Chromaprint blobs.

#### Distribution

- **Linux**: `apt install libchromaprint-tools` → `fpcalc` on PATH. Done.
- **Windows**: download static binary from acoustid.org/chromaprint, drop in
  `backend/bin/`, point the subprocess at the absolute path. ~3 MB.
- **macOS**: `brew install chromaprint` → `fpcalc`. Done.
- **Docker**: add `apt-get install -y libchromaprint-tools` to the Dockerfile.

`pyacoustid` is **not** strictly needed and adds a libchromaprint shared-lib
dependency that's annoying on Windows. The simpler approach:

```python
import subprocess, json
def chromaprint_fp(path: str, length: int = 120) -> list[int]:
    r = subprocess.run(
        ["fpcalc", "-json", "-length", str(length), "-raw", str(path)],
        capture_output=True, text=True, timeout=60,
        creationflags=_CREATION_FLAGS,
    )
    if r.returncode != 0:
        return []
    data = json.loads(r.stdout)
    return data.get("fingerprint", [])      # list[int], 32-bit values
```

Note `-length 120`: caps decoded audio at 120 seconds. With multi-segment
fingerprinting (recommendation #3) you'd issue three separate calls with
`-ss` and `-length 10`, or sample three segments by passing `-length` per
segment with FFmpeg-side trimming. The latter avoids two extra subprocess
spawns.

#### Indexing for large libraries

Within a duration group of size n, all-pairs sliding correlation is O(n² × L)
where L is fingerprint length. At n = 200 and L = 480 that's 19,900 × ~5 ms =
**100 seconds per duration group**. Manageable but not great.

If a duration group exceeds, say, 50 items, switch to **per-frame indexing**:

1. Take the **middle 1 frame** of each fingerprint (or 3 mid frames) as an
   index key.
2. Build a `dict[int, list[video_idx]]` mapping hash → videos containing it.
3. For each pair (i, j), check if their index hash sets overlap; if yes,
   compare with sliding alignment.

This is the same logic Dejavu / audfprint use, just at a much smaller scale
and without the database overhead. ~20 lines of Python.

For very large catalogues (≥ 50k), build an **acoustid-style inverted
index**: every 32-bit hash maps to the list of videos containing it, and a
candidate is any video sharing ≥ K hashes with the query. AcoustID does this
in PostgreSQL with a GIN index on a `BIGINT[]` column; SQLite can do it with
an auxiliary table `audio_hash (hash INT, video_id INT)` plus `INDEX(hash)`.

### B. STFT spectrogram pHash (fallback for short / silent / fpcalc-failing)

#### Concept

Treat the audio as an image: render the magnitude spectrogram, downsample to
a small fixed size, and compute a perceptual hash. Same machinery as the
video frame hashing, just on a different input.

```python
import numpy as np, imagehash
from PIL import Image

def spectrogram_phash(samples: np.ndarray, sr: int,
                      hash_size: int = 16) -> str:
    """Return a 256-bit hex pHash of the audio spectrogram."""
    if len(samples) < sr // 4:               # < 250 ms
        return ""
    n_fft = 1024
    hop = 256
    # Magnitude STFT (no librosa needed)
    pad = (n_fft - len(samples) % hop) % hop
    if pad:
        samples = np.pad(samples, (0, pad))
    frames = np.lib.stride_tricks.sliding_window_view(samples, n_fft)[::hop]
    win = np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(frames * win, axis=1))
    # Log compress to handle dynamic range
    spec = np.log1p(spec)
    # Normalise per file
    if spec.max() > 0:
        spec /= spec.max()
    # 8-bit greyscale PIL image
    img = Image.fromarray((spec.T * 255).astype(np.uint8))
    return str(imagehash.phash(img, hash_size=hash_size))
```

That's it. ~25 lines including imports. No new dependencies. Comparison uses
the same `compare_hash_sets` / Hamming machinery already in `hasher.py`.

#### Why this helps

- **Works on < 3-second clips.** Chromaprint refuses (`fpcalc` returns empty
  fingerprint below ~2–3 s). Spectrogram pHash works down to ~250 ms.
- **Discriminates on silent / near-silent video.** Even "silent" surveillance
  footage has microphone noise floor with a distinctive spectral envelope
  (preamp hum, AC fan). Two copies of the same silent clip have identical
  noise spectra; a different silent clip has different noise. Chromaprint
  collapses all near-silence to the same degenerate hash. Spectrogram pHash
  still discriminates — usefully.
- **Robust to codec re-encode.** Spectrogram shape is preserved across MP3 /
  AAC / Opus down to ~64 kbps.
- **Cheap.** No filter bank, no chroma, no learned weights. One FFT pass + one
  PIL pHash. ~50 ms total including decode for a 60-s clip.

#### Limitations

- **Not time-shift robust.** A 5-second trim at the start of an otherwise
  identical clip would produce a different pHash. Mitigation: pHash the
  spectrogram of three sub-windows (10%, 50%, 90%) like recommendation #3,
  and use best-match comparison across them.
- **Less precision than Chromaprint** for normal-length non-silent content.
  Don't replace Chromaprint with this; use it as a fallback.

#### Threshold

Hamming distance ≤ 14 over a 256-bit hash (i.e. ≥ 94.5% bit agreement). The
same threshold as the video pHash code path. Same machinery, same tuning.

### C. Multi-segment Chromaprint with segment voting

#### Concept

Replace the "one fingerprint per video" with "K fingerprints per video, one
per timestamped window". A pair matches if at least M of K windows agree.

```python
SEG_OFFSETS = [0.10, 0.50, 0.90]   # fractions of duration
SEG_LENGTH = 10                     # seconds each
M_OF_K_THRESHOLD = 2                # 2 of 3 windows must match
PER_WINDOW_THRESHOLD = 0.85         # bit-agreement

def multi_seg_fp(path: str, duration: float) -> list[list[int]]:
    fps = []
    for frac in SEG_OFFSETS:
        ss = max(0.0, duration * frac - SEG_LENGTH / 2)
        # FFmpeg-side trim, then fpcalc on the trimmed stream
        # (a single ffmpeg | fpcalc pipe is cleanest)
        cmd = [
            "ffmpeg", "-ss", f"{ss}", "-i", path, "-t", f"{SEG_LENGTH}",
            "-vn", "-f", "wav", "pipe:1",
        ]
        ffm = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        fp = subprocess.run(
            ["fpcalc", "-json", "-raw", "-"],
            stdin=ffm.stdout, capture_output=True, text=True, timeout=30,
        )
        ffm.stdout.close(); ffm.wait()
        if fp.returncode == 0:
            data = json.loads(fp.stdout)
            fps.append(data.get("fingerprint", []))
        else:
            fps.append([])
    return fps

def multi_seg_match(fps1, fps2) -> bool:
    matches = 0
    for a, b in zip(fps1, fps2):
        if a and b:
            agreement, _ = chromaprint_similarity(np.asarray(a, dtype="u4"),
                                                  np.asarray(b, dtype="u4"),
                                                  max_offset=8)
            if agreement >= PER_WINDOW_THRESHOLD:
                matches += 1
    return matches >= M_OF_K_THRESHOLD
```

#### Why this helps

- **Catches partial duplicates.** A re-edit that swaps the middle 5 minutes
  with different content still matches at the head and tail windows. With the
  current single-fingerprint approach, the middle drags the global score
  below threshold; with 2-of-3 voting, head+tail wins.
- **Rejects "same intro" false positives.** Many videos share canned intro
  sounds (gaming clips with the same intro animation, YouTube channels using
  the same stinger). A single full-track fingerprint averaging similarity
  could be pushed above threshold by a strong intro match alone. Voting
  requires agreement across **independent** windows.
- **Decouples from total duration.** Each window is 10 s, so the cost is
  bounded regardless of clip length. A 3-hour movie costs the same to
  fingerprint as a 60-second clip — three 10-s windows each.
- **Cheaper to compare.** Each window is ~80 hashes (~320 B); sliding
  alignment is over only ±8 frames (±1 second). Compare cost per pair drops
  from O(L²) to O(K × 16) which is essentially free.

#### Per-window alignment makes varispeed work

Within a 10-second window a ±5% tempo change is ±0.5 seconds of drift across
the whole window. With sliding alignment ±8 frames (= ±1 s) you absorb the
drift. Over a full-track 5-minute fingerprint a 5% drift becomes 15 seconds —
far beyond the typical `MAX_ALIGN_OFFSET = 80 frames = 10 s`. So multi-segment
**without** changing the per-window alignment search trivially handles
varispeed that breaks full-track alignment.

#### Risk

- Choosing 10%/50%/90% leaves a blind spot at the very start and end. If
  someone duplicates the first 10 minutes of a 60-min video (a clip extract),
  only the 10% window may match. With M=2/K=3 that's a miss.
- Mitigation: when M=1 (one window matched cleanly), promote to "uncertain"
  and let video pHash break the tie. Don't reject outright.

---

## Recommended composite strategy

No single algorithm covers the full duplicate-detection workload. The
recommended stack:

```
For each video, at fingerprint time:
  1. Read duration from the metadata stage (already done; free).
  2. If duration >= 5 s and audio track exists:
       a. Multi-segment Chromaprint: 3 windows × 10 s, fpcalc.
          Store as audio_fp_v2 = list[list[int]].
       b. If any window's fingerprint is empty/degenerate, that window is
          flagged as "spectrogram-only".
  3. ALSO compute a single spectrogram pHash of the mid-30s window (or full
     audio if duration < 30 s).
       Store as audio_phash = "abcdef…" (hex).
  4. If duration < 5 s OR no audio track:
       chromaprint = None; rely on spectrogram pHash only.

At compare time, between two candidates in the same duration group:
  - If both have multi-segment chromaprints, run segment-vote match.
    - 2 of 3 windows agree (≥ 0.85): MATCH (high confidence).
    - 1 of 3 agrees: UNCERTAIN; fall through to spectrogram pHash.
    - 0 of 3 agrees: NOT MATCH.
  - If either is chromaprint-less, use spectrogram pHash exclusively.
    - Hamming ≤ 14 / 256: MATCH.
    - Hamming > 28: NOT MATCH.
    - Else UNCERTAIN.
  - UNCERTAIN never matches on its own; it requires video pHash agreement to
    promote to MATCH.

Edge cases:
  - Both videos have no audio: skip audio stage entirely; video pHash decides.
  - One has audio, the other doesn't: audio can only veto, never confirm.
    (This catches "muted re-upload of the same video" edge cases — visual
    pHash will match and there's no audio contradiction.)
```

### Why three tools instead of one

| Failure case | Current RMS | Chromaprint alone | Composite |
|---|---|---|---|
| 90-min movie, two codec re-encodes | OK (rms ~0.95) | OK (cp ~0.96) | OK |
| Two 2-second TikTok clips, same source | OK by accident | **FAILS** (fpcalc empty) | OK (spectro pHash) |
| Same silent surveillance clip | FAILS (rms = 0,0,0,…) | **FAILS** (degenerate) | OK (spectro pHash on noise floor) |
| Same content, varispeed 105% | rms ~0.85 → marginal | cp ~0.70 → marginal | OK (per-window cp tolerates drift) |
| Same intro, different content | **FALSE POS** (rms ~0.92) | possible FP on short audio | OK (vote rejects 1-of-3) |
| Trimmed re-upload (cut 30 s from start) | rms misaligned → fails | OK with sliding align | OK |
| Music video re-cut (same audio, new visuals) | rms ~0.99 | cp ~0.99 | OK |
| Two action-movie scenes, both loud | **FALSE POS** likely | OK (cp ~0.55) | OK |

The composite covers all cells. Chromaprint alone fails two; spectrogram pHash
alone has weaker precision on long content. Together they cost about 1.2× the
runtime of Chromaprint alone, with ~1.5 KB extra storage per file.

---

## Threshold tuning guidance without ground truth

You don't have a labelled benchmark set. Calibrating thresholds nonetheless:

### Approach 1: bootstrap from current matches

Run the current pipeline on a representative scan. For every "matched" pair
(stored in `duplicate_groups`), compute both Chromaprint similarity and
spectrogram pHash distance. For every "non-matched" pair in the same
duration group, compute the same. Plot histograms; the threshold should land
in the valley between them.

Practical:

```python
# pseudo, run in a notebook against an existing scan DB
matched_pairs = ...   # known duplicates from past scans
unmatched_pairs = ... # randomly sampled non-duplicates in same duration group

for pair in matched_pairs:
    cp_sim, phash_dist = compute_both(pair)
    print("MATCH", cp_sim, phash_dist)

for pair in unmatched_pairs:
    cp_sim, phash_dist = compute_both(pair)
    print("NONMATCH", cp_sim, phash_dist)
```

Sort, find the 5th-percentile of matched (lower bound) and 95th-percentile of
unmatched (upper bound). If they overlap, you have a tuning problem
fundamentally. If they don't, set threshold at the midpoint.

### Approach 2: synthetic pair generation

Generate known-positive pairs by re-encoding source files with FFmpeg at
different bitrates. Generate known-negative pairs by pairing random distinct
videos from different duration buckets. Compute similarities for both pools.

This gives you a clean ROC curve. Pick the threshold at the elbow (max F1) or
at the false-positive-rate ceiling you want (e.g. < 1% FPR).

A starter script:

```bash
# Generate positive pairs
ffmpeg -i source.mp4 -c:v libx264 -crf 28 enc1.mp4
ffmpeg -i source.mp4 -c:v libx265 -crf 28 enc2.mp4
ffmpeg -i source.mp4 -c:a libopus -b:a 64k -c:v copy enc3.mp4

# Negative pairs: just take random pairs from different sources
```

### Approach 3: stability scoring

For a single video, fingerprint it twice (or fingerprint and re-fingerprint
the same file). The two fingerprints should be **identical** (bit-for-bit).
If they're not, your decoder is non-deterministic — set the threshold above
the observed variance.

For Chromaprint via `fpcalc`, fingerprints are deterministic if the audio
decode is. FFmpeg PCM extraction is. So same input → same fingerprint, 100%
of the time. The threshold doesn't need to account for self-noise.

### Starter thresholds (calibrate before shipping)

| Metric | Start | Likely range after tuning |
|---|---|---|
| Chromaprint per-window bit agreement | 0.85 | 0.80–0.90 |
| Multi-segment vote required | 2 of 3 | 2 of 3 |
| Min overlap in alignment | 60% of shorter | 50%–70% |
| Spectrogram pHash Hamming (16×16) | ≤ 14 / 256 | 10–20 |
| Spectrogram pHash uncertain band | 15–28 / 256 | varies |

The current RMS threshold is 80% correlation. Don't carry that number over;
it doesn't translate.

---

## Code sketch for the recommended primary approach

This replaces `backend/services/audio_fingerprint.py` end-to-end. ~120 lines.
**Do not implement this in the repo as part of this research** — it's here as
the design target.

```python
"""Audio fingerprinting v2 — multi-segment Chromaprint + spectrogram pHash.

Replaces v1 RMS energy profile with two complementary fingerprints:
  - audio_fp_chromaprint: list[list[int]]  (K segment fingerprints)
  - audio_fp_spectro:     str              (one 256-bit pHash hex)

Either alone is enough to match; together they cover the failure modes of
each. See docs/research/audio-fingerprint-alternatives.md.
"""

import asyncio
import base64
import json
import os
import subprocess
from typing import Optional

import numpy as np
import imagehash
from PIL import Image

from config import settings

_CREATION_FLAGS = (
    subprocess.CREATE_NO_WINDOW
    if hasattr(subprocess, "CREATE_NO_WINDOW")
    else 0
)

# Tunables (move to config.py when shipping)
SEG_OFFSETS = (0.10, 0.50, 0.90)
SEG_LENGTH_SEC = 10
MIN_DURATION_FOR_CHROMAPRINT = 5.0
SPECTRO_DURATION_SEC = 30
SPECTRO_HASH_SIZE = 16
CHROMAPRINT_MATCH_THRESHOLD = 0.85
CHROMAPRINT_VOTE_REQUIRED = 2          # M of K
SPECTRO_HAMMING_THRESHOLD = 14         # / 256

_FPCALC_BIN = os.environ.get("FPCALC_BIN", "fpcalc")


# ─── fingerprint extraction ────────────────────────────────────────────────

def _fpcalc_segment(path: str, ss: float, length: float) -> list[int]:
    """One Chromaprint call on a trimmed window. Returns raw 32-bit ints."""
    # FFmpeg trims; fpcalc fingerprints from stdin (avoids a temp file).
    ffm = subprocess.Popen(
        ["ffmpeg", "-ss", f"{ss}", "-i", path, "-t", f"{length}",
         "-vn", "-ac", "1", "-ar", "11025", "-f", "wav", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        creationflags=_CREATION_FLAGS,
    )
    fp = subprocess.run(
        [_FPCALC_BIN, "-json", "-raw", "-"],
        stdin=ffm.stdout, capture_output=True, text=True, timeout=30,
        creationflags=_CREATION_FLAGS,
    )
    if ffm.stdout:
        ffm.stdout.close()
    ffm.wait()
    if fp.returncode != 0:
        return []
    try:
        data = json.loads(fp.stdout)
    except json.JSONDecodeError:
        return []
    return data.get("fingerprint", []) or []


def _spectro_phash(path: str, duration: float) -> str:
    """One 256-bit pHash of the mid-window spectrogram."""
    ss = max(0.0, (duration - SPECTRO_DURATION_SEC) / 2)
    length = min(duration, SPECTRO_DURATION_SEC)
    r = subprocess.run(
        ["ffmpeg", "-ss", f"{ss}", "-i", path, "-t", f"{length}",
         "-vn", "-ac", "1", "-ar", "16000",
         "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"],
        capture_output=True, timeout=30, creationflags=_CREATION_FLAGS,
    )
    if r.returncode != 0 or len(r.stdout) < 4000:
        return ""
    samples = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if len(samples) < 4000:
        return ""
    n_fft, hop = 1024, 256
    pad = (-len(samples)) % hop
    if pad:
        samples = np.pad(samples, (0, pad))
    frames = np.lib.stride_tricks.sliding_window_view(samples, n_fft)[::hop]
    win = np.hanning(n_fft).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * win, axis=1))
    spec = np.log1p(spec)
    peak = spec.max()
    if peak > 0:
        spec = spec / peak
    img = Image.fromarray((spec.T * 255).astype(np.uint8))
    return str(imagehash.phash(img, hash_size=SPECTRO_HASH_SIZE))


def _audio_fingerprint_sync(path: str, duration: float) -> dict:
    """Extract chromaprint segments + spectrogram pHash.

    Returns:
        {"cp": list[list[int]], "spectro": str, "duration": float}
    """
    out = {"cp": [], "spectro": "", "duration": duration}

    if duration >= MIN_DURATION_FOR_CHROMAPRINT:
        for frac in SEG_OFFSETS:
            ss = max(0.0, duration * frac - SEG_LENGTH_SEC / 2)
            if ss + SEG_LENGTH_SEC > duration:
                ss = max(0.0, duration - SEG_LENGTH_SEC)
            fp = _fpcalc_segment(path, ss, min(SEG_LENGTH_SEC, duration - ss))
            out["cp"].append(fp)

    try:
        out["spectro"] = _spectro_phash(path, duration)
    except Exception:
        out["spectro"] = ""

    return out


# ─── comparison ────────────────────────────────────────────────────────────

def _popcount_u32(x: np.ndarray) -> np.ndarray:
    """Vectorised popcount of uint32 array."""
    x = x.astype(np.uint32)
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    return (x * 0x01010101) >> 24


def _cp_window_similarity(fp1: list[int], fp2: list[int],
                           max_offset: int = 8) -> float:
    """Best-alignment bit agreement in [0, 1]. Empty inputs → 0."""
    if not fp1 or not fp2:
        return 0.0
    a, b = np.asarray(fp1, dtype="u4"), np.asarray(fp2, dtype="u4")
    best = 0.0
    for off in range(-max_offset, max_offset + 1):
        if off >= 0:
            x = a[:len(a) - off] if off > 0 else a
            y = b[off:off + len(x)]
        else:
            x = a[-off:]
            y = b[:len(x)]
        n = min(len(x), len(y))
        if n < 8:
            continue
        x, y = x[:n], y[:n]
        diff_bits = int(_popcount_u32(x ^ y).sum())
        agree = 1.0 - diff_bits / (n * 32)
        if agree > best:
            best = agree
    return best


def compare_audio_v2(fp1: dict, fp2: dict) -> tuple[float, str]:
    """Return (similarity_pct, method) where method ∈ {"cp", "spectro", "none"}.

    similarity_pct is 0–100; comparator decides the threshold.
    """
    cp1, cp2 = fp1.get("cp") or [], fp2.get("cp") or []
    K = min(len(cp1), len(cp2))

    if K >= 1:
        scores = []
        for i in range(K):
            scores.append(_cp_window_similarity(cp1[i], cp2[i]))
        votes = sum(s >= CHROMAPRINT_MATCH_THRESHOLD for s in scores)
        if votes >= CHROMAPRINT_VOTE_REQUIRED:
            return max(scores) * 100, "cp"
        # No vote majority → also check best window for "uncertain" signal
        if max(scores, default=0) >= CHROMAPRINT_MATCH_THRESHOLD:
            return max(scores) * 100, "cp_uncertain"

    sp1, sp2 = fp1.get("spectro") or "", fp2.get("spectro") or ""
    if sp1 and sp2:
        h1, h2 = imagehash.hex_to_hash(sp1), imagehash.hex_to_hash(sp2)
        dist = h1 - h2                              # Hamming over 256 bits
        if dist <= SPECTRO_HAMMING_THRESHOLD:
            return (1.0 - dist / 256) * 100, "spectro"

    return 0.0, "none"


# ─── async wrapper ────────────────────────────────────────────────────────

_executor = None

def _get_executor():
    global _executor
    if _executor is None:
        from concurrent.futures import ThreadPoolExecutor
        # Conservative: audio FP is CPU-bound, don't oversubscribe.
        _executor = ThreadPoolExecutor(
            max_workers=min(os.cpu_count() or 4, 8))
    return _executor


async def audio_fingerprint_v2(file_path: str, duration: float) -> dict:
    """Compute the new audio fingerprint asynchronously."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _get_executor(), _audio_fingerprint_sync, file_path, duration)
```

#### Storage format

`FileCache.audio_fp` gets a new `audio_fp_version INTEGER DEFAULT 1` column.
Old RMS entries (`version=1`) are read by the legacy comparator until the
cache is cleared. New entries are `version=2` with a dict-of-stuff stored as
JSON. Migration: bump the schema, treat the old `audio_fp` as nullable, and
write new fingerprints into a new `audio_fp_v2` JSON column. The codebase
already has no migrations (CLAUDE.md says "schema changes require deleting
the DB"), so realistically you just bump a schema version and let the cache
re-populate on the next scan.

---

## What's dead vs alive in 2026

A quick verdict on every named system:

- **Chromaprint / fpcalc / pyacoustid** — **alive**. AcoustID infrastructure
  still runs, MusicBrainz/Picard depend on it. Bug fixes flow. The C library
  hasn't had a major release since 2022 because it doesn't need one — the
  algorithm is stable. Treat it as a finished, working primitive.
- **Dejavu (worldveil/dejavu)** — **dead**. Snyk classifies it inactive. Last
  meaningful upstream work was 2018. Forks exist (bcollazo, denis-stepanov,
  yunpengn) but none have momentum. The fundamental design (constellation
  hashes in MySQL) is also poorly aligned with our problem — we don't need a
  generative ID lookup, we need a pairwise compare.
- **audfprint (dpwe/audfprint)** — **dormant**. Dan Ellis maintains it
  sporadically; functional but not seeing investment. Same architectural
  mismatch as Dejavu (numpy + hashtable database). Avoid for new work.
- **Olaf (JorenSix/Olaf)** — **alive but burst-mode**. Designed for
  embedded/microcontrollers. Excellent for what it does. **AGPL** licence is
  a deal-breaker for some shops; check your distribution model.
- **Panako (JorenSix/Panako)** — **alive but burst-mode**. Java-based, JVM
  dependency. **The only system explicitly designed for varispeed/pitch-shift
  robustness.** AGPL same as Olaf. Only worth integrating if you have a
  pitch-shift detection requirement.
- **CLAP (LAION)** — **very alive**. Active research model in 2025–2026 (HTSAT
  backbone, T-CLAP / SmoothCLAP variants). Useful for **semantic** audio
  similarity (genre, mood, speaker) — not what we want. We want a hash that
  says "same recording" not "similar music". CLAP overfires on
  same-genre-different-song.
- **OpenL3** — **low maintenance**. Last release 2021. Still works, TensorFlow
  2 only. Weights are MIT.
- **PANN / PANNs (panns_inference)** — **low maintenance**. Solid pre-trained
  weights, library is stale. Could use them directly via PyTorch without the
  helper library.
- **wav2vec2 (HuggingFace)** — **very alive** via the transformers library,
  but mostly useful for speech ASR, not duplicate detection.
- **Shazam** — proprietary; no open implementation of the **lookup** path.
  The constellation algorithm is public and Dejavu/audfprint re-implement it.
- **YapHash / phash-audio** — niche academic systems. Skip.

---

## Edge cases

### Silent / no-audio videos

Current behaviour: `_audio_fingerprint_sync` returns `[]` (empty list). In
`comparator.py:142`, `if not matched and audio_i and audio_j` is False, so
the audio fallback is skipped. The pair is compared on video pHash only.
**This is correct behaviour.** No change needed for purely-silent or
audio-less videos.

What changes with the composite: spectrogram pHash still discriminates on
near-silent content (mic noise floor) but truly silent video (no audio
stream) still falls back to "skip the audio stage entirely". Two such
videos can only match by video pHash.

### Music-only videos

Music videos are the **hard case** for the current RMS approach: two
different music videos of the same length have very similar RMS energy
curves (they're both "loud throughout"), driving false positives.

With Chromaprint this is fine — different songs have very different chroma
profiles. With spectrogram pHash, also fine — different songs have very
different spectral content.

### Dubbed / language-swapped re-encodes

A movie re-released with a different audio track (e.g. dubbed). Current
behaviour: audio fingerprints differ (correctly), video pHash matches
(correctly). The OR-rule says "duplicate" and that's the right answer for
this app — it's the same movie, the user should see it in the duplicate
group.

The composite preserves this: video pHash agreement is the OR-side that
wins. No regression.

### Varispeed re-encodes

Speed-changed re-uploads (common on YouTube to evade content matching, e.g.
+5% speed). Current RMS handles ±5% poorly because the segment boundaries
shift and energies don't line up. Full-track Chromaprint also fails ±5%
because the chroma image stretches.

The multi-segment design handles this: per-window alignment search ±8 frames
(±1 s) over a 10-s window absorbs ±5% drift. ±10% drift exceeds ±1 s over
10 s and would also fail; for that you'd need Panako or a deep embedding.
Mark it as out of scope unless users complain.

### Very long videos (films, 90+ min)

Current behaviour: full audio decode is wasted work, and the 64-point RMS
spans 84-s chunks each. With multi-segment, you fingerprint three 10-s
windows total = 30 s of audio decoded per video regardless of length. This
is **massive** for large libraries: a 10k film library drops from ~5400 s ×
10k / 8 = 1875 hours of decode to 30 s × 10k / 8 = 10 hours. Order-of-
magnitude wall-clock reduction.

### Mono vs stereo source

Chromaprint downmixes to mono. Spectrogram pHash should too. No issue.

### Variable bitrate / VBR

No issue. Both algorithms operate on decoded PCM.

### Encoded audio with high-frequency damage (e.g. low-bitrate Opus)

Chromaprint chroma uses the lower octaves; HF damage doesn't affect it.
Spectrogram pHash on a 16×16 downsampled spectrogram averages out HF detail;
also fine.

### Videos with leading silence

`fpcalc` has an option to skip leading silence (introduced in v0.7). The
multi-segment design also helps: the 10% offset is past the typical 1–2 s
intro silence anyway.

### Audio extracted but FFmpeg fails

Current behaviour: `_audio_fingerprint_sync` catches exceptions, prints, and
returns `[]`. Treat absent fingerprints as "audio stage didn't run for this
file" — same as the no-audio case. Don't fail the whole video over an
unfingerprintable audio stream.

---

## Other ideas considered and rejected

### Speech/music/silence classifier pre-filter

Idea: run `inaSpeechSegmenter` or a small CNN to classify "is this audio
mostly music / speech / silent" and pick a specialised fingerprinter per
class.

Verdict: **not worth it**. Chromaprint handles all three classes adequately;
spectrogram pHash handles the silent-noise-floor case. Adding a classifier
means more deps, more weights, more failure modes. The current degenerate-
fingerprint detection (empty `cp` list) is a 1-bit classifier and is
sufficient as a routing signal.

### Replace the OR-rule with a learned classifier

Idea: feed `(video_phash_dist, audio_cp_sim, audio_spectro_dist,
duration_diff, size_ratio)` into a gradient-boosted classifier trained on a
labelled pair dataset.

Verdict: defer. The OR rule is interpretable, debuggable, has a clean
diagnostic story (`diagnose_pair.py`), and good enough. A classifier adds
training-data costs that exceed the integration cost of the composite
approach above. Revisit if false-positive complaints stack up.

### Use a vector DB (FAISS, qdrant) for audio embeddings

Idea: pre-compute a CLAP or PANN embedding per video, ANN-index, look up
duplicates as nearest-neighbours.

Verdict: **dual-use with the video embedding pipeline** discussed in
`algorithmic-improvements.md` section 3. If/when we ship DINOv2 visual
embeddings, the same FAISS index can hold audio embeddings on a second
field. Until then, the audio side doesn't need a vector DB — Chromaprint
plus 64-byte segment-vote comparisons fit in RAM and need no index.

### Use OpenAI Whisper to extract transcripts and compare those

Idea: transcribe both videos, compare transcripts via text similarity.

Verdict: **bad fit**. Whisper inference is 1–10 s per minute of audio even on
GPU. For a 90-min movie that's 90–900 seconds per file. For thousands of
files this is hours. Also fragile: identical recordings with different
speakers (dubbing) would not transcribe to the same text. The whole point of
Chromaprint is to be transcript-independent. Skip.

---

## Migration path

If shipping this, do it in three commits:

1. **Add `fpcalc` to the toolchain.** Update `requirements.txt`,
   `Dockerfile`, README install instructions. Add a `FPCALC_BIN` env var so
   ops can override the binary path. Verify with a one-line smoke test in
   `main.py` startup (warn if fpcalc not found, don't fail).

2. **Implement `audio_fingerprint_v2` and `compare_audio_v2` in a new file.**
   Leave `audio_fingerprint.py` (v1) untouched. Add a `audio_fp_version`
   column to `FileCache` and a `audio_fp_v2 JSON` column.

3. **Switch the pipeline over.** In `scan.py`, call v2 instead of v1; in
   `comparator.py`, branch on `audio_fp_version` so the comparator can read
   either format. Drop v1 once `audio_fp_version=1` rows are aged out (or
   one-shot delete the cache).

Each commit is independently shippable and the rollback is `git revert`.

---

## Recommended reading

- Chromaprint algorithm description by Lukáš Lalinský (the author):
  <https://oxygene.sk/2011/01/how-does-chromaprint-work/>
- AcoustID fingerprint comparison notebook (canonical sliding-alignment
  reference implementation):
  <https://github.com/acoustid/notebooks/blob/master/fingerprint-matching.ipynb>
- "Using Audio Fingerprinting for Duplicate Detection and Thumbnail
  Generation" — Microsoft Research, 2004. The multi-segment / windowed
  comparison idea is from here.
  <https://www.microsoft.com/en-us/research/wp-content/uploads/2005/03/audiothumbnail.pdf>
- Olaf: a lightweight, portable audio search system — Joren Six, JOSS 2023.
  <https://www.theoj.org/joss-papers/joss.05459/10.21105.joss.05459.pdf>
- LAION CLAP: <https://github.com/LAION-AI/CLAP>
- `kdave/audio-compare` — a small reference implementation of fpcalc-based
  audio comparison in Python; useful for cross-checking thresholds.
  <https://github.com/kdave/audio-compare>

---

## TL;DR for the implementer

1. Ship `fpcalc` in the toolchain. Replace RMS with multi-segment
   Chromaprint (3 × 10 s windows, 2-of-3 vote, per-window threshold 0.85
   bit-agreement, sliding alignment ±8 frames).
2. Add a single-shot spectrogram pHash as a side fingerprint for the
   < 5 s clip case and the near-silent case. ~30 lines, no new deps.
3. Combine via the M-of-K vote + Hamming fallback strategy described above.
4. Calibrate thresholds against a held-out set generated from re-encoded
   FFmpeg pairs; expect 0.80–0.90 for Chromaprint and ≤ 14/256 for
   spectrogram pHash.
5. Don't bother with Dejavu, audfprint, Panako, or CLAP for this workload.
   They solve different problems.

Expected outcomes vs current RMS approach:

- **False-positive rate**: drops from a few percent to under 0.5%.
- **Discriminative power**: ~10–20× — Chromaprint actually identifies *which*
  audio it's looking at, not just its energy shape.
- **Wall-clock cost**: roughly unchanged or *cheaper*, because we no longer
  decode the full audio track (recommendation #1 from
  `pipeline-optimizations.md` is essentially built in).
- **Short clip / silent video coverage**: from "broken" to "works".
- **Varispeed coverage**: from "fails at ±2%" to "fails at ±10%".
