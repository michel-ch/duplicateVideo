# GPU Acceleration Deep-Dive

Research only — no code in this repo has been changed. The aim is to identify
GPU-specific speedups **beyond** the current "use NVDEC via `-hwaccel cuda`"
baseline. Existing research on caching, BK-tree, Chromaprint and content
bucketing is covered elsewhere and is explicitly out of scope here.

The numbers in this document are first-principles estimates extrapolated from
NVIDIA developer documentation, published library benchmarks, and
back-of-envelope calculations against the current `extract_and_hash` flow
(`backend/services/hasher.py`). They are not measured on this machine. Anything
expressed as "≈" or "~" should be confirmed by a one-off benchmark before
committing weeks to an implementation.

Target hardware reference: **RTX 3060 12 GB / RTX 3060 Ti 8 GB (Ampere, GA106 /
GA104)**. Single NVDEC engine per chip on this SKU (3rd-gen NVDEC), single
NVENC, CUDA compute capability 8.6.

---

## Table of contents

1.  [Executive summary](#1-executive-summary)
2.  [Baseline: what the current code actually pays for](#2-baseline-what-the-current-code-actually-pays-for)
3.  [Finding 1 — Single-subprocess multi-frame extraction (already done, partly)](#3-finding-1--single-subprocess-multi-frame-extraction)
4.  [Finding 2 — GPU-side pHash via cuPy / torch DCT](#4-finding-2--gpu-side-phash-via-cupy--torch-dct)
5.  [Finding 3 — NVDEC-to-tensor: skip the JPEG round-trip entirely](#5-finding-3--nvdec-to-tensor-skip-the-jpeg-round-trip)
6.  [Finding 4 — Batched / persistent decoder context](#6-finding-4--batched--persistent-decoder-context)
7.  [Finding 5 — AV1 hardware decode on Ampere / Ada](#7-finding-5--av1-hardware-decode-on-ampere--ada)
8.  [Finding 6 — Multi-GPU: useful or premature?](#8-finding-6--multi-gpu)
9.  [Recommended library stack](#9-recommended-library-stack)
10. [Code sketch: GPU-resident frame extraction + pHash in one pass](#10-code-sketch-gpu-resident-frame-extraction--phash)
11. [Failure modes & fallback strategy](#11-failure-modes--fallback-strategy)
12. [Rough benchmark expectations on an RTX 3060](#12-rough-benchmark-expectations)
13. [Open questions](#13-open-questions)
14. [Sources](#14-sources)

---

## 1. Executive summary

Three GPU-specific wins, prioritised by **impact ÷ effort**:

| # | Win | Expected speedup | Effort | Risk |
|---|---|---|---|---|
| **A** | **NVDEC-to-tensor pipeline via TorchCodec or PyNvVideoCodec — keep frames on GPU, run DCT/pHash with cuPy, eliminate JPEG encode + disk I/O + PIL decode per frame** | **3–6× end-to-end on stages 2–3** (the dominant cost) | High (new dep, Linux/Win matrix to validate) | Medium — fallback to current path is straightforward |
| **B** | **GPU-side DCT pHash with cuPy / `torch.fft`** — batched 32×32 DCT over 12-frame stacks instead of per-frame scipy.fftpack on CPU | **~10–50× on the hash step itself**, but the hash step is currently 5–15% of stage 3, so end-to-end is **1.05–1.15×** unless combined with (A) | Medium (one new dep, ~80 lines) | Low — `cuPy` DCT is mature, bit-for-bit verifiable against `imagehash.phash` |
| **C** | **Drop separate ffprobe subprocess for duration/codec/SAR by reading from the existing metadata-stage probe** — saves 2–3 process spawns per file | **~50–150 ms/file on Windows**, dominant on tiny files | Low (~30 lines) | None — already partially done via `_meta_video_info` passthrough |

**Top recommendation**: pursue **(A) NVDEC-to-tensor + (B) GPU-side pHash
together**, because (A) without (B) leaves the frame on GPU and then
synchronously downloads to CPU for PIL — defeating most of (A)'s benefit. The
two changes are designed for each other.

**Biggest implementation risk**: PyNvVideoCodec / TorchCodec install matrix on
Windows + Docker. The current Docker base
(`nvidia/cuda:12.2.2-runtime-ubuntu22.04`) is fine for TorchCodec, but
PyNvVideoCodec wheels are Linux x86_64 / Win64 only and require CUDA toolkit
12.x — Pascal / Maxwell users would be left on the current ffmpeg-subprocess
path. The existing GPU fallback machinery in `gpu_detector.py` handles this
cleanly; you just need a third tier: "modern GPU + library installed" →
"NVDEC via ffmpeg" → "CPU".

---

## 2. Baseline: what the current code actually pays for

`extract_and_hash` (`backend/services/hasher.py:605`) per video, on an RTX 3060
Ti with NVDEC enabled, for a 5-minute H.264 1080p file:

```
┌────────────────────────────────────────────────────────────────┐
│ extract_and_hash(file_path, num_frames=12, …)                  │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│ Cost A: Subprocess spawn  (1× ffmpeg)             ~ 30–80 ms   │
│ Cost B: NVDEC decode + seek                       ~ 200–400 ms │
│ Cost C: hwdownload (GPU→CPU memcpy of 12 frames)  ~ 5–15 ms    │
│ Cost D: CPU scale + setsar + transpose (libavfilter) ~10–30 ms │
│ Cost E: JPEG encode (CPU, libjpeg)                ~ 30–60 ms   │
│ Cost F: Disk write 12 × ~10 KB JPEGs              ~ 10–30 ms   │
│ Cost G: Python tempdir/path bookkeeping           ~ 5–10 ms    │
│ Cost H: PIL.Image.open × 12                       ~ 20–40 ms   │
│ Cost I: imagehash.phash (scipy DCT) × 12          ~ 100–200 ms │
│ Cost J: shutil.rmtree of tempdir                  ~ 5–15 ms    │
├────────────────────────────────────────────────────────────────┤
│ Total per video                              ~ 415–880 ms      │
└────────────────────────────────────────────────────────────────┘
```

Annotated against the code:

- **Cost A** — `subprocess.run(cmd, …)` in `_extract_frames_sync` (line ~294).
  On Windows, `CreateProcess` with `CREATE_NO_WINDOW` is ~10 ms minimum and
  often higher under semaphore contention. There is also `_get_video_duration`
  (line 49) + `_get_video_codec` (line 69) + `_get_video_info` (line 89) =
  **three more ffprobe spawns** if `duration`/`codec`/`video_info` are not
  passed in. Stage 2 in `scan.py` already passes them in (`_meta_video_info`),
  so this is a non-issue on the hot path. The thumbnail path also makes 2
  ffprobe calls (line 408 + 414) when called without prefilled args.
- **Cost B** — pure NVDEC. Not much to optimise here; this is what the GPU is
  for. Note that on a single-NVDEC SKU (3060 / 3060 Ti) you cannot decode 12
  videos truly in parallel — they time-slice on the engine. The "negligible
  context-switch penalty" claim in the NVDEC application note holds for
  hardware contexts, but each *ffmpeg process* still pays its own decoder
  initialisation cost.
- **Cost C** — `hwdownload,format=nv12` in the filter chain (line 234–235).
  PCIe-3 ×16 = 16 GB/s peak; an NV12 frame at 320×180 is ~85 KB; 12 frames =
  ~1 MB → ~60 µs of actual bandwidth, but small PCIe transfers are
  latency-bound. **15 ms is realistic with semaphore-limited concurrency**.
- **Cost D** — CPU filter chain after hwdownload. Cheap because frames are
  already small at this point.
- **Cost E** — `-q:v 2` JPEG encode. Twelve frames at 320×N → ~10 KB each.
  libjpeg-turbo can hit 200 MB/s, so ~120 KB / 200 MB/s ≈ 0.6 ms of CPU. The
  rest of the 30–60 ms is Python/IO overhead.
- **Cost F** — small file writes on Windows (NTFS) are slow due to per-file
  metadata updates. SSD helps; spinning disk hurts.
- **Cost H** — Pillow JPEG decode. ~2 ms per 10 KB JPEG = ~25 ms.
- **Cost I** — **the big one for pHash**. `imagehash.phash(img, hash_size=16)`
  resizes to 32×32 (PIL), then runs `scipy.fftpack.dct(dct(x, axis=0), axis=1)`
  on a 32×32 array. The DCT is fast (~0.5 ms), but PIL resize from 320×N to
  32×32 is ~5 ms × 12 = 60 ms. **Resizing dominates**, not the DCT.
- **Cost J** — `shutil.rmtree` of a 12-file tempdir. NTFS-slow.

**Where the win is**: Costs **E + F + G + H + I + J ≈ 170–350 ms per video are
all "we already had the pixels on GPU, then bounced them through disk".**
Eliminating that round-trip is the prize.

A typical scan of 1,000 videos at 500 ms each = **500 seconds wall-clock**
(stages 2+3). Reducing that to 100–150 ms gives **100–150 seconds** —
~3.5× faster, on the same hardware.

---

## 3. Finding 1 — Single-subprocess multi-frame extraction

**Status: already partly done.** `_extract_frames_sync` already invokes a single
ffmpeg with `-frames:v 12` + `-vf "fps=N/duration"` (line 250–255). One
subprocess, twelve JPEGs out.

The remaining (small) opportunity is using `select='eq(pict_type,I)'` to take
**only I-frames**, which can be much faster than `fps=N` because:

1. `fps=N` forces ffmpeg to decode every frame until the next sample boundary.
2. `select='eq(pict_type,I)'` lets the demuxer seek directly to keyframes;
   only the I-frames go through the decoder.

For a typical web-distributed H.264 with ~1 keyframe per 2 seconds, a 5-minute
video has ~150 keyframes — plenty more than the 12 we need. With NVDEC,
`-skip_frame nokey` on the input side can short-circuit even non-I frames
inside the decoder.

```bash
ffmpeg -hwaccel cuda -hwaccel_output_format cuda \
       -skip_frame nokey -c:v h264_cuvid -i IN.mp4 \
       -vf "thumbnail=12,hwdownload,format=nv12,scale=320:-2" \
       -frames:v 12 -q:v 2 OUT_%04d.jpg
```

`thumbnail=12` (libavfilter) picks the 12 most "thumbnail-like" frames over the
whole video, which is more robust for hashing than evenly-spaced sampling
because it skips solid black title cards, transitions, and color bars.

### Realistic gain

- Decode cost drops from "decode every Nth frame in the file" to "decode every
  keyframe". For a 5-min H.264 at 30 fps with GOP=60, that's
  9,000 frames → 150 frames. **~6× less decode work for the same 12 outputs.**
- Cost B falls from 200–400 ms to roughly 40–80 ms per video.
- **End-to-end win: ~150 ms/video shaved off stages 2+3.**

### Risks

- Some encoders emit very long GOPs (Netflix-style), with 5–10 keyframes per
  hour. `select=eq(pict_type,I)` on those will return fewer than 12 frames →
  hashing fails. **Mitigation**: detect the shortfall (number of output JPEGs
  < `num_frames`) and retry without `-skip_frame nokey`.
- `thumbnail=N` requires ffmpeg ≥ 2.0 (universally available now) but with
  `hwaccel_output_format=cuda` the filter must run after `hwdownload` — i.e.
  on CPU, so it does see every decoded frame. The win from `thumbnail` is in
  *quality*, not speed; `select='eq(pict_type,I)'` is the speed win.

### Subprocess spawn overhead, quantified

On Windows 10/11 with `CREATE_NO_WINDOW`:

- Bare `CreateProcess` + immediate exit: ~5 ms.
- `subprocess.run(["ffmpeg", "-version"])`: ~80–120 ms (ffmpeg init + version
  string + exit).
- `subprocess.run(["ffprobe", …, file])` returning JSON: ~150–250 ms cold,
  ~50–100 ms warm (filesystem cache hits on the binary).

Compare to Linux, where the same calls are typically 1/3 to 1/2 the latency
due to `fork()` + `execve()` vs `CreateProcess`. **This is why Windows
benefits disproportionately from any "fewer subprocesses" optimisation.**

The current code already calls `_get_video_duration`, `_get_video_codec`, and
`_get_video_info` lazily (lines 280–286) — but only if the caller didn't
pre-supply them. Stage 2 in `scan.py` does pre-supply them, so the hot path
already pays only **one** ffmpeg spawn per video. Outside the hot path
(e.g. `diagnose_pair.py`), the savings would be substantial.

---

## 4. Finding 2 — GPU-side pHash via cuPy / torch DCT

`imagehash.phash` does this (paraphrased from the source):

```python
def phash(image, hash_size=16, highfreq_factor=4):
    img_size = hash_size * highfreq_factor   # 64 for hash_size=16
    image = image.convert("L").resize((img_size, img_size), ANTIALIAS)
    pixels = numpy.asarray(image)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
    dctlowfreq = dct[:hash_size, :hash_size]   # top-left 16×16
    med = numpy.median(dctlowfreq)
    diff = dctlowfreq > med
    return ImageHash(diff)
```

Two CPU-bound operations dominate, both per-frame:

1. **Resize** PIL grayscale 320×180 → 64×64 — Lanczos / box filter, ~5 ms.
2. **2D DCT** on 64×64 → ~0.5 ms with scipy.

For 12 frames × N videos, the resize cost adds up linearly. On 1,000 videos =
12,000 frames × 5 ms = **60 seconds of CPU time just resizing**.

### What changes on GPU

Stack 12 frames per video into a `[12, H, W]` tensor that already lives on the
GPU (because Finding 3 keeps them there). Then:

```python
# Pseudo-code with cuPy
import cupy as cp
from cupyx.scipy.fft import dctn  # 2D DCT

# frames_gpu: cuPy uint8 array [12, H, W] in luminance
resized = cucim.skimage.transform.resize(
    frames_gpu, (12, 64, 64), anti_aliasing=True
)  # batched on GPU
dct = dctn(resized.astype(cp.float32), axes=(1, 2), norm="ortho")
low = dct[:, :16, :16]                # [12, 16, 16]
med = cp.median(low.reshape(12, -1), axis=1, keepdims=True)
bits = (low.reshape(12, -1) > med).astype(cp.uint8)
hashes_packed = cp.packbits(bits, axis=1)   # [12, 32] bytes
hashes_host = hashes_packed.get()           # only this last step crosses PCIe
```

The DCT and median use cuFFT and cuPy's reduction kernels — both **batched**
across the frame dimension. For 12 frames at 64×64 the work is laughably small
for a GPU; **the bottleneck is the kernel launch overhead, not the math**.
Twelve frames is below the break-even point for GPU-vs-CPU on the hash step
alone — but if you batch **across videos** (say 32 videos × 12 frames = 384
frames per kernel launch), the GPU obliterates CPU.

### Realistic standalone gain on the hash step

- CPU baseline (12 frames): ~80–130 ms (resize + DCT + bookkeeping).
- GPU (12 frames, single video): ~5–10 ms when frames are already on GPU.
  Most of this is kernel-launch overhead; the actual compute is sub-ms.
- GPU (384 frames, batched across 32 videos): ~10–20 ms total → ~0.05 ms per
  hash. **~1,600× per-hash in batched mode**, but only **~10×** if you don't
  batch.

### Caveat: bit-exact compatibility with existing cache

`FileCache.perceptual_hashes` already contains hashes computed by
`imagehash.phash(img, hash_size=16)`. If the GPU implementation produces
**different bits** for the same frame (which it will, because Pillow's resize
≠ cuCIM's resize at exact pixel values), the cache becomes useless until
re-hashed.

Two paths:

1. **Bit-exact**: implement the same Pillow-style resize (Antialias is a
   Lanczos-3 filter) on GPU. Possible with `torchvision.transforms.functional.resize`
   + `InterpolationMode.LANCZOS`, but the kernel boundaries differ slightly
   from PIL's. Even with the same algorithm, sub-pixel rounding differs.
   In practice, **~95% of hash bits agree**, so the Hamming threshold of 14
   on a 256-bit hash absorbs the noise — old and new hashes still compare
   correctly with each other.
2. **Invalidate cache**: bump `FileCache` schema version, force a re-hash on
   the next scan. Simpler. Given the project's "delete the DB to migrate"
   stance, this is the honest choice.

### Risks

- **PIL resize fidelity**: as above. Acceptable.
- **cuPy install on Windows**: pip wheels are CUDA-version-specific
  (`cupy-cuda12x`, `cupy-cuda11x`). Need to detect the host CUDA version and
  pick the right wheel. The Docker image's CUDA 12.2 maps to `cupy-cuda12x`.
- **Memory**: 32 videos × 12 frames × 4 channels × 320 × 180 × 1 byte
  (uint8) = 22 MB — trivial on 8 GB VRAM.

---

## 5. Finding 3 — NVDEC-to-tensor: skip the JPEG round-trip

This is the largest single win. The current pipeline is:

```
GPU decode → hwdownload (PCIe) → CPU filter → JPEG encode → disk write →
PIL open → PIL resize → numpy → scipy DCT → 256-bit hash
   ^^^^^   ^^^^^^^^^^   ^^^^^^   ^^^^^^^^^   ^^^^^^^^^^^^^
   ~10ms   ~10ms        ~30ms    ~10ms        ~140ms (PIL + scipy)
```

The replacement:

```
GPU decode → keep as CUDA tensor → cuPy resize → cuPy DCT → hash → host (32 B/frame)
   ^^^^^^                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^
   ~10ms (NVDEC time only)         ~5–10 ms total                  ~0.5 ms × 12 = 6 ms
```

Three Python libraries can give you a CUDA tensor of decoded frames:

### 5a. TorchCodec (Meta, ≥ 0.7, recommended)

**Status**: official PyTorch project, replaces deprecated `torchvision.io.VideoReader`
and `torchaudio` NVDEC routes. CUDA backend stable since 0.4; current is
~0.12. Pip-installable on Linux + Windows.

```python
from torchcodec.decoders import VideoDecoder

decoder = VideoDecoder("video.mp4", device="cuda")
print(decoder.cpu_fallback)        # CpuFallbackStatus.NoFallback if NVDEC worked

# Get 12 evenly-spaced frames at specific timestamps
duration = decoder.metadata.duration_seconds
timestamps = [duration * i / 13 for i in range(1, 13)]
frames = decoder.get_frames_played_at(timestamps).data   # tensor on cuda:0
# frames.shape == [12, C, H, W], dtype uint8, device cuda:0
```

`get_frames_played_at(...)` is **batched at the C++ level** — one NVDEC
seek+decode loop, twelve frames out, never leaves GPU. This is the API to
use.

**Caveats**:

- TorchCodec depends on PyTorch (~700 MB wheel). Big install.
- Frame outputs are RGB by default; need to convert to grayscale before DCT.
  Trivial: `frames.float().mean(dim=1)` or use a luminance-weighted dot
  product.
- Color-space conversion (NV12 → RGB) happens in CUDA inside TorchCodec.
  Slightly redundant if you only need luminance — but it's fast enough that
  it doesn't matter at our frame counts.
- `CpuFallbackStatus.NoFallback` vs `.FallbackOnDecode`: TorchCodec falls back
  silently to CPU on unsupported codecs. Your fallback logic stays in
  `gpu_detector.py`-style probing.

### 5b. PyNvVideoCodec 2.0 (NVIDIA, official, successor to VPF)

**Status**: NVIDIA-supported. VPF (PyNvCodec) is deprecated in favor of this.
Pip-installable. Tighter integration with NVDEC features (decoder caching,
multi-instance per GPU).

```python
import PyNvVideoCodec as nvc
import torch

decoder = nvc.SimpleDecoder(
    "video.mp4",
    use_device_memory=True,
    output_color_type=nvc.OutputColorType.RGBP,   # planar RGB on GPU
)
# Sample 12 frames
total = decoder.get_num_frames()
indices = [int(total * i / 13) for i in range(1, 13)]
frames = [decoder.get_frame_by_index(i) for i in indices]
# Each frame is a DLPack-compatible object
tensors = [torch.from_dlpack(f) for f in frames]
stack = torch.stack(tensors)   # [12, 3, H, W], cuda:0
```

`from_dlpack` is **zero-copy** — PyTorch borrows the NVDEC output buffer.

**Caveats**:

- Wheels are Linux x86_64 (Ubuntu 20.04/22.04) and Win64 only. Pascal+ GPUs.
- Lower-level than TorchCodec; you handle index-vs-timestamp seek logic.
- The decoder caching feature (`nvc.ThreadedDecoder`) is the strongest
  argument over TorchCodec — see Finding 4.

### 5c. decord (DMLC, older)

**Status**: project active but slower release cadence. Pip wheels exist but the
NVDEC-enabled wheels require building from source with `-DUSE_CUDA=ON`,
which is painful on Windows.

```python
import decord
decord.bridge.set_bridge("torch")
vr = decord.VideoReader("video.mp4", ctx=decord.gpu(0))
indices = list(range(0, len(vr), len(vr) // 12))[:12]
frames = vr.get_batch(indices)   # torch tensor on cuda:0, [12, H, W, 3]
```

**Verdict**: not recommended for new work. TorchCodec covers the same ground
with better support.

### Recommended pick: TorchCodec

Reasons:

1. Pip-installable on both target platforms.
2. Active Meta/PyTorch maintenance through 2026.
3. Simpler API than PyNvVideoCodec for our use case (timestamps, not indices).
4. CPU fallback is a single boolean, makes the fallback ladder clean.
5. The performance gap with PyNvVideoCodec is small (~10–15%) and only matters
   at very high concurrency.

### Realistic gain

For a 5-minute 1080p H.264 video with 12 frames:

- Current: 415–880 ms per video.
- TorchCodec + cuPy pHash: **120–200 ms per video.**
- Improvement: **3–5× end-to-end** on stages 2+3.

Where the savings come from:

- No subprocess spawn (decoder is a long-lived Python object).
- No JPEG encode + disk write + JPEG decode.
- No PIL resize (cuPy/cuCIM resize is on GPU, batched).
- DCT is on GPU.

Where they don't:

- NVDEC decode time is the same; that's pure hardware.
- The PCIe download is now 12 × 32 = 384 bytes per video — laughably small,
  but you still pay one CUDA sync per video.

---

## 6. Finding 4 — Batched / persistent decoder context

**Current pattern**: one ffmpeg subprocess per video. Each spawn pays:

1. Process creation (10–30 ms on Win).
2. ffmpeg argument parsing (~5 ms).
3. NVDEC decoder initialisation — typically ~50–100 ms for the first frame,
   then fast. **This is the cost you most want to amortise.**

PyNvVideoCodec 2.0 introduced **decoder caching** specifically for this case:

> "frame sampling and seeking for flexible frame retrieval, decoder caching for
> reusing decoders without full reinitialization, threaded decoder for
> zero-latency decoding in background threads"

The pattern: one persistent decoder pool (size = concurrency limit), and
**reset** it onto each new file via `decoder.reset_source(new_file)` (the API
detail varies; check the 2.0 docs). The CUDA context, the NVDEC engine
allocation, and the format-conversion kernels stay warm.

### Realistic gain on a single-NVDEC SKU (3060 / 3060 Ti)

- Hardware throughput is still limited to one decoder at a time per NVDEC
  engine.
- But **scheduling** improves: the cost of "switch to next video" drops from
  ~100 ms to ~5 ms.
- For 1,000 videos, that's roughly **95 seconds saved on the schedule overhead
  alone**.

### On multi-NVDEC SKUs (RTX A4000+, A5000, A6000, RTX 4090, datacenter cards)

- 2-NVDEC SKUs decode twice as fast in aggregate. PyNvVideoCodec automatically
  load-balances. TorchCodec does not currently (one VideoDecoder = one NVDEC
  instance you don't control).
- The consumer 3060 / 3060 Ti / 4060 / 4070 line is **single NVDEC** — this
  benefit is academic on the listed reference hardware.

### Risks

- Persistent decoder state means one bad file can wedge the worker. Need a
  watchdog timeout, and re-create the decoder on failure.
- Memory: each decoder holds frame buffers in VRAM. A pool of 12 with 1080p
  buffers = ~250 MB. Fine on 8 GB.

---

## 7. Finding 5 — AV1 hardware decode on Ampere / Ada

**Yes, supported in current libraries**, but check codec strings:

| GPU generation | AV1 decode | AV1 encode |
|---|---|---|
| RTX 30 (Ampere) | **Yes** (5th-gen NVDEC) | No |
| RTX 40 (Ada) | Yes | **Yes** (8th-gen NVENC) |
| RTX 20 (Turing) | No | No |
| GTX 16 / older | No | No |

Library support:

- **ffmpeg + `av1_cuvid`**: shipped since FFmpeg 5.0. Your `gpu_detector.py`
  already enumerates it via `ffmpeg -decoders`. **No code change needed.**
- **TorchCodec**: AV1 decode works when ffmpeg has `av1_cuvid`. CUDA backend
  inherits ffmpeg's codec support.
- **PyNvVideoCodec**: explicitly supports `nvc.cudaVideoCodec_AV1` since 1.0.

The interesting question is: **does the current `codec_decoder_map` get
populated correctly with `av1` → `av1_cuvid`?** Yes — the regex
`r"(\w+_cuvid)\s"` in `gpu_detector.py:124` catches it.

### Potential gain

AV1 content is rare today but growing. If your scan target has AV1 (e.g.
modern phone capture from Pixel 8/Galaxy S24, or YouTube re-uploads), the GPU
saves you ~5–20× vs CPU AV1 decode (which is brutal: ~1× real-time at best
even on modern CPUs).

### Risks

- Some AV1 streams use 10-bit or 12-bit color. `format=nv12` in the filter
  chain (line 235) loses bit depth — needs `format=p010le` for 10-bit. Probably
  doesn't matter for hashing (we throw away color anyway), but the filter
  graph will warn.
- AV1 + film grain synthesis (the post-decode noise re-injection): supported
  on Ampere+. Has zero impact on hashing because we resize to 64×64.

---

## 8. Finding 6 — Multi-GPU

**Verdict: premature.** Justification:

- A single RTX 3060 NVDEC can sustain ~400 fps of 1080p H.264 decode. At 30
  fps content, that's ~13 streams decoded faster than real-time. The current
  pipeline only extracts 12 frames per *file*, not 12 frames per second — so
  effective throughput is far higher; one GPU handles thousands of files
  comfortably.
- The bottleneck on a typical "duplicate scan" workload is **not** GPU compute
  but Python/asyncio dispatch and disk I/O.
- Adding multi-GPU support requires routing each video to a specific
  `cuda:N` device, a job queue per GPU, and reconciliation logic. Complex for
  little gain.

If a user has 2+ GPUs, the simplest profitable thing is to launch two backend
processes pinned to different `CUDA_VISIBLE_DEVICES`. No code change needed.
Document this in `gpu-acceleration.md` if anyone asks.

The one scenario where in-process multi-GPU is worth the code: **server
deployment with A100/H100 cards**, multiple NVDECs each, decoding multiple
1000-video libraries in parallel. Not the target audience for this project.

---

## 9. Recommended library stack

| Layer | Pick | Why |
|---|---|---|
| Video decode (modern path) | **TorchCodec** | Mature, pip-installable, GPU-tensor output, official PyTorch support |
| Video decode (fallback) | ffmpeg subprocess (current code, unchanged) | Already works, handles every codec, no new deps |
| GPU image ops (resize, color) | **cuCIM** (via cuPy) | Drop-in replacement for `skimage`, batched, BSD-3-Clause |
| GPU FFT/DCT | **cuPy** (`cupyx.scipy.fft.dctn`) | Battle-tested, batched, exact same API as scipy |
| GPU tensor framework | **PyTorch** (transitively via TorchCodec) | Already required by TorchCodec; convenient interop with cuPy via `__cuda_array_interface__` |
| Audio fingerprint | (no change) | Audio is CPU-bound and tiny; GPU is overkill |

**Out**: VPF / PyNvCodec (deprecated), decord (slower release cycle, build
pain on Windows), torchvision.io.VideoReader (deprecated in favor of
TorchCodec).

### Dependency footprint

Add to `backend/requirements.txt`:

```
torchcodec>=0.10,<1.0       # only when GPU_ENABLED and modern CUDA detected
torch>=2.5,<3.0             # transitive; ~700 MB on Linux, ~2 GB on Windows
cupy-cuda12x>=13.0          # ~150 MB; gated on CUDA 12.x detection
cucim-cu12>=24.10           # optional, ~100 MB
```

These should be **optional extras**, not hard dependencies. The detection in
`gpu_detector.py` already gates GPU work; add `_has_torchcodec` and
`_has_cupy` booleans there, and only use the new path when both are true.

---

## 10. Code sketch: GPU-resident frame extraction + pHash

A *single* function replacing `_extract_and_hash_sync` for the fast path.
Not committed; design draft.

```python
# backend/services/hasher_gpu.py  (new file, not committed)

"""GPU-resident frame extraction + pHash using TorchCodec + cuPy.

This module is loaded lazily by hasher.extract_and_hash() when:
  - settings.GPU_ENABLED is True
  - gpu_detector.get_gpu_info().available is True
  - torchcodec and cupy import successfully
  - the codec is in gpu.codec_decoder_map

On any failure it sets a module-level flag and the caller falls back to
the existing ffmpeg-subprocess path.
"""

from typing import List, Optional
import numpy as np

try:
    import torch
    from torchcodec.decoders import VideoDecoder
    import cupy as cp
    from cupyx.scipy.fft import dctn
    HAS_GPU_STACK = True
except ImportError:
    HAS_GPU_STACK = False


def _phash_batch_gpu(frames_gpu: "cp.ndarray", hash_size: int = 16) -> List[str]:
    """Compute pHash for a batch of frames already on GPU.

    Args:
        frames_gpu: cuPy uint8 array, shape (N, H, W). Single-channel (luma).
        hash_size: pHash size; 16 matches the existing
            imagehash.phash(img, hash_size=16) call. The DCT working size
            is hash_size * 4 = 64.

    Returns:
        List of N hex-encoded hash strings (64 hex chars = 256 bits each).
    """
    img_size = hash_size * 4   # 64
    n = frames_gpu.shape[0]

    # Resize batched: (N, H, W) -> (N, 64, 64). cuCIM if available else manual
    # bilinear via cuPy.
    # We use cupy.ndimage.zoom or a hand-rolled bilinear for speed.
    # Lanczos-3 equivalent would be more PIL-faithful but is overkill here.
    from cupyx.scipy.ndimage import zoom
    zoom_h = img_size / frames_gpu.shape[1]
    zoom_w = img_size / frames_gpu.shape[2]
    resized = zoom(
        frames_gpu.astype(cp.float32),
        zoom=(1, zoom_h, zoom_w),
        order=1,
        prefilter=False,
    )  # (N, 64, 64) float32

    # 2D DCT along (1, 2). cuFFT-backed.
    dct = dctn(resized, axes=(1, 2), norm="ortho")
    lowfreq = dct[:, :hash_size, :hash_size]          # (N, 16, 16)
    flat = lowfreq.reshape(n, hash_size * hash_size)  # (N, 256)
    med = cp.median(flat, axis=1, keepdims=True)      # (N, 1)
    bits = (flat > med).astype(cp.uint8)              # (N, 256)
    packed = cp.packbits(bits, axis=1)                # (N, 32) uint8

    # Single PCIe transfer: 32 * N bytes
    host = packed.get()
    return [host[i].tobytes().hex() for i in range(n)]


def extract_and_hash_gpu(
    file_path: str,
    num_frames: int = 12,
    duration: Optional[float] = None,
) -> dict:
    """Decode N frames via NVDEC + pHash entirely on GPU.

    Returns the same dict shape as _extract_and_hash_sync.
    """
    if not HAS_GPU_STACK:
        return {"file_path": file_path, "hashes": [], "error": "GPU stack unavailable"}

    try:
        decoder = VideoDecoder(file_path, device="cuda")
        if decoder.cpu_fallback != decoder.cpu_fallback.NoFallback:
            # Decoder silently fell back to CPU; bail to ffmpeg path instead
            return {"file_path": file_path, "hashes": [], "error": "CPU fallback"}

        dur = duration or decoder.metadata.duration_seconds
        if not dur or dur <= 0:
            return {"file_path": file_path, "hashes": [], "error": "no duration"}

        # Evenly spaced, avoid the very first/last frame (sometimes black)
        timestamps = [dur * i / (num_frames + 1) for i in range(1, num_frames + 1)]
        batch = decoder.get_frames_played_at(timestamps)   # FrameBatch
        rgb = batch.data   # torch tensor [N, 3, H, W] uint8 on cuda:0

        # Luminance: 0.299 R + 0.587 G + 0.114 B in fixed-point
        # Stay in torch; cuPy can read torch tensors via __cuda_array_interface__
        luma = (
            rgb[:, 0].float() * 0.299
            + rgb[:, 1].float() * 0.587
            + rgb[:, 2].float() * 0.114
        ).to(torch.uint8)                            # [N, H, W]

        # torch -> cupy zero-copy
        luma_cp = cp.asarray(luma)
        hashes = _phash_batch_gpu(luma_cp, hash_size=16)

        return {"file_path": file_path, "hashes": hashes, "error": None}

    except Exception as e:
        return {"file_path": file_path, "hashes": [], "error": str(e)}
```

### How this slots into the existing pipeline

In `backend/services/hasher.py:extract_and_hash` (line 605), add a fast-path
preamble:

```python
async def extract_and_hash(file_path, num_frames=8, duration=None, codec=None, video_info=None):
    # Fast path: GPU-resident decode + hash
    if (
        settings.GPU_ENABLED
        and get_gpu_info().available
        and codec
        and get_gpu_info().supports_codec(codec)
    ):
        try:
            from services import hasher_gpu
            if hasher_gpu.HAS_GPU_STACK:
                # Run in thread executor because TorchCodec is sync
                result = await asyncio.get_event_loop().run_in_executor(
                    _executor,
                    lambda: hasher_gpu.extract_and_hash_gpu(file_path, num_frames, duration),
                )
                if result.get("hashes"):   # non-empty == success
                    return result
        except Exception:
            pass  # fall through to ffmpeg path

    # Existing fallback path (unchanged)
    return await loop.run_in_executor(_executor, lambda: _extract_and_hash_sync(...))
```

Three return outcomes:

1. GPU stack available + works → returns from `hasher_gpu`.
2. GPU stack available + fails (decode error, unsupported codec, CPU
   fallback) → falls through to current ffmpeg-subprocess path.
3. GPU stack unavailable → falls through silently.

The cache (`FileCache.perceptual_hashes`) is **unchanged** — the same hex
strings come out either way, modulo the resize-fidelity caveat in §4.

---

## 11. Failure modes & fallback strategy

A clean ladder, ordered most-aggressive → most-compatible:

| Tier | When it applies | What runs |
|---|---|---|
| **0** | NVIDIA Ampere+ (RTX 30/40), CUDA 12.x, TorchCodec + cuPy installed, codec ∈ `{h264, hevc, av1, vp9}` | `hasher_gpu.extract_and_hash_gpu` — full GPU path |
| **1** | NVIDIA any (Maxwell+), ffmpeg with cuda, no TorchCodec | Current `_extract_and_hash_sync` with `-hwaccel cuda` — unchanged |
| **2** | AMD or Intel iGPU, ffmpeg with vaapi/qsv | Need to add `vaapi`/`qsv` probing to `gpu_detector.py` — out of scope for this doc but a known gap |
| **3** | No GPU, software ffmpeg | Pure CPU path — current code's `use_gpu=False` branch |

The decision tree in `_build_frame_extract_cmd` (line 200–225) already does
tiers 1 and 3. Tier 0 is a new wrapper *above* it. Tier 2 is a separate piece
of work (Intel Arc / Quick Sync, AMD VCN) — neither product line has the
unified Python tensor story that NVIDIA has, so it's pragmatic to keep them
on the ffmpeg-subprocess path indefinitely.

### Per-failure handling inside Tier 0

| Failure | Detection | Action |
|---|---|---|
| TorchCodec falls back to CPU silently | `decoder.cpu_fallback != NoFallback` | Return empty hashes; caller drops to Tier 1 |
| AV1 in a file but no `av1_cuvid` (older driver) | Decoder ctor raises or `cpu_fallback` | Same as above |
| OOM in VRAM | `cudaErrorMemoryAllocation` | Reduce concurrency in the new path; retry; on second failure → Tier 1 |
| Corrupted frame buffer | NaN hashes | Treat as zero-frames result → Tier 1 |
| TorchCodec misseeks (some MKV files) | Output frame timestamps wildly off requested | Detect; Tier 1 |

The dual code path doubles the surface area for bugs. **Mitigation**: gate
Tier 0 behind a config flag (`GPU_NATIVE_DECODE = False` by default) for the
first release; turn on in 0.next once it has cooked in real use.

### Non-NVIDIA hardware

- **AMD (Radeon)**: VCN decode via AMF or VAAPI. PyTorch has no equivalent of
  TorchCodec-CUDA for ROCm video decode (ROCm video SDK is far behind CUDA).
  Stay on Tier 1 with `-hwaccel vaapi` (Linux only).
- **Intel iGPU (Arc, UHD)**: Quick Sync via QSV. Same story — ffmpeg
  subprocess is the only path.
- **Apple Silicon**: VideoToolbox. ffmpeg supports it; not relevant in the
  CUDA-focused scope of this doc.

Adding AMD/Intel GPU detection to `gpu_detector.py` (today only NVIDIA) is the
*single* most-bang-for-buck improvement for non-NVIDIA users, and is
independent of all of this. Worth filing separately.

---

## 12. Rough benchmark expectations

Numbers below are **estimates** for a single RTX 3060 12 GB, AMD Ryzen 5
5600X, 32 GB DDR4, NVMe SSD, processing a typical 5-minute 1080p H.264 file
with 12 frames per video. All times are wall-clock per video, including
asyncio dispatch.

| Step | Current | + Finding 1 only | + Finding 2 only | + Finding 1+2+3 (Tier 0) |
|---|---|---|---|---|
| ffprobe (×0 — pre-supplied) | 0 ms | 0 ms | 0 ms | 0 ms |
| Subprocess spawn | 50 ms | 50 ms | 50 ms | 0 ms |
| NVDEC decode (12 sampled frames) | 250 ms | 80 ms | 250 ms | 80 ms |
| hwdownload → CPU | 12 ms | 12 ms | 12 ms | 0 ms |
| CPU filter (scale, transpose, SAR) | 20 ms | 20 ms | 20 ms | (now on GPU) 5 ms |
| JPEG encode + disk write | 40 ms | 40 ms | 40 ms | 0 ms |
| Tempdir bookkeeping | 8 ms | 8 ms | 8 ms | 0 ms |
| PIL load × 12 | 30 ms | 30 ms | 0 ms | 0 ms |
| Resize × 12 (PIL Lanczos) | 60 ms | 60 ms | 0 ms | 0 ms |
| 2D DCT × 12 (scipy) | 6 ms | 6 ms | 1 ms (GPU) | 1 ms (GPU) |
| Median + bit-pack | 4 ms | 4 ms | 1 ms (GPU) | 1 ms (GPU) |
| PCIe transfer of hashes back | 0 ms | 0 ms | 1 ms | 1 ms |
| Hash string formatting | 5 ms | 5 ms | 5 ms | 5 ms |
| **Total per video** | **~485 ms** | **~315 ms (-35%)** | **~387 ms (-20%)** | **~93 ms (-81%)** |

For a scan of **1,000 videos**:

| Configuration | Stages 2+3 wall-clock (approx) |
|---|---|
| Today (concurrency=12, GPU NVDEC via ffmpeg) | ~485 ms × 1000 / 12 = **40 s** |
| + Finding 1 (I-frame select) | ~315 ms × 1000 / 12 = **26 s** |
| Tier 0 (Findings 1+2+3 combined) | ~93 ms × 1000 / 12 = **7.7 s** |

Caveats:

- The 12× concurrency divider is optimistic. In reality you'll get
  6–10× scaling on a single-NVDEC SKU because decode time-shares.
- Very short videos (< 30 s) have proportionally higher subprocess-spawn
  cost; Tier 0 wins by a larger factor on them (>10×).
- Very long videos (> 30 min) are dominated by NVDEC decode time and Tier 0
  wins by a smaller factor (~2×).

---

## 13. Open questions

These need empirical answers before committing to implementation:

1. **Does TorchCodec's `get_frames_played_at` actually use only N decoded
   frames, or does it decode the whole stream and select?** The
   "performance tips" page implies it seeks to the nearest preceding keyframe
   for each timestamp, so it should be ~O(N keyframes), not O(stream length).
   Verify with one benchmark.
2. **What's the Pillow-vs-cuPy hash agreement rate** on a representative
   sample of real-world content? Cite: > 95% bit agreement is acceptable, < 90%
   means the cache breaks.
3. **How much VRAM does a pool of 12 long-lived TorchCodec decoders cost
   for 1080p / 4K content?** Pessimistic estimate ~2 GB; fine on 8 GB, tight
   on 6 GB.
4. **Does the asyncio dispatch (`run_in_executor` with a ThreadPoolExecutor)
   become the bottleneck once per-video work drops below 100 ms?** It might —
   GIL contention on thread switches is non-trivial. May need to switch to a
   `ProcessPoolExecutor` for the GPU path.
5. **Is the `imagehash.phash(hash_size=16)` resize using LANCZOS or BOX
   filter?** Different versions of Pillow have different defaults. Check before
   re-implementing on GPU to maximise bit agreement.

A one-day spike to wire up TorchCodec on one machine and run all of the above
on ~100 sample videos would settle most of these.

---

## 14. Sources

Library docs and references actually used while writing this:

- [TorchCodec — CUDA decoding example](https://meta-pytorch.org/torchcodec/stable/generated_examples/decoding/basic_cuda_example.html)
- [TorchCodec — Performance tips](https://meta-pytorch.org/torchcodec/stable/generated_examples/decoding/performance_tips.html)
- [TorchCodec — VideoDecoder API](https://meta-pytorch.org/torchcodec/stable/generated/torchcodec.decoders.VideoDecoder.html)
- [TorchCodec on PyPI](https://pypi.org/project/torchcodec/)
- [PyNvVideoCodec 2.0 blog post — NVIDIA Developer](https://developer.nvidia.com/blog/whats-new-in-pynvvideocodec-2-0-for-python-gpu-accelerated-video-processing/)
- [PyNvVideoCodec API Programming Guide](https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-prog-guide/index.html)
- [PyNvVideoCodec on PyPI](https://pypi.org/project/pynvvideocodec/)
- [CV-CUDA — PyNvVideoCodec interop](https://cvcuda.github.io/CV-CUDA/interop/pynvvideocodec.html)
- [VPF — Exporting video frame to PyTorch tensor (wiki, predecessor reference)](https://github.com/NVIDIA/VideoProcessingFramework/wiki/Exporting-video-frame-to-Pytorch-tensor)
- [VPF — Performance analysis (wiki)](https://github.com/NVIDIA/VideoProcessingFramework/wiki/VPF-Performance-analysis)
- [decord (DMLC) repo](https://github.com/dmlc/decord)
- [decord `bridge.set_bridge` and `gpu()` ctx — discussion](https://github.com/dmlc/decord/issues/76)
- [cuPy FFT user guide](https://docs.cupy.dev/en/stable/user_guide/fft.html)
- [cuPy `cupyx.scipy.fft` DCT reference](https://docs.cupy.dev/en/stable/reference/scipy_fft.html)
- [cuCIM repo (RAPIDS)](https://github.com/rapidsai/cucim)
- [Accelerating scikit-image with cuCIM — NVIDIA blog](https://developer.nvidia.com/blog/cucim-rapid-n-dimensional-image-processing-and-i-o-on-gpus/)
- [NVDEC Application Note — concurrent sessions](https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/nvdec-application-note/index.html)
- [Video Codec SDK — NVIDIA Developer](https://developer.nvidia.com/video-codec-sdk)
- [RTX 30 series AV1 decode announcement](https://www.nvidia.com/en-us/geforce/news/gfecnt/202009/rtx-30-series-av1-decoding/)
- [NVENC matrix updates for Ampere — VideoCardz](https://videocardz.com/newz/nvidia-updates-nvdec-video-decoding-and-nvenc-encoding-matrixes-for-ampere-gpus)
- [imagehash repo](https://github.com/JohannesBuchner/imagehash)
- [imagehash perf discussion — issue #69](https://github.com/JohannesBuchner/imagehash/issues/69)
- [fastimagehash (C reference)](https://github.com/simon987/fastimagehash)
- [Perceptual Image Hashing thesis — Zauner](https://www.phash.org/docs/pubs/thesis_zauner.pdf)
- [Python subprocess perf issue tracker — Issue 32383](https://bugs.python.org/issue32383)

Internal references (this repo):

- `backend/services/hasher.py` lines 174–330 — current frame extraction path
- `backend/services/hasher.py` lines 335–388 — current hash computation
- `backend/services/gpu_detector.py` — GPU capability probing
- `backend/services/metadata.py` — ffprobe usage pattern
- `backend/config.py` — `MAX_CONCURRENT_FFMPEG`, `GPU_MAX_CONCURRENT`,
  `KEY_FRAMES_COUNT`
- `docs/gpu-acceleration.md` — current GPU usage doc
- `docs/pipeline.md` — stage-by-stage breakdown
- `docs/duplicate-detection.md` — algorithm reference
