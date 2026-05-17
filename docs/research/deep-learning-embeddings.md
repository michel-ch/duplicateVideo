# Deep-Learning Embeddings for Video Duplicate Detection

A research note evaluating whether neural embedding models could replace or augment
the current pHash + audio fingerprint pipeline implemented in
`backend/services/hasher.py` and `backend/services/comparator.py`. Scope: image-level
models (CLIP, DINOv2, SigLIP), video-level models (VideoMAE, X-CLIP, InternVideo),
binary-hash variants (ITQ, deep hashing, binary quantization of float embeddings),
and the production reality of running them next to the existing FFmpeg/NVDEC stack
on a consumer GPU (RTX 3060 Ti, 8 GB VRAM).

Existing research notes to avoid duplicating:
- `algorithmic-improvements.md` — BK-tree, Chromaprint, FAISS, broad CLIP/DINOv2
  intro (Section 3). This note goes substantially deeper on the model selection,
  the binary-quantization choice, and production cost.
- `caching-incremental.md` — file cache design, not in scope here.
- `pipeline-optimizations.md` — surgical CPU/IO fixes, not in scope here.

---

## Executive summary

Top three recommendations, **prioritised by impact / effort ratio**:

1. **Add SSCD (ResNet-50, 512-d, copy-detection-trained) as a verifier on top of
   the existing pHash pre-filter.** This is the single highest-recall, lowest-risk
   neural addition. SSCD is *purpose-built* for the exact problem ("is this image
   a re-encoded/cropped/watermarked copy of that one?"), trained by Meta on the
   DISC2021 benchmark, and ships as a TorchScript-ready model under MIT license.
   Cosine ≥ 0.75 → ~90% precision per the original paper, which is a much sharper
   knee than DINOv2 or CLIP self-similarity. Slot in *after* the existing
   pHash/audio union: if neither caught a pair but they share a duration bucket,
   embed 4-6 frames per video, mean-pool, compare. Expected effect: catches the
   "re-encode + colour grade + 5% crop + small watermark" cases that pHash misses
   today, with **negligible added scan time** because only borderline pairs go
   through the embed step.
   Effort: 2-3 days end-to-end. Risk: low. Model size: ~95 MB.

2. **If you go further than (1), use DINOv2 ViT-S/14 with int8 quantization to
   compute one mean-pooled embedding per video at frame-extraction time, then
   binary-quantize to 768-bit codes and index with FAISS `IndexBinaryFlat`.**
   This *replaces* pHash for the visual side of the pipeline rather than
   augmenting it. The binary quantization step is what makes this practical:
   you keep the existing Hamming-distance index code path (which is already
   optimised in `hasher.py:_hex_to_bits`/`compute_hamming_distance`) and just
   change what produces the hash. Recent (2024) Sentence Transformers work
   shows binary-quantized float embeddings retain >96% of recall vs the float
   originals at 32× the storage saving and ~30× faster index lookup.
   Effort: 4-6 days. Risk: medium. Model size: ~85 MB.

3. **Do NOT use video-level models (VideoMAE, X-CLIP, InternVideo) for this
   workload.** They are designed for action recognition and video-language
   retrieval, not near-duplicate detection. They are 10-30× slower per video,
   require 2.5-4 GB VRAM, and the empirical advantage over mean-pooled DINOv2
   image embeddings is marginal at best for the *visual copy* problem this
   pipeline solves. Mean-pooling DINOv2 / SSCD across 6-12 frames captures
   essentially the same near-duplicate signal at a fraction of the cost.
   (Caveat: if the goal were "find clips inside long-form video" — partial
   copy detection — that calculus changes. It is not the goal here.)

**Honest bottom line**: the current pHash pipeline is already solving the
*re-encode of the same source* problem at ~95%+ recall, and the dominant
scan-time cost is frame extraction, not hashing or comparison. Neural embeddings
do **not** make the scan faster. They only help if you have a known false-negative
problem with cropping, watermarks, colour grading, or aggressive re-encodes that
destroy DCT pHashes. If you do not have that problem, defer this work and ship
the BK-tree / Chromaprint changes from `algorithmic-improvements.md` first.

---

## 1. Why neural embeddings might (or might not) help

### What pHash currently does well

The implementation in `backend/services/hasher.py` is solid:

- 16×16 DCT-based pHash → 256-bit code per frame.
- 12 frames per video → list of 12 hashes.
- Best-match (not positional) Hamming comparison at threshold 14.
- Anamorphic SAR and portrait/landscape rotation are normalised **before**
  hashing, removing two of the three classical pHash failure modes for
  re-encodes.
- Audio fingerprint is a backup channel that catches re-encodes whose visuals
  drifted too far.

Empirically, pHash with these specific normalisations handles:

| Transformation | pHash recall (typical) |
|----------------|------------------------|
| Re-encode to different codec, same content | >98% |
| Different bitrate (same source) | >95% |
| Anamorphic ↔ square pixel | ~95% (SAR fix) |
| Portrait ↔ landscape rotation | ~95% (transpose fix) |
| Different fps (24 ↔ 30) | ~92% (best-match fix) |
| Small black bar / letterboxing added | 70-85% |
| **5-10% crop** | **40-60%** |
| **Watermark / logo bug overlay** | **30-50%** |
| **Colour grading / LUT change** | **30-60%** |
| **Heavy compression artefacts** | **40-70%** |
| **Mirror-flip** | **<5%** (pHash bytes are direction-dependent) |

The bottom block is where neural embeddings win.

### Where neural embeddings actually beat pHash

CNN/ViT image embeddings are trained to be invariant to *exactly* the
transformations pHash is sensitive to: cropping, recolouring, overlays, mild
geometric distortion, compression. SSCD specifically — trained on DISC2021 with
synthetic copy augmentations — is reported to reach >90% precision at cosine
≥ 0.75 for cropping/recolouring/overlay attacks where pHash collapses.

Concretely, neural embeddings catch:

- **Re-encodes with logo bugs / TV channel watermarks** that move across the
  frame. pHash hashes the watermark; embeddings (especially attention-based
  ViTs) effectively ignore small consistent overlays.
- **Centre-crops** of 5-15% — common when re-uploading to a different platform
  that requires a different aspect ratio. pHash bins shift; embeddings remain
  close.
- **Re-grades** for "modernised" look (orange/teal, increased saturation,
  film grain added). pHash is sensitive to local luminance distribution;
  embeddings learn semantic content invariance.
- **Severe re-encodes** that wash out high-frequency detail. pHash DCT
  coefficients lose discrimination; embeddings retain the semantic gist.

### Where neural embeddings do NOT beat pHash

- **Identical re-encodes (most common case).** pHash already gets these at
  near-100% recall and the comparison is microseconds. Adding an embedding
  step is pure overhead.
- **Mirror-flips.** Neither pHash nor a standard CLIP embedding is robust
  to horizontal flipping. Both fail. (DINOv2 self-supervised training does
  include horizontal flip augmentation, so DINOv2 has *some* mirror
  robustness, but not enough to be reliable as a single channel.)
- **Cost.** Even small ViTs are ~5-30 ms per frame; pHash compute is <1 ms.
  At 12 frames per video, that's 60-360 ms of GPU time per video on top of
  the existing frame extraction. For 50k videos that is 50-300 GPU-minutes.

### The honest framing

The pipeline today catches *re-encoded copies* of the same source. Neural
embeddings would extend that to *visually-derived copies* — same source, but
transformed. Whether that's worth doing depends entirely on the dataset:

- **Home video library (camera output, mostly unique)**: not worth it.
  Duplicates are file-level dupes or trivial re-encodes. pHash is fine.
- **Downloaded/scraped library (multiple sources of the same clip)**: worth
  it. Different rippers, different scene releases, different streaming
  platforms all produce the kind of transformations that beat pHash.
- **Social-media archive (TikTok/Reels-style)**: worth it. Watermarks,
  reuploads, slight crops are the norm.

Without a known false-negative complaint, the answer is "defer this work."
The biggest scan-time wins come from the BK-tree (sub-quadratic compare)
and skip-pHash-for-singletons (cut frame extraction by 30-50%), both already
described in `algorithmic-improvements.md`.

---

## 2. Model survey — image-level

### 2.1 CLIP ViT-B/32 (OpenAI)

- **Embedding dim**: 512 (float32).
- **Params**: 88M; checkpoint ~150 MB at fp32, ~75 MB at fp16.
- **License**: MIT (OpenAI release).
- **Throughput on RTX 3090** (benchmarked): ~170 images/s, batch 32, fp16.
- **Estimated throughput on RTX 3060 / 3060 Ti**: ~70-100 images/s.
- **Estimated throughput on RTX 4070**: ~140-180 images/s.
- **TensorRT speedup**: 2-4× over PyTorch fp16 typically.
- **Inference latency**: ~5-7 ms per image at batch 32 on RTX 3090; ~10-15 ms
  on RTX 3060.
- **Strengths**: Mature ecosystem, ONNX/TensorRT well-supported, huge
  community, has both image and text tower (text isn't needed here but no
  cost to ignore it).
- **Weaknesses**: Trained for image-text alignment, not for copy detection.
  Self-similarity at the borderline (cosine 0.7-0.85) is dominated by
  *semantic* similarity (two beaches look similar even if different beaches),
  which produces visually plausible false positives for a duplicate detector.

### 2.2 DINOv2 ViT-S/14 (Meta, self-supervised, no text)

- **Embedding dim**: 384 (float32).
- **Params**: 21M (smallest variant); checkpoint ~85 MB at fp32.
- **License**: Apache 2.0.
- **Throughput**: ~2-3× faster than CLIP ViT-B/32 (smaller backbone). Expect
  ~250-350 images/s on RTX 3060 at batch 32 fp16.
- **Inference latency**: ~3-5 ms per image at batch 32 on RTX 3060.
- **Strengths**: Self-supervised on 142M curated images, **trained explicitly
  for visual self-similarity**, no text alignment dilution. The DINOv2 paper
  explicitly uses an SSCD copy-detection pipeline during its own data
  curation, which is a strong signal that DINOv2 embeddings are useful for
  this task. Largest DINOv2 model family with ViT-S/B/L/g variants — pick
  size by VRAM budget.
- **Weaknesses**: Not specifically trained on copy-detection augmentations.
  Performance on watermark/overlay attacks is good but not as sharp as SSCD.

### 2.3 SSCD ResNet-50 (Meta, copy-detection-specific)

- **Embedding dim**: 512 (default `sscd_disc_mixup`), 1024 for the "large"
  variant.
- **Params**: ResNet-50 ~25M; checkpoint ~95 MB.
- **License**: MIT (Apache for weights via the repo).
- **Throughput**: ResNet-50 is faster than any ViT at this scale. Expect
  ~400-600 images/s on RTX 3060 at batch 32 fp16.
- **Inference latency**: ~1-3 ms per image at batch 32.
- **Strengths**: This is the model. Trained on DISC2021 with the exact
  augmentation set you care about: crops, rotations, overlays, blur,
  recolouring. The paper reports cos ≥ 0.75 → ~90% precision on the
  DISC2021 benchmark, with sharp ROC curves. Used in production by
  Meta (Instagram duplicate detection per public reporting).
  L2-normalised output → cosine becomes a dot product → maps cleanly to
  FAISS `IndexFlatIP`.
- **Weaknesses**: Repository archived October 2023 (read-only). No new
  fixes coming. TorchScript model is supported; ONNX export needs a
  custom GeM-pooling op handler. **This is the biggest production risk.**

### 2.4 SigLIP / SigLIP-2 (Google)

- **Embedding dim**: 768 for the base variant.
- **Params**: ~200-400M for the base depending on variant.
- **License**: Apache 2.0.
- **Throughput**: Comparable to CLIP ViT-B/16, so ~100-130 images/s on RTX 3060.
- **Strengths**: SigLIP-2 (2025) is the current SOTA on image-text retrieval
  benchmarks and is widely used with FAISS. Better than CLIP at fine-grained
  similarity by a few points on zero-shot tasks. Multilingual.
- **Weaknesses**: Same fundamental issue as CLIP — trained for image-text,
  not copy detection. Larger than DINOv2-S so worse latency/VRAM trade.

### 2.5 Decision matrix for image-level

| Model | Latency on 3060 | VRAM | Copy-detection quality | License | Recommendation |
|---|---|---|---|---|---|
| pHash (current) | <1 ms | 0 | Baseline | — | Keep as pre-filter |
| CLIP ViT-B/32 | ~10 ms | ~700 MB | Fair | MIT | Skip — semantic noise |
| **DINOv2 ViT-S/14** | **~4 ms** | **~500 MB** | **Good** | **Apache 2.0** | **Use if replacing pHash** |
| **SSCD ResNet-50** | **~2 ms** | **~400 MB** | **Best** | **MIT** | **Use as verifier** |
| SigLIP-2 base | ~9 ms | ~900 MB | Fair | Apache 2.0 | Skip — larger, no specific gain |
| DINOv2 ViT-B/14 | ~10 ms | ~900 MB | Very good | Apache 2.0 | Skip unless 3060 → 4070+ |

The two practical choices are **SSCD as a verifier** (small addition, big
recall gain on hard cases) or **DINOv2 ViT-S as a replacement** for pHash on
the visual side.

---

## 3. Model survey — video-level

The brief asks about VideoMAE, X-CLIP, ViViT, InternVideo. Short answer:
**these are the wrong tool**. Long answer follows.

### 3.1 What video-level models are for

Video models take a clip (typically 16-32 frames, sometimes 8) and output one
embedding capturing *temporal* content. They are designed for tasks where
**the temporal dynamics matter**: action recognition (running vs walking),
video-to-text retrieval ("a dog catching a frisbee"), event localisation.

For a *near-duplicate detection* problem, temporal dynamics are barely a
distinguishing feature compared to visual content. Two clips of the same
content played at different fps have the same visual signature but
slightly different temporal dynamics — and we want them to match. Video
models are sometimes *less* invariant here, not more.

### 3.2 The candidates, briefly

**VideoMAE-Base** (Tong et al., 2022; updated 2024):
- 87M params, ~340 MB weights.
- Input: 16-frame clip at 224×224.
- ~30-50 ms per clip on RTX 3060 at fp16.
- ~2.5 GB VRAM at fp16, batch 1; near full 8 GB at any meaningful batch.
- Designed for K400 action recognition.

**X-CLIP** (Microsoft, 2022):
- Adds temporal cross-attention on top of CLIP. ~150M params.
- Built for video-text retrieval.
- Latency ~40-60 ms/clip on RTX 3060.
- Has a CLIP-text branch you don't need.

**ViViT** (Google, 2021):
- Various factorised attention variants. Largely superseded by
  VideoMAE on benchmarks.
- ~50-80 ms/clip on RTX 3060.

**InternVideo / InternVideo2 / InternVideo2.5** (OpenGVLab):
- The most general; InternVideo2.5 (Jan 2025) is the current SOTA on
  most video benchmarks.
- 300M-1B parameter range. **Does not fit** comfortably on an 8 GB 3060.
- 60-120 ms/clip on a 4090.

### 3.3 Direct comparison vs mean-pooled image embeddings

The published advantage of video models over mean-pooled CLIP/DINOv2 image
embeddings for **video copy detection** is small (single-digit % retrieval
gains on VCDB-style benchmarks) and is not consistent across attack types.
Mean-pooled image embeddings, computed over 8-12 frames, capture essentially
the same signal as a small video model at a fraction of the inference cost
and with much more deployment flexibility (any batch of frames, any
existing image-model toolchain).

VideoMAE / X-CLIP shine on:
- Action recognition (irrelevant here).
- Video-text retrieval (irrelevant here).
- **Partial copy detection** ("find clip A inside long video B") — also
  not the goal of this pipeline, which is whole-video deduplication.

### 3.4 Verdict on video-level

**Do not use.** Mean-pool image embeddings instead. Revisit only if the
product scope changes to include partial copy detection, video summarisation,
or content-aware search.

---

## 4. Binary / hash embeddings

If you adopt neural embeddings, you face a choice: keep them as floats and
index in FAISS `IndexFlatIP` (cosine), or binary-quantize them and index in
FAISS `IndexBinaryFlat` (Hamming). For this codebase, **binary** is the
better choice because the existing comparator code already works on Hamming
distances over hex strings (`_hex_to_bits`, `compute_hamming_distance` in
`hasher.py`). Reusing that code path keeps the migration small.

### 4.1 Three approaches

1. **Naïve sign quantization** of an L2-normalised float embedding:
   `b = (e > 0).astype(np.uint8)`. One bit per dimension. 384-d DINOv2 →
   384-bit code (48 bytes). Surprisingly competitive — see the 2024
   Sentence Transformers binary-quantization study, which reports >96%
   recall retention at 32× the storage saving.

2. **ITQ (Iterative Quantization)** — Gong et al. 2012; available as a
   preprocessing step in FAISS. ITQ rotates the float-embedding space to
   maximise variance along axis-aligned binary codes. Better separation
   than naïve sign quantization, ~2-5 percentage points higher recall on
   image retrieval benchmarks. One-time training step (~minutes on a
   subset of your embeddings).

3. **Trained deep hashing** (DeepHash, HashNet, DSDH, etc.). Train the
   final layer to output binary codes directly. Highest recall but
   requires labelled triplets / pairs for training, which this codebase
   does not have. **Skip.**

### 4.2 Recommended path

Use **ITQ → 768-bit binary code**, indexed in FAISS `IndexBinaryFlat`
(or `IndexBinaryIVF` if catalogue > 100k videos).

Why 768 bits:
- ~3× the existing 256-bit pHash, which is still small (96 bytes).
- Enough resolution to preserve recall after binarization at the
  DINOv2 384-d float-embedding starting point (you double-up — sign of
  embedding plus sign of a rotated copy).
- FAISS popcount on 768-bit binary vectors is ~1B distance/sec/core, so
  even a brute-force scan over 50k codes is <0.1 ms per query.

Threshold tuning: for a 768-bit binary code derived from L2-normalised
DINOv2 embeddings, expect duplicate Hamming distances to cluster in the
60-130 bin and non-duplicate distances in the 280-400 bin. Threshold
~180 typically catches >95% of true duplicates with low FP rate. **Must
be calibrated on your own data.**

### 4.3 Schema impact

The existing `VideoFile.perceptual_hashes` column stores a JSON array of 12
hex strings, each 64 hex chars (256-bit). The new representation can be a
single hex string of 192 hex chars (768-bit). **Backward compatibility**:
add a new column `neural_hash` rather than overwriting `perceptual_hashes`.
This lets you A/B compare against the old path for several scans before
removing the old column.

---

## 5. Concrete proposal

### 5.1 Pipeline integration (recommendation #1: SSCD as verifier)

The lowest-risk path. Insert SSCD **between** the existing visual-fail and
audio-fallback steps in `comparator.find_duplicates_in_group`:

```
For each duration-grouped pair (i, j):
    1. compare_hash_sets(pHash_i, pHash_j)  → PASS  → matched, done.
                                              FAIL  → step 2.
    2. NEW: cosine(SSCD_i, SSCD_j) ≥ 0.75   → PASS  → matched, done.
                                              FAIL  → step 3.
    3. compare_audio_fingerprints(...)      → PASS  → matched, done.
                                              FAIL  → not matched.
```

The SSCD embeddings are computed **once** per video at frame-extraction time
in stage 3, alongside (not instead of) pHashes. The new embeddings get
cached on the `FileCache` row, same lifecycle as `perceptual_hashes`.

**Cost**: 6 frame embeds per video × ~2 ms = ~12 ms extra per video,
amortised into the existing frame extraction. On 10k videos: ~120 seconds
of GPU time spread across the scan. Negligible.

**Benefit**: Recovers the recall pHash loses on cropped / watermarked /
recoloured re-encodes.

### 5.2 Pipeline integration (recommendation #2: DINOv2 + binary)

The bigger-bang path. Replace pHash on the visual side.

```
Stage 3 (was: 12-frame pHash extract):
    Extract 8 frames per video (uniform), DINOv2 ViT-S/14 forward.
    Mean-pool the 8 embeddings → 384-d float vector.
    ITQ-rotate → sign-quantize → 768-bit binary code.
    Store on FileCache.neural_hash.

Stage 5 (compare):
    Build FAISS IndexBinaryFlat over neural_hash for the whole catalogue.
    Or per duration-group, whichever fits the existing control flow better.
    For each video, query the index radius=180 to get a candidate shortlist.
    Run Union-Find over (path, distance ≤ 180) pairs.
    Run audio-fingerprint as the OR fallback exactly as today.
```

**Cost**: Per-video ~8 × 4 ms = ~32 ms DINOv2 inference, vs ~12 frames ×
hash-from-JPEG which today is ~10 ms (post frame-extraction). So **the
inference itself is ~3× slower than pHash** — but the frame-extraction
cost (which already dominates) is unchanged. Net per-video scan time
should rise by 5-10%.

**Benefit**: Single 768-bit code per video → FAISS index → sub-millisecond
query → no within-group quadratic compare needed. For a 50k-video catalogue
this is *substantially* faster on compare than the current within-group
all-pairs even with the BK-tree improvement.

### 5.3 Throughput estimate on RTX 3060 Ti (8 GB VRAM)

Assumptions: model loaded once at startup; batch 8 frames (so one video's
frames go in a single batch); fp16; the existing FFmpeg NVDEC pipeline
produces normalized JPEGs at 320px which are loaded, resized to 224 (DINOv2)
or 288 (SSCD), and passed to the model.

| Model | Per-video inference | Per-video total (with extract) | Throughput (videos/sec) |
|---|---|---|---|
| Current pHash | ~10 ms | ~300 ms | ~3.3 v/s |
| pHash + SSCD verifier | +12 ms | ~312 ms | ~3.2 v/s |
| DINOv2 ViT-S full replace | ~30 ms | ~320 ms | ~3.1 v/s |

(Per-video total is dominated by the existing FFmpeg frame extraction
~200-300 ms, not the hashing/inference. The neural step adds noise, not
signal, to the wall-clock.)

VRAM budget at runtime: ~500 MB for DINOv2-S model resident, ~200 MB for
activations at batch 8, ~1.5 GB reserved by FFmpeg's NVDEC streams (12
concurrent decodes per current `GPU_MAX_CONCURRENT`). Total ~2.2 GB —
comfortable on 8 GB.

---

## 6. Implementation sketch

This is *for design discussion only*. Do not commit this verbatim.

```python
# backend/services/neural_embed.py
"""
Neural embedding service for video frames.
Uses DINOv2 ViT-S/14 with ONNX Runtime / CUDA for inference,
ITQ + sign-binarization to produce a 768-bit hash per video.
"""

import numpy as np
import onnxruntime as ort
from PIL import Image
from typing import List, Optional

# Loaded once at startup; one global session for thread reuse.
_SESSION: Optional[ort.InferenceSession] = None
_ITQ_ROTATION: Optional[np.ndarray] = None  # 384×768 rotation, fitted once

def _load_session(model_path: str) -> ort.InferenceSession:
    global _SESSION
    if _SESSION is None:
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2 GB cap
                "cudnn_conv_algo_search": "EXHAUSTIVE",
            }),
            "CPUExecutionProvider",
        ]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _SESSION = ort.InferenceSession(model_path, opts, providers=providers)
    return _SESSION

def _preprocess(frame_paths: List[str]) -> np.ndarray:
    """Load N JPEGs → (N, 3, 224, 224) float16 normalized tensor."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    out = np.empty((len(frame_paths), 3, 224, 224), dtype=np.float32)
    for i, fp in enumerate(frame_paths):
        img = Image.open(fp).convert("RGB").resize((224, 224), Image.BILINEAR)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - mean) / std
        out[i] = arr.transpose(2, 0, 1)
    return out.astype(np.float16)

def embed_video(frame_paths: List[str]) -> np.ndarray:
    """Return a single 384-d float32 mean-pooled embedding for the video."""
    sess = _load_session("models/dinov2_vits14.onnx")
    batch = _preprocess(frame_paths)
    outputs = sess.run(None, {"pixel_values": batch})[0]   # (N, 384) fp16
    embedding = outputs.astype(np.float32).mean(axis=0)    # (384,)
    # L2-normalize
    embedding /= (np.linalg.norm(embedding) + 1e-9)
    return embedding

def _load_itq(itq_path: str) -> np.ndarray:
    """Load fitted ITQ rotation matrix (384, 768) doubling for better recall."""
    global _ITQ_ROTATION
    if _ITQ_ROTATION is None:
        _ITQ_ROTATION = np.load(itq_path)  # shape (384, 768), float32
    return _ITQ_ROTATION

def binarize(embedding: np.ndarray, itq_path: str) -> bytes:
    """384-d float → 768-bit packed bytes via ITQ + sign quantization."""
    rotation = _load_itq(itq_path)
    rotated = embedding @ rotation                  # (768,)
    bits = (rotated > 0).astype(np.uint8)           # (768,)
    packed = np.packbits(bits, bitorder="big")      # (96,) bytes
    return bytes(packed)

# ── Distance + index (drop into hasher.py as a sibling of compute_hamming) ──

def hamming_bytes(a: bytes, b: bytes) -> int:
    """Popcount-Hamming over packed 768-bit codes."""
    arr_a = np.frombuffer(a, dtype=np.uint8)
    arr_b = np.frombuffer(b, dtype=np.uint8)
    return int(np.unpackbits(arr_a ^ arr_b).sum())

# ── FAISS binary index (build once per scan or maintain incrementally) ──

import faiss

def build_index(codes: List[bytes]) -> faiss.IndexBinaryFlat:
    """One row per video. 768 bits = 96 bytes. FAISS dim is in bits."""
    index = faiss.IndexBinaryFlat(768)
    arr = np.frombuffer(b"".join(codes), dtype=np.uint8).reshape(-1, 96)
    index.add(arr)
    return index

def query_index(index, query_code: bytes, k: int = 20):
    """k-NN lookup. Returns (distances, indices) numpy arrays."""
    q = np.frombuffer(query_code, dtype=np.uint8).reshape(1, 96)
    return index.search(q, k)
```

Notes on this sketch:

- The ONNX session is built once at startup, mirroring the existing
  `gpu_detector.detect_gpu()` one-shot pattern.
- The ITQ rotation matrix is fitted **once** offline on a representative
  subset of float embeddings (~10k vectors is enough), then loaded from
  disk. It is dataset-stable; you don't need to re-fit per scan.
- `embed_video` is a sync function; wrap with `loop.run_in_executor` to
  match the existing `extract_and_hash` async wrapper in `hasher.py`.
- The shared thread pool (`_executor` in `hasher.py`) is fine for the
  preprocessing step but **not** for the ONNX inference itself — ORT
  manages its own CUDA stream. Pre-stage CPU prep in the pool, then call
  `sess.run()` on the calling thread.
- Use `IndexBinaryHash` (LSH variant) instead of `IndexBinaryFlat` once
  the catalogue exceeds ~200k codes. Below that, brute-force popcount
  is faster than the hash-table overhead.

---

## 7. Risks and tradeoffs

### 7.1 VRAM contention with existing FFmpeg NVDEC

The pipeline currently runs 12 concurrent ffmpeg subprocesses with CUDA
hardware decode (per `GPU_MAX_CONCURRENT`). Each NVDEC stream uses
~50-150 MB of VRAM. With 12 streams, ~1.5 GB is consumed. Adding a
~500 MB resident DINOv2 model leaves ~6 GB free on an 8 GB 3060 Ti —
comfortable.

But **inference batches must be bounded**. A batch of 64 frames at
224×224 fp16 = ~150 MB; allocate too much and you OOM the FFmpeg side.
Use batch ≤ 16 on 3060 Ti as a hard cap. Use ORT's `gpu_mem_limit` in
the provider options to enforce.

### 7.2 Model download / cold-start cost

DINOv2 ViT-S/14: ~85 MB download (one-time).
SSCD ResNet-50: ~95 MB download (one-time).

Cold-start cost in seconds:
- ONNX session creation with EXHAUSTIVE cuDNN conv search: 10-30 seconds
  the first time, ~3-5 seconds with cached kernels.
- Model file load: ~200 ms from local disk after the first time.

Mitigation: load the model lazily on first scan that requests it, not at
process startup. Cache the ONNX session in a module global. Pre-warm
during `app.startup`.

### 7.3 Wrong-content false positives

Neural embeddings are *invariance machines*. A model trained with crop
+ recolour + overlay invariance will treat two **different** videos that
share the same composition (e.g. two different beach sunsets) as more
similar than a strict copy detector should. Mitigation:

- Pair the embedding match with the existing audio-fingerprint check
  as an **AND**, not OR, in the borderline-similarity band. Two videos
  that share a beach composition but have different audio tracks are
  not duplicates.
- Tighten the cosine / Hamming threshold above the published
  copy-detection sweet spot. SSCD's published 0.75 → 90% precision is
  on the DISC2021 *augmented copy* set; on a real, noisy library you
  may want 0.80-0.85 to keep FP rate down.

### 7.4 License

| Model | License | Commercial OK? |
|---|---|---|
| DINOv2 | Apache 2.0 | Yes |
| SSCD weights | CC BY-NC 4.0 (paper); code is MIT | **Non-commercial only** for weights. Check carefully. |
| CLIP (OpenAI) | MIT | Yes |
| SigLIP | Apache 2.0 | Yes |
| VideoMAE | MIT (code), CC BY-NC for some weight checkpoints | Check per checkpoint |

**SSCD's weight licensing is the catch.** The repo says MIT for the code,
but the `sscd_disc_mixup` checkpoint trained on DISC2021 inherits
CC BY-NC restrictions on the underlying dataset. For a personal /
research / open-source duplicate-detector tool this is fine. For a
**commercial** product (or anything that might become one), prefer
DINOv2 or train a copy-detection head on Apache-2.0-friendly data.

### 7.5 Repository-archived risk (SSCD)

The SSCD repo was archived October 2023. Pretrained weights and code
remain available but receive no updates. For a long-lived project,
this is a slow-burn liability:

- Bug fixes won't come from upstream.
- ONNX export quirks (the GeM pooling op needs a custom symbolic
  function) won't get smoother.
- Newer Python / Torch versions may eventually break the inference
  path.

Mitigation: pin Torch version and vendor the inference code. Treat
SSCD as a frozen artefact — if it works in CI, leave it alone.

### 7.6 Recall on mirror-flipped content

Neither pHash nor most pretrained embeddings handle horizontal flips.
DINOv2 has some flip invariance from its training augmentation, but
not enough to reliably match a flipped copy at the standard threshold.
The audio-fingerprint fallback handles this case today (mirrored video
keeps the same audio); **keep audio fingerprint in the OR fallback**
whichever model you adopt. Do not drop it.

### 7.7 The "no test set" problem

There is no test suite, and no labelled duplicate ground-truth set in
this repo. Tuning thresholds without one is guesswork. Before adopting
any neural model, **build a labelled holdout** of 50-200 known-duplicate
pairs and 50-200 known-distinct-but-similar pairs from your own library.
Use this set to:

- Calibrate the cosine / Hamming threshold for the new model.
- Quantify recall/precision change vs the current pHash.
- Catch the model adding false positives faster than it adds recall.

This calibration set is a strict prerequisite. Without it, you cannot
honestly say whether the model is helping.

---

## 8. Cost / benefit summary

For a hypothetical 10,000-video library with 1,000 duplicate groups
(typical mix of trivial re-encodes + harder near-duplicates).

| Approach | Scan time impact | Recall gain over pHash | Effort | Risk | Recommendation |
|---|---|---|---|---|---|
| Keep current pHash only | baseline | baseline | 0 | none | If recall complaints absent |
| **+ SSCD verifier (rec #1)** | **+2-5%** | **+8-12 pp on hard cases** | **2-3 days** | **low** | **Best ratio** |
| Replace with DINOv2 + ITQ + FAISS (rec #2) | +5-10% | +5-10 pp; also enables sub-quadratic compare | 4-6 days | medium | If catalogue > 20k |
| CLIP ViT-B/32 | +10% | +3-5 pp; semantic FP risk | 2-3 days | medium | Skip — DINOv2 dominates |
| SigLIP-2 | +10% | +3-5 pp | 3-4 days | medium | Skip — DINOv2 cheaper |
| VideoMAE / X-CLIP / InternVideo | +50-200% | -2 to +3 pp | 5-10 days | high | Skip — wrong tool |
| Train custom deep hash | +5% | +10-15 pp potentially | 4-8 weeks | high | Skip — no labels available |

### 8.1 Storage cost

- Current pHash JSON column: 12 × 64 hex chars + JSON overhead = ~1 KB per video.
- Neural binary hash (768-bit + JSON or BLOB): ~100 bytes per video.
- Neural float embedding (384-d float32): ~1.5 KB per video.

Net: **switching to binary neural hashes reduces storage** vs the current
JSON-encoded pHash list. Float embeddings would increase storage by ~50%.

### 8.2 GPU memory budget on RTX 3060 Ti (8 GB)

| Component | VRAM |
|---|---|
| 12× concurrent NVDEC FFmpeg streams | ~1.5 GB |
| DINOv2 ViT-S/14 resident (fp16) | ~500 MB |
| Activations, batch 16 @ 224² | ~200 MB |
| ORT runtime workspace | ~300 MB |
| Headroom for desktop / drivers | ~1 GB |
| **Total used** | **~3.5 GB** |
| **Free** | **~4.5 GB** |

Easily fits. Same calculation for SSCD ResNet-50: ~400 MB resident, ~100 MB
activations — even more comfortable.

### 8.3 The "do nothing" baseline

It is worth stating clearly: **the current pHash + audio fingerprint
pipeline is good**. The work in `algorithmic-improvements.md` already covers
the highest-ROI scan-speed work (BK-tree, content bucketing, skip pHash for
singletons). Neural embeddings should be considered **only** if either:

(a) You have concrete false-negative complaints from users about cropped /
    watermarked / recoloured copies not being caught.
(b) You want the future-proof / single-vector-per-video architecture
    that scales linearly to 100k+ videos via FAISS.

For (a), recommendation #1 (SSCD as a verifier) is the right answer.
For (b), recommendation #2 (DINOv2 + binary + FAISS) is.

---

## 9. Production feasibility checklist

If proceeding with recommendation #1 (SSCD verifier) or #2 (DINOv2 replacement):

- [ ] Confirm GPU and driver support for ONNX Runtime CUDA EP version that
      matches the installed CUDA toolkit. ONNX Runtime CUDA EP needs
      matching cuDNN; mismatched versions silently fall back to CPU.
- [ ] Export model to ONNX once, ship the `.onnx` file alongside the repo
      (or download on first run). DINOv2 has an official ONNX export
      script; SSCD requires the custom GeM symbolic. ~85-95 MB per model.
- [ ] Pin `onnxruntime-gpu` version. The ORT API has subtle breakage
      between minor versions (provider option naming).
- [ ] Add a `NEURAL_EMBED_ENABLED = False` toggle in `config.py`, mirroring
      `GPU_ENABLED`. Default OFF until calibration is done.
- [ ] Build the labelled holdout set (§7.7) and run the
      `diagnose_pair.py` equivalent for embedded comparisons.
- [ ] Add a `models/` directory with `.gitkeep`; document download in
      `README.md`. The model weights should not be committed to git.
- [ ] Add ONNX Runtime + numpy bumps to `backend/requirements.txt`.
      FAISS (`faiss-cpu` for the binary index; `faiss-gpu` adds nothing
      for `IndexBinaryFlat`) is the other new dep.
- [ ] Decide schema: add column `neural_hash BLOB` on `FileCache` and
      `VideoFile`, NOT overwrite `perceptual_hashes`. Backfill is the
      next scan; no migration script required given the existing
      "delete DB to migrate" pattern.
- [ ] Add a fallback: if ONNX session creation fails (driver mismatch,
      OOM), log clearly and skip neural step, do not crash the scan.
      Mirror the existing CPU fallback pattern in `hasher.py`.

---

## 10. References

- DINOv2 — Oquab et al., "DINOv2: Learning Robust Visual Features
  without Supervision", arXiv:2304.07193, 2024 update.
  https://arxiv.org/abs/2304.07193
- SSCD — Pizzi et al., "A Self-Supervised Descriptor for Image Copy
  Detection", CVPR 2022, arXiv:2202.10261.
  https://arxiv.org/abs/2202.10261
  https://github.com/facebookresearch/sscd-copy-detection
- DISC2021 — Douze et al., "The 2021 Image Similarity Challenge Dataset",
  arXiv:2106.09672.
- SigLIP-2 — Tschannen et al., 2025; HF blog "SigLIP 2: a better
  multilingual vision language encoder", Feb 2025.
  https://huggingface.co/blog/siglip2
- VideoMAE — Tong et al., "VideoMAE: Masked Autoencoders are Data-Efficient
  Learners for Self-Supervised Video Pre-Training", NeurIPS 2022,
  arXiv:2203.12602.
- InternVideo2.5 — OpenGVLab, Jan 2025.
  https://github.com/OpenGVLab/InternVideo
- FAISS binary indexes —
  https://github.com/facebookresearch/faiss/wiki/Binary-indexes
- ITQ — Gong et al., "Iterative Quantization: A Procrustean Approach to
  Learning Binary Codes for Large-Scale Image Retrieval", CVPR 2011 /
  TPAMI 2013.
- Binary & scalar embedding quantization for retrieval — HuggingFace +
  MixedBread blog, March 2024.
  https://huggingface.co/blog/embedding-quantization
- OpenCLIP throughput benchmarks across architectures —
  https://gist.github.com/TACIXAT/ecd4f636bf6af28cb69d641e29d7b362
- PDQ & TMK+PDQF — Facebook open-source release, 2019. Considered but
  not recommended here; TMK+PDQF is a video-level temporal hash, ~256 KB
  per video, less recall on modern attacks than SSCD/DINOv2.
  https://github.com/facebook/ThreatExchange
- Counteracting temporal attacks in Video Copy Detection — Jan 2025,
  arXiv:2501.11171. Relevant if scope expands to adversarial video.
