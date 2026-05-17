# Documentation

This folder contains technical documentation for the **Duplicate Video Detector**.

## Contents

| Document | Purpose |
|---|---|
| [architecture.md](architecture.md) | High-level design: process model, request flow, data flow |
| [pipeline.md](pipeline.md) | The scan pipeline, stage-by-stage (incl. cache lookup + sweep) |
| [duplicate-detection.md](duplicate-detection.md) | How videos are determined to be duplicates (pHash + audio fingerprint) |
| [quality-scoring.md](quality-scoring.md) | How the "best" file in a duplicate group is chosen |
| [gpu-acceleration.md](gpu-acceleration.md) | NVIDIA CUDA / CUVID detection and FFmpeg integration |
| [database.md](database.md) | SQLAlchemy schema and data lifecycle (incl. `file_cache`) |
| [api.md](api.md) | REST + WebSocket reference |
| [frontend.md](frontend.md) | React app structure and routing |
| [configuration.md](configuration.md) | All settings, defaults, and what they affect |
| [development.md](development.md) | Running, debugging, and extending the codebase |
| [deployment.md](deployment.md) | Docker / native production setup |

### Research / future work

Start with **[research/README.md](research/README.md)** — it indexes all 14 research notes, calls out cross-cutting consensus, and sequences the recommendations into a 5-tier adoption roadmap (with what's already shipped vs deferred).

| Document | Purpose |
|---|---|
| [research/README.md](research/README.md) | **Start here.** Index + integrated roadmap across all 14 notes |
| [research/algorithmic-improvements.md](research/algorithmic-improvements.md) | Sub-quadratic matching, modern embeddings, Chromaprint |
| [research/pipeline-optimizations.md](research/pipeline-optimizations.md) | Audit of inefficiencies in the existing pipeline (audio FP sampling, tiered extraction, etc.) |
| [research/caching-incremental.md](research/caching-incremental.md) | Design of the Phase 1 cache (now implemented) and Phases 2–4 (cross-scan dedup, SHA-256 fast path, polish) |
| [research/quick-rejection-strategies.md](research/quick-rejection-strategies.md) | head/tail xxh3, fused ffprobe, MinHash LSH on metadata signature |
| [research/codec-aware-shortcuts.md](research/codec-aware-shortcuts.md) | Byte-sample xxhash + frame-size signature pre-pHash |
| [research/frame-sampling-strategies.md](research/frame-sampling-strategies.md) | I-frame seeking inside a blackdetect-trimmed window |
| [research/gpu-acceleration-deep-dive.md](research/gpu-acceleration-deep-dive.md) | NVDEC-to-tensor + GPU-side pHash via cuPy DCT |
| [research/parallelism-patterns.md](research/parallelism-patterns.md) | Streaming pipeline with TaskGroup + bounded queues |
| [research/ann-indexing-structures.md](research/ann-indexing-structures.md) | FAISS binary index per duration bucket (shipped) |
| [research/single-video-hashing.md](research/single-video-hashing.md) | Median-bit SimHash aggregate (shipped as the FAISS prescreen key) |
| [research/audio-fingerprint-alternatives.md](research/audio-fingerprint-alternatives.md) | Multi-segment Chromaprint + spectrogram pHash fallback |
| [research/deep-learning-embeddings.md](research/deep-learning-embeddings.md) | SSCD verifier for cropped / watermarked re-encodes |
| [research/perceptual-hash-comparison.md](research/perceptual-hash-comparison.md) | pHash vs dHash vs wHash vs PDQ |
| [research/production-systems-survey.md](research/production-systems-survey.md) | What Meta / YouTube / open-source dedup tools use at scale |

## Reading order

Newcomers should read in this order:

1. [architecture.md](architecture.md) — get the big picture
2. [pipeline.md](pipeline.md) — understand what a scan actually does
3. [duplicate-detection.md](duplicate-detection.md) and [quality-scoring.md](quality-scoring.md) — the core algorithms
4. The rest as reference

## Quick links

- Backend entry: [`backend/main.py`](../backend/main.py)
- Pipeline orchestrator: [`backend/api/scan.py`](../backend/api/scan.py) — `run_scan_pipeline()`
- Frontend entry: [`frontend/src/App.tsx`](../frontend/src/App.tsx)
- Diagnostic CLI: [`backend/diagnose_pair.py`](../backend/diagnose_pair.py)
