# Architecture

A two-process design: a Python/FastAPI backend that owns all I/O and computation, and a React/Vite SPA that is purely a client.

## High level

```
┌─────────────────────┐         HTTP /api/*          ┌──────────────────────────────┐
│  React SPA          │ ──────────────────────────▶ │  FastAPI (uvicorn :9000)     │
│  Vite dev :3000     │ ◀──── WebSocket /api/scan/  │                              │
│  (proxy → :9000)    │       {id}/ws               │  ┌────────────────────────┐  │
└─────────────────────┘                              │  │  Routers               │  │
                                                     │  │  - api/scan.py         │  │
                                                     │  │  - api/duplicates.py   │  │
                                                     │  │  - api/actions.py      │  │
                                                     │  │  - api/websocket.py    │  │
                                                     │  └──────────┬─────────────┘  │
                                                     │             │                │
                                                     │  ┌──────────▼─────────────┐  │
                                                     │  │  Services              │  │
                                                     │  │  scanner / metadata /  │  │
                                                     │  │  hasher / audio_fp /   │  │
                                                     │  │  comparator / quality/ │  │
                                                     │  │  file_manager / gpu /  │  │
                                                     │  │  scan_control          │  │
                                                     │  └──┬───────────┬─────────┘  │
                                                     │     │           │            │
                                                     │  ┌──▼─────┐  ┌──▼────────┐   │
                                                     │  │ SQLite │  │ FFmpeg /  │   │
                                                     │  │ (async)│  │ FFprobe   │   │
                                                     │  └────────┘  │ + CUVID   │   │
                                                     │              └───────────┘   │
                                                     └──────────────────────────────┘
```

## Process model

- **One uvicorn process** runs the entire backend. SQLAlchemy async + asyncio drive concurrency.
- **One scan at a time.** New scans submitted while another is active enter `status="queued"` and are launched by `_start_next_queued()` in the `finally` block of the active pipeline.
- **In-process state.** Pause/resume/stop signals (`scan_control.py`) and the cached `GPUInfo` (`gpu_detector.py`) live in module-level globals — they do not survive a process restart, which is acceptable because scans cannot resume across restarts either.
- **FFmpeg/FFprobe are subprocesses.** Each frame extraction, audio decode, or metadata probe spawns its own short-lived process. Concurrency is bounded by `asyncio.Semaphore(max_concurrent)`, where `max_concurrent` is `GPU_MAX_CONCURRENT` (12) when CUDA is available, else `MAX_CONCURRENT_FFMPEG` (8).

## Layered backend

```
┌─────────────────────────────────────────────┐
│  api/         — FastAPI routers & WebSocket │
├─────────────────────────────────────────────┤
│  services/    — Domain logic, no HTTP       │
│    scanner    metadata    hasher            │
│    audio_fp   comparator  quality_scorer    │
│    file_mgr   gpu_detect  scan_control      │
├─────────────────────────────────────────────┤
│  models/      — SQLAlchemy ORM + Pydantic   │
│    database.py        schemas.py            │
├─────────────────────────────────────────────┤
│  config.py    — Single Settings (BaseSettings)│
└─────────────────────────────────────────────┘
```

Layers may only call **down**: `api` → `services` → `models`. Services never import from `api`. The single exception is `api/scan.py:run_scan_pipeline`, which orchestrates the whole pipeline and is intentionally large — it is the only place that has end-to-end knowledge of the stages.

## Data flow during a scan

```
  HTTP POST /api/scan {path, options}
        │
        ▼
  scan.py:start_scan
    ─ inserts ScanJob row (status=queued|pending)
    ─ schedules run_scan_pipeline as BackgroundTask
        │
        ▼
  run_scan_pipeline (background coroutine)
    ─ scan_control.register(scan_id)
    ─ Stage 1   discover_videos         ┐
    ─ Stage 1.5 cache lookup + partition  → hits skip stages 2–4b
    ─ Stage 2   extract_metadata + thumbnail   (misses only)
    ─ Stage 3   extract_and_hash        │  Each stage:
    ─ Stage 4a  duration pre-grouping   │   - batches of size max_concurrent*4
    ─ Stage 4b  audio_fingerprint       │   - _pipeline_check between batches
    ─ Stage 5   run_duplicate_pipeline  │   - WS progress updates throttled to ≥2%
    ─ Stage 6   rank_group + persist    ┘
    ─ Stage 7   sweep stale cache rows under root
    ─ scan_control.unregister(scan_id)
    ─ _start_next_queued()
        │
        ▼
  WebSocket /api/scan/{id}/ws
    ─ ConnectionManager fans out to all connected clients
    ─ Frontend useWebSocket reconnects on close
```

## Frontend

Single-page React 19 app. Vite dev server proxies `/api` and `/thumbnails` to the backend so dev and prod use identical URLs. In production, the Python backend serves `frontend/dist` itself with an SPA fallback for client-side routes (see `main.py` lines 71–88).

Components are intentionally vanilla — no global state library. State lives in:
- React Router URL params (which group is being compared, etc.)
- `useScanProgress` and `useWebSocket` hooks for live scan data
- 3-second polling on the Dashboard for stats and scan list

## Persistence

SQLite via `sqlalchemy[asyncio]` + `aiosqlite`. The DB file lives at `backend/duplicate_detector.db` in dev. There are no migrations — the schema is materialised by `Base.metadata.create_all()` on startup. **Schema changes require deleting the DB.** This is acceptable because the data is regenerable by re-scanning.

The DB holds five tables: `scan_jobs`, `video_files` (per-scan records), `duplicate_groups`, `deletion_logs`, and `file_cache` (Phase 1 — cross-scan cache of pipeline outputs keyed by `(file_path, file_size, mtime_ns)`). See [database.md](database.md).

## Static files

- **Thumbnails** — generated during stage 2, stored under `backend/thumbnails/`, served at `/thumbnails/<file>`.
- **Frontend dist** — when present, mounted at `/assets` and served at `/` with SPA fallback. Allows the same uvicorn process to serve a production build on a single port.

## Why this shape

The pipeline's batched-with-checkpoints design is the only non-obvious choice. It exists because users wanted **responsive pause/stop** even during long scans of tens of thousands of files. A single `gather()` over all files would be efficient but uninterruptible. Batching with `_pipeline_check()` between batches gives O(batch_size × per-file-time) worst-case latency from pause to actual pause — typically a few seconds.

The cross-scan `file_cache` table exists because re-scanning the same library re-extracted every frame and re-decoded every audio track even when nothing had changed. With the cache, an unchanged file costs a single `stat()` plus an indexed lookup; only new or modified files pay the full pipeline cost. The identity key is `(file_path, file_size, mtime_ns)` — same as `rsync`/`git`, free since stage 1 already calls `stat()`.
