# Development

How to run, debug, and extend the codebase.

## Running locally

### Quick start (Windows)

Double-click `start.bat`. Two terminal windows open:
- Backend: `python -m uvicorn main:app --host 0.0.0.0 --port 9000 --reload`
- Frontend: `npm run dev -- --port 3000`

### Manual

```bash
# Backend (one terminal)
cd backend
python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 0.0.0.0 --port 9000

# Frontend (another terminal)
cd frontend
npm install
npm run dev -- --port 3000
```

Backend serves at http://localhost:9000, frontend at http://localhost:3000. The Vite proxy forwards `/api` and `/thumbnails` to the backend, so the frontend always uses relative URLs.

### Prerequisites

- **Python 3.10+** (3.11 in Docker)
- **Node.js 18+**
- **FFmpeg** + **ffprobe** on PATH. On Windows: `choco install ffmpeg`.
- (Optional) **NVIDIA GPU + CUDA-built FFmpeg** for hardware acceleration. Standard `choco install ffmpeg` includes CUVID support.

## Diagnostics

### Diagnose a single pair

When two specific videos aren't matching the way you expect:

```bash
python backend/diagnose_pair.py "path/to/A.mp4" "path/to/B.mp4"
```

Prints durations, SAR/rotation, frame counts, video similarity, audio correlation, and the final verdict. Reuses the actual pipeline functions, so the result matches what a full scan would produce.

### Inspect the DB

```bash
sqlite3 backend/duplicate_detector.db
.schema
SELECT id, status, total_files, scanned_files FROM scan_jobs;
SELECT id, similarity_score, file_count FROM duplicate_groups ORDER BY total_wasted_space DESC LIMIT 10;
```

### Reset everything

```bash
# stop the backend, then:
rm backend/duplicate_detector.db
rm -rf backend/thumbnails/*
```

The DB is recreated on next startup (no migrations); thumbnails on the next scan. **This also wipes the `file_cache` table**, so the next scan will be a full one (all files are misses).

### Inspect or clear just the cache

```bash
# how many cache rows? hit rate on the last scan?
sqlite3 backend/duplicate_detector.db <<'SQL'
SELECT COUNT(*) FROM file_cache;
SELECT cache_hit, COUNT(*) FROM video_files
  WHERE scan_job_id = (SELECT MAX(id) FROM scan_jobs)
  GROUP BY cache_hit;
SQL

# clear only the cache (keeps scan history, deletion logs, duplicates)
sqlite3 backend/duplicate_detector.db "DELETE FROM file_cache;"
```

### GPU sanity check

```bash
nvidia-smi                          # is the GPU visible?
ffmpeg -hwaccels 2>&1 | grep cuda   # is CUDA hwaccel built in?
ffmpeg -decoders 2>&1 | grep cuvid  # are CUVID decoders present?

# in the running backend:
curl http://localhost:9000/api/gpu-status
```

## Common debugging recipes

### "My scan stuck on 'Computing perceptual hashes…'"

Most likely a GPU decode hang. Check:

1. `nvidia-smi` — if there are dead `ffmpeg` processes, kill them.
2. Set `GPU_ENABLED=False` in `config.py` and re-run. If it works, the GPU decode path is the culprit.
3. The hasher has an automatic CPU fallback **per video** if the GPU produces zero frames. But it can't recover from a GPU hang — the timeout is 60s per ffmpeg call.

### "Two clearly identical files aren't matching"

1. Run `diagnose_pair.py` on them.
2. Inspect the printed SAR / rotation / portrait flags. If one is portrait and the other landscape but neither shows `Portrait: True`, rotation metadata may be wrong.
3. Inspect the video similarity %. If close to but below threshold, raise `HASH_SIMILARITY_THRESHOLD` (e.g. 14 → 18).
4. Inspect the audio correlation. If high (>80%) and video low, the pipeline should match by audio. If audio correlation is also low, the files genuinely have different content (or one has been re-encoded with audio replacement / silenced).

### "Pause/stop is slow to respond"

Pause is gated by the batch size — `max_concurrent * 4` files per batch (default 32–48). The pipeline checks signals between batches, so worst-case latency is one batch's worth of work. With slow files (long videos, GPU contention), this can be 20–30s.

To make pause more responsive: lower `MAX_CONCURRENT_FFMPEG` and `GPU_MAX_CONCURRENT` in `config.py`. Smaller batches = more frequent checkpoints, but slightly lower overall throughput.

### "My re-scan didn't get faster"

Phase 1 cache is keyed on `(file_path, file_size, mtime_ns)`. A miss on any of these forces a full re-process. Common reasons for unexpected misses:

1. **The path changed** — `discover_videos` returns `Path.resolve()`-normalized paths. If the user re-scanned through a different mount/drive letter or symlink, the resolved paths differ. Cache rows from the previous mount become orphans (and get swept).
2. **`mtime` changed** — `touch`, file editors that re-write metadata, or tools with `--preserve=timestamps` *off* will bump `mtime_ns`. The bytes might be identical but the cache key isn't.
3. **The file_cache table is empty** — first scan ever, or someone ran `DELETE FROM file_cache`. The next scan rebuilds it.
4. **Schema mismatch** — if the DB exists but predates the Phase 1 schema (no `file_cache_id` on `video_files`), inserts fail at scan time. Delete the DB.

Check the cache hit rate query in [Inspect or clear just the cache](#inspect-or-clear-just-the-cache) to see how the last scan partitioned.

### "WebSocket disconnects randomly"

`useWebSocket` reconnects automatically with a 2s backoff if the scan is non-terminal. The reconnect uses a `useRef` for the latest progress to avoid stale closure bugs. If you see persistent disconnects, check for:

1. **Vite proxy missing `ws: true`** — the WS URL goes to `/api/scan/{id}/ws` and won't be proxied without it.
2. **Reverse proxy in front (nginx, etc.)** — needs `proxy_set_header Upgrade` and `proxy_set_header Connection "upgrade"`.

### "Settings UI changes don't persist"

By design. `PUT /api/settings` modifies the in-memory `settings` instance only. To persist, edit `backend/config.py` or set `.env` values.

## Code style

The codebase mixes a few conventions; match the file you're editing.

### Backend

- **Type hints** on all public functions.
- **Async first** — every IO function has an `async def` wrapper. Sync internals (`_extract_metadata_sync`) stay private and run via `loop.run_in_executor`.
- **Docstrings** are dense and useful — read them before changing internals. The pipeline's `_pipeline_check` and the comparator's best-match algorithm are documented inline.
- **Module-level globals** for caches (`_cached_gpu_info`, `_registry` in `scan_control`). These are intentional and process-scoped.
- **Subprocess calls** always set `creationflags=subprocess.CREATE_NO_WINDOW` on Windows to suppress console pop-ups.

### Frontend

- **Function components** only. No class components.
- **Hooks for shared logic.** No higher-order components.
- **`async/await` over `.then()`** in handlers, except for fire-and-forget `.catch(() => {})` on poll loops.
- **TypeScript strict** — `interface` definitions in `types/index.ts` mirror the Pydantic schemas. Update both together when adding API fields.
- **No CSS-in-JS.** All styles in `App.css` and `index.css`.

## Adding a new pipeline stage

If you add a new stage between, say, hashing and audio fingerprinting:

1. **Create a service function** in `backend/services/your_stage.py`. It should be pure — input dicts, output dicts. No DB access.
2. **Update `run_scan_pipeline`** in `backend/api/scan.py`:
   - Add a `scan.status = "your_stage"` block with WS update.
   - Add a batched `gather()` loop with `_pipeline_check` between batches.
   - Allocate a slice of the `progress_percent` budget (currently 5–45 for metadata, 45–75 for hashing, 75–90 for comparing).
3. **Update the schema** if you need to persist new per-video data — add a column to `VideoFile` in `models/database.py` and **delete the DB** to recreate.
4. **Update the comparator** if your stage produces signals consumed by it.
5. **Update the frontend status string** if needed (hardcoded in `Dashboard.tsx`'s `ACTIVE` array).

Look at how Stage 4b (audio fingerprinting) was added — it's the most recent addition and follows this pattern cleanly.

## Adding a new API endpoint

1. Add the route to the appropriate router in `backend/api/`.
2. If it returns a new shape, add a Pydantic model to `backend/models/schemas.py`.
3. Add the corresponding TypeScript interface to `frontend/src/types/index.ts`.
4. Add the call to `frontend/src/services/api.ts`. **All fetch lives there.**
5. Use the new method from a page or component.

## What to avoid

- **Synchronous file I/O in handlers.** Always use async or push to executors. Blocking the event loop kills scan progress for everyone.
- **Per-file ffprobe in stages 3+.** All metadata you need is already in `_meta_video_info` (stashed in stage 2). Re-probing wastes minutes on big scans.
- **Unbounded concurrency.** Always go through `asyncio.Semaphore(max_concurrent)`. Forgetting this can spawn 10,000+ ffmpeg processes.
- **Persisting from the live `settings` instance.** It's mutated by `PUT /api/settings` and only reflects current state. Use `config.py` defaults or `.env` for permanent changes.
- **New DB columns without resetting.** Schema is created via `Base.metadata.create_all`, then `_migrate_add_columns` auto-runs idempotent `ALTER TABLE ADD COLUMN` for new **nullable** columns on `file_cache` (currently `head_tail_xxh3` and `aggregate_hash`). Anything more involved — renamed columns, type changes, new tables on a non-empty DB with constraints, new indices — still requires either deleting the DB or extending `_migrate_add_columns` with the matching SQL.
