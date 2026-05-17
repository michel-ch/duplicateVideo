# Database

SQLite via `sqlalchemy[asyncio]` + `aiosqlite`. File: `backend/duplicate_detector.db`.

Models are defined in [`backend/models/database.py`](../backend/models/database.py).

## Schema

```
┌──────────────────────────────────────┐
│ scan_jobs                            │
│  id                  PK              │
│  root_path                           │
│  status                              │  queued, pending, scanning, metadata,
│  total_files                         │  hashing, comparing, completed,
│  scanned_files                       │  failed, paused, stopped
│  current_file                        │
│  current_stage                       │
│  progress_percent                    │
│  duplicate_groups_found              │
│  recoverable_space    (bytes)        │
│  started_at                          │
│  completed_at                        │
│  error_message                       │
│  options              (JSON string)  │
└────────────┬─────────────────────────┘
             │ 1:N (cascade delete)
             ▼
┌──────────────────────────────────────┐
│ video_files                          │
│  id                  PK              │
│  scan_job_id         FK→scan_jobs    │
│  file_path           UNIQUE per scan │
│  file_name                           │
│  file_size           (bytes)         │
│  created_at          (mtime)         │
│  modified_at                         │
│  duration            (s)             │
│  width / height                      │
│  bitrate             (bps)           │
│  video_codec / audio_codec           │
│  fps                                 │
│  audio_channels                      │
│  audio_sample_rate                   │
│  perceptual_hashes   (JSON array)    │
│  hash_computed       (bool)          │
│  quality_score       (0-100)         │
│  duplicate_group_id  FK              │ ─┐
│  is_best_quality     (bool)          │  │
│  thumbnail_path                      │  │
│  is_deleted                          │  │
│  deleted_at                          │  │
│  trash_path                          │  │
│  file_cache_id       FK→file_cache   │  │
│  cache_hit           (bool)          │  │
└──────────────────────────────────────┘  │
             ▲                            │
             │ N:1                        │
┌────────────┴─────────────────────────┐  │
│ duplicate_groups                     │◀─┘
│  id                  PK              │
│  scan_job_id         FK→scan_jobs    │
│  similarity_score    (0-100)         │
│  total_wasted_space  (bytes)         │
│  file_count                          │
│  status                              │  pending, resolved (in_queue: legacy)
│  best_file_id        (NOT a real FK) │
│  created_at                          │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│ deletion_logs                        │
│  id                  PK              │
│  original_path                       │
│  trash_path                          │
│  file_size                           │
│  deletion_mode       trash|permanent │
│  deleted_at                          │
│  is_undone                           │
│  undone_at                           │
│  scan_job_id         (no FK)         │
│  duplicate_group_id  (no FK)         │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│ file_cache                           │  Cross-scan cache of pipeline
│  id                  PK              │  outputs, keyed by content identity.
│  file_path                           │
│  file_size                           │
│  mtime_ns            ┐ UNIQUE        │
│  sha256_full         (reserved, opt) │
│  duration / width /  │ stage 2       │
│    height / bitrate /│ output        │
│    codecs / fps /    │ (cached       │
│    audio_* / sar_* / │ metadata)     │
│    rotation          ┘               │
│  perceptual_hashes   (stage 3 cache) │
│  audio_fp            (stage 4b cache)│
│  thumbnail_path                      │
│  head_tail_xxh3      (byte-id fast   │   blake2b of first + last 64 KiB
│                       path key)      │
│  aggregate_hash      (FAISS index    │   per-bit majority over the
│                       key)           │   12 frame pHashes
│  first_seen_at                       │
│  last_seen_at                        │  pruned by end-of-scan sweep
│  cache_version                       │  gates per-output freshness
└──────────────────────────────────────┘
```

### Cache versioning

The `cache_version` integer gates per-output freshness against two version constants in `services/`:

- `PHASH_VERSION` (currently 3) — gates `perceptual_hashes` AND `aggregate_hash`. A row with `cache_version < PHASH_VERSION` is treated as a cache miss for stage 3.
- `AUDIO_FP_VERSION` (currently 2) — gates `audio_fp`. A row with `cache_version < AUDIO_FP_VERSION` is treated as a cache miss for stage 4b.

Writers always `cache_row.cache_version = max(cache_row.cache_version or 1, X)` so the version is monotonically increasing. A single integer can't track per-output versions independently, but in practice the constants only ratchet upward and the gating is "is this row at least as fresh as the writer expects". Metadata (duration, width, codec, etc.) has no version gate — it's stable across code revisions.

Bump a version when the underlying output format/algorithm changes in a way that makes old cached values incomparable with new ones (e.g. new preprocessing filter, hash size change, sampling window change).

## Notable details

- **Auto-migration is limited to nullable column adds.** `init_db()` runs `Base.metadata.create_all()` (creates missing tables) and then `_migrate_add_columns()`, which uses `PRAGMA table_info` to detect missing columns on existing tables and runs `ALTER TABLE … ADD COLUMN` for them. Currently this covers `file_cache.head_tail_xxh3` and `file_cache.aggregate_hash` (both `VARCHAR NULL`). Idempotent across startup re-runs.
- **Any other schema change still requires deleting the DB.** Renamed columns, type changes, dropped columns, new tables on existing schemas (well, `create_all` handles new tables, but new constraints / indices don't auto-migrate either). When in doubt: `rm backend/duplicate_detector.db`.
- **The earlier Phase 1 cache rollout** (`video_files.file_cache_id`, `video_files.cache_hit`, the `file_cache` table itself) did not auto-migrate; users who pre-date that need to delete the DB. New `file_cache` columns added in this revision DO auto-migrate, so users on the Phase 1 schema can update without losing their cache.
- **`scan_jobs.options`** is a JSON string, not a JSON column type. Keep it small; it's just the `ScanOptions` Pydantic dump.
- **`(scan_job_id, file_path)` is unique** on `video_files`. Two scans of the same directory produce two separate `VideoFile` rows for the same file.
- **`best_file_id` is not a real FK.** It's just an integer pointing into `video_files.id`. There's no cascade or constraint.
- **`deletion_logs.scan_job_id` and `duplicate_group_id` are not FKs either.** Deleting a scan doesn't cascade-delete its logs, so deletion history outlives the scans that produced it. This is intentional — undo should still work after the source scan is gone.

## Status workflows

### `scan_jobs.status`

```
queued ──▶ pending ──▶ scanning ──▶ metadata ──▶ hashing ──▶ comparing ──▶ completed
                          │            │            │             │
                          ├──┬─────────┼─────┬──────┼─────┬───────┤
                          ▼  │         ▼     │      ▼     │       ▼
                       paused│      paused   │   paused   │    failed
                          │  └─────► stopped ┴───────────────────►
                          └──► (resume returns to scanning)
```

The pipeline transitions the status when starting each stage. The `paused` and `stopped` transitions can happen from any active state via `_pipeline_check`. `failed` is reached only from the outer `except Exception` block.

**Crash recovery.** If the server process dies while a scan is still in an active status (Ctrl-C, OOM, power loss), the row is left looking alive forever. On next startup, `main.py:_recover_orphaned_scans()` (runs in the FastAPI lifespan after `init_db()`) finds every `ScanJob` row whose status is in `{pending, scanning, metadata, hashing, comparing, paused}`, flips it to `stopped`, sets `completed_at = now`, and writes `current_stage = "Server restarted while scan was in progress"`. This unblocks `_has_active_scan()` so any queued scans can start, and prevents the dashboard from showing a phantom spinner.

The `FileCache` rows themselves are unaffected — every output committed before the crash (per-batch in stages 2/3/4b, per-chunk in the head/tail backfill, every 256 rows in the aggregate-hash compute) survives, so re-running the scan picks up exactly where the prior one left off.

### `duplicate_groups.status`

```
pending ──▶ resolved
```

- `pending` — created by the scan, untouched by the user.
- `resolved` — user has acted on the group (set by `POST /duplicates/{id}/resolve` and `POST /auto-clean` after a successful deletion). Both endpoints delete files synchronously, so `resolved` means the action is complete.
- `in_queue` — legacy state from an earlier "stage files for batch deletion" workflow that was never wired up. New rows do not enter this state, but old rows may still carry it; the frontend renders a yellow "In Queue" badge if encountered.

## Connection management

Two connection patterns:

```python
# Request handlers — FastAPI dependency
async def get_db():
    async with async_session() as session:
        yield session

@router.get("/duplicates")
async def list_duplicates(db: AsyncSession = Depends(get_db)): ...
```

```python
# Background tasks — manual session
async def run_scan_pipeline(scan_id, root_path, options):
    async with async_session() as db:
        ...
```

Background tasks must use `async_session` directly because they have no request scope. The dependency-injected `get_db()` only works inside endpoint handlers.

## Performance notes

- The `selectinload(DuplicateGroup.videos)` in `api/duplicates.py:list_duplicates` is critical — without it, paginating 20 groups becomes 21 queries (1 + N).
- The `aiosqlite` async driver is much slower than the synchronous `sqlite3`, but it's required so DB calls don't block the event loop. For batched scan-time updates, the pipeline `commit`s once per batch, not per row. The aggregate-hash compute step (pure-Python loop, no `await`s) additionally flushes every 256 rows for crash safety.
- SQLite's WAL mode is **not** enabled. Single-process app, so it's not needed. If you ever multi-process this, set `PRAGMA journal_mode=WAL`.

## Inspecting the DB

```bash
# CLI
sqlite3 backend/duplicate_detector.db

# Common queries
.schema video_files
SELECT status, COUNT(*) FROM scan_jobs GROUP BY status;
SELECT id, similarity_score, file_count FROM duplicate_groups ORDER BY total_wasted_space DESC LIMIT 10;
```

## Resetting

To wipe everything and start fresh:

```bash
rm backend/duplicate_detector.db
rm -rf backend/thumbnails/*
```

The DB is recreated on next startup. The thumbnails dir is recreated on the next scan.
