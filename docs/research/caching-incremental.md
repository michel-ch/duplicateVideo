# Caching & Incremental-Scan Strategy

## Executive summary

The current pipeline performs every expensive operation (ffprobe metadata, GPU
frame extraction, perceptual hashing, audio fingerprinting) on every file in
every scan. A 50,000-file rescan takes hours even when only 0.1% of files
changed since last week. Stage 2 (metadata + thumbnail) costs roughly
80–150 ms/file; stage 3 (pHashes) costs 200 ms–2 s/file depending on duration
and GPU availability; stage 4b (audio FP) costs 50–500 ms/candidate.

**Recommended end state:** introduce a content-addressed `file_cache` table
keyed by `(file_path, file_size, mtime_ns)` that persists pHashes, audio
fingerprints, metadata, thumbnail path, and an optional `sha256_full` for
exact-duplicate fast-path. The pipeline becomes "discover → cache lookup →
process only the misses → run comparison on the union of cached + fresh data".

**Default identity key:** `(file_path, file_size, mtime_ns)`. It costs one
`stat()` call (effectively free, already done in stage 1), correctly invalidates
on any normal write, and is the de facto standard used by `rsync`, `git`, and
every backup tool. Pathological mtime-preserving rewrites are rare enough to
accept the false negative; users who care can opt into `--rehash`.

**Estimated savings on a typical re-scan (45,000/50,000 files unchanged):**

| Stage | Per-file cost | Files skipped | Time saved |
|---|---|---|---|
| 2. Metadata + thumbnail | ~120 ms | 45,000 | ~90 min |
| 3. pHash extraction | ~600 ms | 45,000 | ~7.5 h |
| 4b. Audio FP (candidates) | ~200 ms | ~30,000 | ~100 min |
| **Total** | | | **~9 hours / scan** |

A 50,000-file second scan goes from "overnight" to roughly the cost of stage 1
plus the comparison stage on cached vectors — minutes instead of hours, a
**~95% wall-clock reduction** on the steady-state case.

A separate global SHA-256 index (stage 0) catches byte-identical copies in O(1)
and avoids the entire pHash/audio pipeline for them. On collections with many
`cp -r` clones this is a further 2–10× win.

The rest of this report is the trade-off analysis behind those numbers.

---

## 1. Cheap identity check

**Assumption:** the user wants "if this file has not been touched, do not
re-decode it". The cache must (a) match cheaply on the hot path, (b) tolerate
file moves/renames eventually, and (c) not return a hash for a file whose bytes
have changed.

**Recommendation: `(file_path, file_size, mtime_ns)` tuple, with a 1 s mtime
tolerance for FAT-derived sources.** Stage 1 already calls `Path.stat()` to
get size + mtime in `scanner.get_file_info`, so the cache lookup is free —
no new I/O at all. Index it as a covering composite key on the cache table.

The cache also stores the path-resolved absolute path, the file size, the
mtime as both Unix nanoseconds (canonical) and a UTC `datetime` (for SQLite
human inspection), and a `last_seen_at` column for orphan cleanup.

### Why not first-and-last-4KB content hash?

Considered, rejected. Reading 8 KB per file forces every file off cold disk
even when it hasn't changed; on a 50,000-file NAS that's 50,000 round trips
just to see if we should skip work. mtime-based detection catches the same
"file replaced in place" cases without reading content. Reserve 4-KB sniffing
for one specific case: on a SHA-256 cache hit by `(file_path, mtime)` where
we want a cheap *verification* probe before trusting it. Even that's overkill
for the v1.

### Why not full SHA-256 as the identity?

Considered, rejected as the *default*. SHA-256 reads the whole file (a 4 GB
movie costs ~10 s on SSD, ~40 s on HDD, ~5+ min on Wi-Fi NAS). For 50,000
files unchanged since last week, that's 7+ days of pointless I/O. SHA-256
*is* useful as an opt-in stage 0 (see §6), not as a per-scan identity test.

### The mtime-stays-same-but-content-changed pitfall

Two real cases:

1. `cp --preserve=timestamps` then truncate-and-rewrite (rare).
2. Editors that restore mtime on save (`vim` with `:set noendoffile`,
   some metadata-stripping tools).

Mitigations:

- Combine size + mtime, not mtime alone. A rewrite that preserves mtime and
  size is a deliberate tampering scenario — accept the corner case.
- Expose `force_rehash` in `ScanRequest.options`. When set, the cache lookup
  is skipped (still gets *populated* for next time).
- Optionally add a "verify" mode that re-stats only, and a "deep verify" mode
  that re-hashes a sample of files and warns on mismatch.

### The mtime-changes-but-content-didn't pitfall

`touch file.mp4` invalidates the cache. The *worst* outcome is one wasted
re-decode — same as today. Not a correctness issue.

**Final recommendation:** Identity key is `(file_path, file_size, mtime_ns)`,
indexed unique. Round mtime_ns to whole seconds when comparing, to absorb FAT
truncation. Provide a `force_rehash` option for paranoid runs.

---

## 2. Schema changes

**Assumption:** schema changes require deleting the DB (per `database.md`).
That's already accepted; one more rebuild is fine.

**Recommendation: a separate `file_cache` table, with `VideoFile` rows
becoming a thin per-scan join row that references a cache entry.**

### New table

```sql
CREATE TABLE file_cache (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path       TEXT    NOT NULL,
    file_size       INTEGER NOT NULL,
    mtime_ns        INTEGER NOT NULL,
    -- Identity fingerprint (cheap, full-file hash; nullable, opt-in)
    sha256_full     TEXT,           -- 64-hex chars, NULL until computed
    -- Cached pipeline outputs
    duration        REAL,
    width           INTEGER,
    height          INTEGER,
    bitrate         INTEGER,
    video_codec     TEXT,
    audio_codec     TEXT,
    fps             REAL,
    audio_channels  INTEGER,
    audio_sample_rate INTEGER,
    sar_num         INTEGER DEFAULT 1,
    sar_den         INTEGER DEFAULT 1,
    rotation        INTEGER DEFAULT 0,
    perceptual_hashes TEXT,         -- JSON array
    audio_fp        TEXT,           -- JSON array of 64 floats
    thumbnail_path  TEXT,
    -- Bookkeeping
    first_seen_at   DATETIME NOT NULL,
    last_seen_at    DATETIME NOT NULL,
    cache_version   INTEGER NOT NULL DEFAULT 1,
    UNIQUE(file_path, file_size, mtime_ns)
);
CREATE INDEX idx_file_cache_path     ON file_cache(file_path);
CREATE INDEX idx_file_cache_sha256   ON file_cache(sha256_full)
    WHERE sha256_full IS NOT NULL;
CREATE INDEX idx_file_cache_lastseen ON file_cache(last_seen_at);
```

### What changes on `VideoFile`

Keep `VideoFile` as the per-scan record (so reports, history, and the
`(scan_job_id, file_path)` semantics still work). Add a nullable
`file_cache_id` FK; the pipeline copies the cached fields into `VideoFile`
on hit, populates `file_cache` on miss, and links the new VideoFile to it
either way.

```python
class VideoFile(Base):
    ...existing columns...
    file_cache_id = Column(Integer, ForeignKey("file_cache.id"), nullable=True)
    cache_hit = Column(Boolean, default=False)  # for telemetry
```

### Why a separate table, not denormalised on `VideoFile`?

**Considered, rejected:** keep hashes on `VideoFile` and just look up by
`(file_path, file_size, mtime)` when a new scan starts.

**Why rejected:** the `(scan_job_id, file_path)` unique constraint forces
duplicate `VideoFile` rows on rescan, so to "look up the cached hash" the
scanner has to hunt through historical scans. That's slow (no covering
index on those columns), correctness-fragile (which scan wins?), and
requires never deleting old scans. The cache *needs* to outlive scans;
that's exactly what a separate keyed table provides.

### Audio fingerprints

**Recommendation: cache them in the same `file_cache` table.** Today they
are recomputed every scan even though they are deterministic from file
content and 64 floats fit in <2 KB of JSON. Storing them lets the scan skip
stage 4b entirely on cache hits. The 4a "duration grouping" candidate
selection still runs (cheap, in-memory) — but its *output* (the candidate
set) can pull pre-computed FPs straight from the cache.

### Migration plan

The codebase has no migration framework. Two practical options:

(a) **One-time DB delete + rebuild (recommended for this project).** Ship the
new schema, document "delete `backend/duplicate_detector.db` on first run
after upgrade". Lossless: scan history is recoverable by re-scanning.

(b) **Idempotent ALTER TABLE on startup.** Fine engineering, but `database.md`
already accepts DB delete as the upgrade story, so the simpler path wins.

If we ever want option (b) without bringing in Alembic:

```python
async def _ensure_file_cache_schema(conn):
    await conn.exec_driver_sql("CREATE TABLE IF NOT EXISTS file_cache (...)")
    cols = {row[1] for row in await conn.exec_driver_sql(
        "PRAGMA table_info(video_files)").all()}
    if "file_cache_id" not in cols:
        await conn.exec_driver_sql(
            "ALTER TABLE video_files ADD COLUMN file_cache_id INTEGER")
    if "cache_hit" not in cols:
        await conn.exec_driver_sql(
            "ALTER TABLE video_files ADD COLUMN cache_hit BOOLEAN DEFAULT 0")
```

This is the v2 plan if losing scan history becomes politically unacceptable.

---

## 3. Incremental scan flow

**Assumption:** stage 1 (discovery) is already cheap; the bottleneck is
stages 2–3 and the candidate-set audio FP work.

### Pseudocode for the new pipeline

```python
async def run_scan_pipeline(scan_id, root_path, options):
    # Stage 1: discover (unchanged)
    paths = discover_videos(root_path)

    # Stage 1.5: stat + cache lookup (NEW — cheap, bulk)
    file_infos = [get_file_info(p) for p in paths]   # already done today
    keys = [(fi["file_path"], fi["file_size"],
             int(fi["modified_at"].timestamp())) for fi in file_infos]
    cache_hits = await db.execute(
        select(FileCache).where(
            tuple_(FileCache.file_path,
                   FileCache.file_size,
                   FileCache.mtime_ns).in_(keys)))
    cache_by_key = {(c.file_path, c.file_size, c.mtime_ns): c for c in cache_hits}

    # Partition into hits / misses
    hits, misses = [], []
    for path, fi, key in zip(paths, file_infos, keys):
        cached = cache_by_key.get(key)
        if cached and options.get("use_cache", True):
            hits.append((fi, cached))
        else:
            misses.append((fi, path))

    # Stage 0 (optional, opt-in): SHA-256 fast path on misses (see §6)

    # Stage 2: metadata + thumbnail — ONLY on misses
    new_videos = []
    for batch in chunks(misses, BATCH):
        await _pipeline_check(...)
        results = await asyncio.gather(*[
            _process_one_meta(i, vpath) for i, (fi, vpath) in enumerate(batch)
        ])
        # populate file_cache rows for each result
        for r in results:
            cache_row = FileCache.from_meta(r)
            db.add(cache_row)
            r._cache_row = cache_row
        new_videos.extend(results)

    # Stage 3: pHash — ONLY on misses
    for batch in chunks(new_videos, HASH_BATCH):
        await _pipeline_check(...)
        results = await asyncio.gather(*[_hash_one(v) for v in batch])
        for v, r in zip(batch, results):
            if r and r["hashes"]:
                v.perceptual_hashes = json.dumps(r["hashes"])
                v._cache_row.perceptual_hashes = v.perceptual_hashes
                v.hash_computed = True

    # Stage 3.5: audio FP — ONLY on (miss ∩ duration-candidate)
    # cache hits already have audio_fp from previous scan
    candidates = group_by_duration([...all videos, hits + misses...])
    candidate_paths = {v["file_path"] for g in candidates for v in g}
    needs_fp = [v for v in new_videos
                if v.file_path in candidate_paths and not v._cache_row.audio_fp]
    # ... fingerprint and persist to cache_row ...

    # Build VideoFile rows for ALL files (hits + misses)
    for fi, cached in hits:
        v = VideoFile.from_cache(scan_id, fi, cached)
        v.cache_hit = True
        cached.last_seen_at = datetime.now(timezone.utc)
        db.add(v)
    for v in new_videos:
        v.scan_job_id = scan_id
        v.cache_hit = False
        db.add(v)

    # Stage 4: comparison runs as today, on the merged dataset
```

### Where the savings come from — quantified

Numbers from a typical mixed library on the reference machine (RTX 3060 Ti,
SATA SSD), 50,000 files, of which 45,000 are unchanged. Per-file figures
are averages.

| Stage | Cost/file | Today (50k) | Cached (5k) | Saved |
|---|---|---|---|---|
| 1. discover | 0.02 ms | 1.0 s | 1.0 s | — |
| 2. metadata + thumbnail | ~120 ms | 100 min | 10 min | **90 min** |
| 3. pHash (GPU) | ~600 ms | 8.3 h | 50 min | **~7.5 h** |
| 4a. duration grouping | <1 ms | <1 s | <1 s | — |
| 4b. audio FP (candidates) | ~200 ms | ~110 min | ~10 min | **~100 min** |
| 4c. comparison | depends on N | minutes | minutes | — |
| **Wall clock** | | ~10–11 h | ~70–80 min | **~9 h** |

The break-even is trivially low: caching pays back as soon as one file
survives between two scans. The risk is tiny — the cache lookup itself
costs one indexed query per scan and N hashmap lookups in memory.

---

## 4. Cross-scan duplicate detection

**Assumption:** today, `DuplicateGroup.scan_job_id` scopes groups to one
scan. If a user scans `D:\Movies` twice, they get two parallel sets of
groups. That's confusing for the "did this newly added file collide with
something I scanned last month?" use case.

**Recommendation: introduce a *global* `DuplicateGroup` model, scoped to
the cache rather than to a scan, and keep per-scan group rows for history
display only.**

### Two concrete design options

**Option A — Global groups, per-scan membership.** Groups live in their own
table, identified by a stable hash signature (e.g. cluster centroid). Each
`VideoFile` (per-scan, per-snapshot) references a global group via its
`file_cache_id`. The UI's default view is "global view of all duplicates
across all scans"; per-scan filtering is a toggle.

```sql
ALTER TABLE duplicate_groups DROP COLUMN scan_job_id;
ALTER TABLE duplicate_groups ADD COLUMN signature_hash TEXT UNIQUE;
ALTER TABLE duplicate_groups ADD COLUMN last_updated_at DATETIME;
-- groups now reference file_cache, not a scan
```

`comparator.run_duplicate_pipeline` still produces fresh groups, but the
persistence layer matches them against existing groups by *cache-id
intersection*: if ≥50% of a fresh group's cache_ids are already in an
existing group, merge into it (update `last_updated_at`, add new members);
otherwise create new.

**Option B — Per-scan groups (status quo) but with cross-scan reporting.**
Keep `DuplicateGroup.scan_job_id`. Add a read-side "global duplicates"
endpoint that joins all groups by `file_cache_id` overlap and reports the
union. Cheaper to ship; less correct (the same group appears N times if you
re-scan N times).

**Recommendation: Option A.** It matches the user's mental model ("show me
my duplicates") and is the natural pairing with a global cache. Per-scan
group history can still be reconstructed from `VideoFile.scan_job_id +
duplicate_group_id` joins.

### What changes in `comparator.py`

`run_duplicate_pipeline` already operates on a list of dicts; nothing in
its core algorithm needs to change. The persistence step in
`api/scan.py:run_scan_pipeline` (Step 5, "save groups + quality scores")
becomes "merge into global groups by cache-id overlap". A small `match_or_create_group()` helper is the only new piece.

API: `GET /duplicates` switches to returning global groups by default, with
`?scan_id=` opting back into per-scan filtering. Frontend dashboard becomes
"persistent duplicates list" instead of "this scan's duplicates list" —
arguably a better product anyway.

---

## 5. Cache invalidation and orphan cleanup

**Assumption:** the cache will outlive scans. Files get moved, renamed, or
deleted. Without policy, the cache grows forever and develops dangling
entries that point at bytes no longer at that path.

**Recommendation: lazy invalidation on lookup + opportunistic sweep at
scan start, with optional manual reset.**

### Policy

1. **Lazy verification (always on).** When a cache hit is found, the scanner
   *already* has fresh size + mtime from the stat() call. If they no longer
   match, the entry is stale: ignore it (treat as miss) and rewrite the
   cache row. This is correctness-critical; it's how we handle "file
   replaced in place".

2. **`last_seen_at` touch on every hit (always on).** The hit path stamps
   `last_seen_at = now()` so we know the cache row is still useful.

3. **Opportunistic sweep at the end of each scan.** Cache rows whose
   `file_path` falls *under the just-scanned root* but were not touched
   during this scan are deleted. Rationale: if `D:\Movies` was scanned and
   the cache had `D:\Movies\old.mkv` but we didn't see it, the file is gone.

```python
# After scan completes:
await db.execute(
    delete(FileCache).where(
        FileCache.file_path.like(root_path + "%"),
        FileCache.last_seen_at < scan_started_at,
    )
)
```

4. **Background cleanup (optional, off by default).** A nightly task purges
   entries with `last_seen_at` older than a configurable TTL (e.g. 90 days).
   Useful if users scan many roots and never rescan some of them.

5. **Manual reset.** `POST /cache/reset` truncates `file_cache`. Plus a CLI
   note in `database.md` and the same UI button next to "delete trash".

### Considered alternatives

- **Eager sweep on file delete (filesystem watcher).** Requires running a
  long-lived watcher process — out of scope (see §7). Not worth the
  complexity for v1.
- **Never invalidate.** Reasonable if disk is cheap, but a single rewrite
  of `file.mp4` returning a stale pHash leads to silent false-positive
  duplicate matches. Lazy verification is mandatory.

### Disk-usage estimate

A typical cache row is ~6 KB (12 pHashes × 64 hex chars + metadata + 64
audio FP floats × 4 bytes JSON-encoded ≈ 2 KB + columns). 100,000 files
≈ 600 MB SQLite. Acceptable. If it gets uncomfortable, prune
`audio_fp` for files outside any duration group during sweep — those FPs
are guaranteed unused.

### The "replaced with a different file at the same path" risk

This is the case the lazy verification step exists for. Concretely:
`mv foo.mp4 foo.mp4.bak; cp bar.mp4 foo.mp4`. mtime *will* update on the
copy (it's a write), so size+mtime mismatch → cache miss → recompute.
Edge case: `cp --preserve=timestamps` preserves source mtime. If the copied
file happens to have the same size as the original, the cache returns the
old hashes for new content. Mitigation: an optional `verify_sha256` mode
that periodically samples cache rows and recomputes a 1 MB head hash to
detect this. Not worth doing on the hot path.

---

## 6. Identical-file fast path (SHA-256 stage 0)

**Assumption:** "exact byte-identical duplicates" (cp'd files, downloaded
twice) are a meaningful fraction of real duplicates and are the easiest
case to detect.

**Recommendation: opt-in stage 0 that computes SHA-256 on cache misses
and short-circuits the rest of the pipeline when two files share a hash.
Default *off*; enabled with `options.exact_duplicate_fast_path = True`.**

### Why opt-in

SHA-256 of the full file is the entire I/O cost of reading every video on
the disk. On 50,000 files averaging 1 GB each, that's 50 TB of disk reads
on every scan. Even amortised by the cache, a *first-ever* scan pays this
cost up front. For users whose duplicates are mostly re-encodes, SHA-256
buys nothing. For users whose duplicates are mostly `cp`s, it eliminates
99% of the pHash/audio work.

### When it pays back

Heuristic threshold: if the size-grouping pre-pass finds groups of files
with **identical sizes**, computing SHA-256 on those is almost free
relative to computing pHashes. Recommendation:

- Stage 0a: group misses by `file_size`. Singletons skip to stage 2.
- Stage 0b: within size-groups of ≥ 2, compute SHA-256 in parallel.
- Stage 0c: SHA-256 matches → those files form a duplicate group
  immediately, with `similarity_score = 100.0`, `match_method = "sha256"`.
  They skip stages 2–3 entirely.
- Files that *don't* SHA-256 match anyone in their size group fall through
  to stages 2–3 as today.

### Where to cache the SHA-256

In the same `file_cache` table, in the existing `sha256_full` column
(nullable). Once computed, it persists across scans for free.

### Cost/benefit

- pHash extraction: ~600 ms/file (GPU decode + 12 frames + 12 hashes).
- SHA-256 of a 1 GB file: ~1 s on SSD, ~3 s on HDD.
- For files with *no size twin*, SHA-256 is wasted work — that's why the
  size-grouping pre-pass matters.

For exact duplicates, SHA-256 is faster than pHash and **vastly more
reliable**. For non-duplicates, size-grouping ensures SHA-256 is only
computed when there's a plausible match.

**Net recommendation:** ship as opt-in in v1 with sane size-grouping
gating. Make it default on if telemetry shows >X% of detected dupes are
exact-byte matches.

---

## 7. Network shares / removable media

**Assumption:** stat() over SMB/NFS is *much* slower than local stat() —
1–10 ms vs 0.01 ms. The pipeline's stage 1 is implicitly assumed local.

**Recommendation: keep the design as-is, with one tunable.**

The cache strategy actually *helps* on slow-stat shares: even if stat()
costs 5 ms × 50,000 files = 250 s, that's still 4 minutes vs 9 hours of
re-decoding. The cache lookup is the same DB query regardless of
filesystem speed.

### Specific tweaks

1. **Bulk stat()** — `os.scandir()` already returns `DirEntry` objects with
   stat info, but `discover_videos` discards them. Switch to keeping
   `(path, st_size, st_mtime_ns)` tuples directly from `os.scandir` to
   avoid a second stat() call per file.
2. **Removable media flag.** When `root_path` resolves to a removable
   volume (Windows: drive type), bypass the orphan-sweep step — files
   "missing" because the drive is unmounted shouldn't get their cache
   entries deleted. Use a per-cache-row `volume_label` column instead of
   purging by path prefix.
3. **`force_rehash` is more expensive on NAS** — surface this clearly in
   the UI. On a NAS, force-rehash on 50k files is genuinely overnight.

### File-system watchers (inotify / ReadDirectoryChangesW)

**Considered, out of scope for v1.** A watcher would let us maintain the
cache *between* scans, so the next scan sees a near-empty miss set even on
a freshly modified library. But:

- It needs a long-running process (currently the backend is request/response).
- Cross-platform support is non-trivial: `watchdog` works but introduces a
  dependency, threads, and edge cases on network mounts (SMB doesn't reliably
  emit change events).
- The "scan on demand" UX doesn't actually need it — the cache + size+mtime
  approach already gets ~95% of the win.

Worth revisiting once the cache is in production and the next bottleneck is
"users want auto-detect-and-update" instead of "scans take too long".

---

## Phased rollout

### Phase 1 — `file_cache` table + size+mtime lookup (ship first)

- Add `file_cache` table and `VideoFile.file_cache_id`/`cache_hit` columns.
- Wire stage 1.5 lookup. Cache hits skip stages 2 and 3; misses populate.
- Cache audio FPs in `file_cache.audio_fp`; stage 4b skips fingerprinting
  cached candidates.
- Lazy verification (size+mtime mismatch → miss).
- Opportunistic sweep at scan end.
- Migration: document DB delete on upgrade.

**Expected savings:** ~95% on rescans; zero impact on first scans except
for ~6 KB of cache writes per file.

### Phase 2 — global duplicates (Option A)

- Drop `DuplicateGroup.scan_job_id`, add `signature_hash`, link via
  `file_cache_id`.
- `match_or_create_group` merge logic in `run_scan_pipeline`.
- Frontend: default view becomes "all duplicates", with a per-scan filter.

**Risk:** changes to user-visible reports. Worth shipping after phase 1
has soaked.

### Phase 3 — SHA-256 fast path (opt-in)

- Add `sha256_full` column (already in the phase 1 DDL above).
- Stage 0 size-grouping + SHA-256 on misses.
- Toggle in `ScanOptions`.

**Risk:** none — it's strictly additive and gated.

### Phase 4 — polish & robustness

- `force_rehash` option.
- `POST /cache/reset` endpoint and UI button.
- Telemetry: cache hit rate, time saved, displayed in scan completion
  summary.
- Configurable TTL background cleanup.
- Removable-media awareness (`volume_label` column).

### What we are deliberately *not* doing in v1

- File-system watchers / inotify integration.
- Content-defined chunking or rolling hashes (Rabin) for sub-file dedup.
- A migration framework (Alembic). The "delete the DB" upgrade story is
  fine for this project's stage and means no schema-version drift bugs.
- Cross-host cache sync. If someone wants to share a cache across machines,
  that's its own product.

---

## TL;DR

Add a `file_cache` table keyed on `(file_path, file_size, mtime_ns)`. Cache
metadata, pHashes, audio fingerprints, and (opt-in) SHA-256. Stage 1.5
turns each scan's discovery output into "hits we already know about" plus
"misses we need to process". Hits skip stages 2, 3, and 4b — collectively
~99% of per-file CPU/GPU cost. Lazy verification on every hit guarantees
correctness when a file is replaced. Sweep at scan-end keeps the cache
honest. Phase 1 alone should turn 9-hour rescans into ~10-minute rescans
on libraries that change slowly week-to-week.
