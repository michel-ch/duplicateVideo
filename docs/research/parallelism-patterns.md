# Concurrency & Parallelism Patterns for the Duplicate-Video Scanner

A research note focused specifically on **concurrency primitives, scheduling
patterns, and process/thread topology** for the FastAPI scan pipeline in
`backend/api/scan.py`. The companion notes
[`pipeline-optimizations.md`](pipeline-optimizations.md),
[`algorithmic-improvements.md`](algorithmic-improvements.md), and
[`caching-incremental.md`](caching-incremental.md) cover the *what* (algorithmic
shortcuts) and *what data* (caching). This note covers the *how* — how should
the runtime actually schedule the work?

The current pipeline serialises CPU-bound and GPU-bound stages behind a single
semaphore, batches each stage in a synchronous "fan-out then await-all" pattern,
and spawns a fresh ffmpeg subprocess per file. On Windows, where
`CreateProcess` is roughly 5–10× slower than Linux `vfork`+`exec`, this is the
dominant per-file overhead for short videos.

---

## Executive summary

Top three concurrency-pattern changes, in order of impact-per-effort:

| # | Change | Where | Expected wall-clock win |
|---|--------|-------|--------------------------|
| 1 | **Stream pipeline with bounded `asyncio.Queue`s between stages** instead of "barrier-after-stage" batching. Stage 2/3/4b run as continuous worker pools draining their input queue. | `backend/api/scan.py:264–576` | 20–35% on mixed CPU/GPU work; up to 50% when stage 3 (GPU) and 4b (CPU) run in true parallel |
| 2 | **Persistent ffprobe + ffmpeg processes** with `-progress pipe:` + `-listen` patterns OR pre-warmed subprocess pool, especially on Windows where each `CreateProcess` is 20–50 ms of pure syscall overhead | `backend/services/metadata.py`, `backend/services/hasher.py`, `backend/services/audio_fingerprint.py` | 15–30% for short files (≤30s); negligible for long files |
| 3 | **`asyncio.TaskGroup` + dedicated `ProcessPoolExecutor` for pHash compute** so `imagehash.phash()` runs across cores without GIL contention, and one task failure cancels its siblings cleanly (no leaked ffmpeg children) | `backend/services/hasher.py:259–326`, `backend/api/scan.py:415–443` | 10–20% on CPU-bound pHash; eliminates a class of subprocess leak bugs |

These three sit *on top of* the work proposed in `pipeline-optimizations.md`
(separate GPU/CPU semaphores, overlapping stages 3 & 4b); they replace the
"how do we run a batch" mechanism, not the "what work does the batch do"
mechanism.

The biggest implementation risk is #1: it requires re-thinking pause/stop
semantics across multiple long-lived worker tasks. The biggest *theoretical*
win sitting unexploited today is finding #5 (NVDEC saturation): Ampere/Ada
consumer GPUs only enable **2 NVDEC engines**, and per-engine throughput
plateaus at ~3–4 concurrent streams. Running 12 concurrent CUVID jobs (current
`GPU_MAX_CONCURRENT`) is roughly **3× past the knee** — you're paying CPU
context-switch overhead for no incremental GPU work.

---

## 1. Current concurrency architecture (sketch)

```
                ┌─────────────────────────────────────────────────┐
                │   run_scan_pipeline()  (single async coroutine) │
                │   running in FastAPI BackgroundTasks            │
                └─────────────────────────────────────────────────┘
                                       │
   ┌───────────────────────────────────┼───────────────────────────────────┐
   │                                   ▼                                   │
   │   Stage 1   discover_videos()  (sync)                                 │
   │                                   │                                   │
   │   Stage 1.5 cache-lookup partition (sync sqlalchemy IN clauses)       │
   │                                   │                                   │
   │   Stage 2   metadata + thumbnail  ─── batches of max_concurrent*4 ────│
   │             ┌──────────────┐                                          │
   │             │ asyncio.gather(*[_process_one_miss(...)]) │   ◀── BARRIER
   │             │ gated by  sem = Semaphore(max_concurrent) │             │
   │             └──────────────┘                                          │
   │                                   │                                   │
   │   Stage 3   perceptual hashing    ─── batches of max_concurrent*4 ────│
   │             ┌──────────────┐                                          │
   │             │ asyncio.gather(*[_hash_one(...)])         │   ◀── BARRIER
   │             │ same sem (Semaphore(12 on GPU, 8 on CPU)) │             │
   │             └──────────────┘                                          │
   │                                   │                                   │
   │   Stage 4a  group_by_duration  → candidate set  (sync, fast)          │
   │                                   │                                   │
   │   Stage 4b  audio fingerprints    ─── batches of max_concurrent*4 ────│
   │             ┌──────────────┐                                          │
   │             │ asyncio.gather(*[_fp_one_v(...)])         │   ◀── BARRIER
   │             │ audio_sem = Semaphore(max_concurrent)     │             │
   │             └──────────────┘                                          │
   │                                   │                                   │
   │   Stage 5   run_duplicate_pipeline()                                  │
   │             (synchronous Python: duration sort, union-find, all-pairs)│
   │                                   │                                   │
   │   Stage 6   rank_group + DB persist  (sync)                           │
   └───────────────────────────────────┼───────────────────────────────────┘
                                       ▼
                              ScanJob.status="completed"

Per-file subprocess fan-out:
  Stage 2 metadata  → 1× ffprobe   (Popen + wait)
  Stage 2 thumbnail → 1× ffmpeg    (Popen + wait)
  Stage 3 hashing   → 1× ffmpeg    (Popen + wait, GPU-accel when available)
  Stage 4b audio FP → 1× ffmpeg    (Popen + wait, full audio decode)
  ⇒  3–4 subprocess spawns per file × N files
```

### Properties of this architecture (not all bad)

- **Simple to reason about**: linear stages, one barrier between each, all
  errors caught by the per-stage `try/except`. `_pipeline_check` between
  batches makes pause/stop responsive within ~one batch (~32–48 files).
- **One control flow** — easy to thread pause/stop, easy to write progress
  updates in monotonic order.
- **Cache-friendly for the DB** — each stage commits in one transaction at the
  end (mostly).

### Where it loses

1. **`gather` barrier at end of every batch.** The slowest task in a batch
   gates the next batch's start. With 48 files in flight and one being a
   2-hour Blu-ray rip, 47 cores idle while one ffmpeg seek runs.
2. **Stage barriers are total**: stage 3 can't start hashing file A until
   stage 2 has finished file Z, even though file A's metadata may be ready in
   ms.
3. **Single shared semaphore for GPU/CPU stages.** A semaphore counts slots,
   not what they're costing. 12 GPU-stream slots are way more than NVDEC can
   actually use; 12 audio-FP slots are way more than CPU can sustainably feed.
4. **Process-per-file overhead.** On Windows, `CreateProcess` takes 20–50 ms
   regardless of the actual decode work. For a 5-second clip that decodes in
   ~80 ms, you're paying 25–50% overhead on spawn alone.
5. **`asyncio.gather(*tasks, return_exceptions=True)`** does *not* cancel
   sibling tasks if one fails — and since stop signals are checked via flags
   rather than `task.cancel()`, leaked ffmpeg subprocesses on failure are a
   real possibility (see finding 7).

---

## 2. Proposed concurrency architecture (sketch)

```
                   ┌──────────────────────────────────────────┐
                   │   run_scan_pipeline()                    │
                   │   async with asyncio.TaskGroup() as tg:  │
                   └──────────────────────────────────────────┘
                                       │
                                       ▼
   ╔══════════════════════════ Pipeline graph ═══════════════════════════╗
   ║                                                                     ║
   ║   discover  ──────────────────────┐                                 ║
   ║                                   │                                 ║
   ║                                   ▼                                 ║
   ║                          ┌────────────────┐                         ║
   ║                          │ cache_lookup_q │  (maxsize=200)          ║
   ║                          │  asyncio.Queue │                         ║
   ║                          └────────────────┘                         ║
   ║                                   │                                 ║
   ║                       ┌───────────┴───────────┐                     ║
   ║                       ▼                       ▼                     ║
   ║             ┌──────────────────┐    ┌──────────────────┐            ║
   ║             │ HIT path:        │    │ MISS path:       │            ║
   ║             │ build VideoFile  │    │ feed metadata_q  │            ║
   ║             │ straight to      │    └──────────────────┘            ║
   ║             │ comparator_in_q  │            │                       ║
   ║             └──────────────────┘            ▼                       ║
   ║                       │            ┌────────────────┐               ║
   ║                       │            │ metadata_q     │  maxsize=64   ║
   ║                       │            └────────────────┘               ║
   ║                       │                    │                        ║
   ║                       │            CPU pool: 4–8 metadata workers   ║
   ║                       │            (ffprobe + thumbnail subprocess) ║
   ║                       │                    │                        ║
   ║                       │                    ▼                        ║
   ║                       │            ┌────────────────┐               ║
   ║                       │            │ hash_q         │  maxsize=32   ║
   ║                       │            └────────────────┘               ║
   ║                       │              │            │                 ║
   ║                       │              ▼            ▼                 ║
   ║                       │      GPU pool: 3-4    CPU pool: 4-8         ║
   ║                       │      pHash workers    audio-FP workers      ║
   ║                       │      (concurrent      (separate semaphore)  ║
   ║                       │       with audio)                           ║
   ║                       │              │            │                 ║
   ║                       │              └──┬─────────┘                 ║
   ║                       │                 │                           ║
   ║                       └────────────────►▼                           ║
   ║                                ┌──────────────────┐                 ║
   ║                                │ comparator_in_q  │                 ║
   ║                                │  (drained at     │                 ║
   ║                                │   end-of-stages) │                 ║
   ║                                └──────────────────┘                 ║
   ║                                          │                          ║
   ║                                          ▼                          ║
   ║                                Stage 5 comparator                   ║
   ║                                (numpy-vectorised hamming;           ║
   ║                                  inner pairwise dist in a           ║
   ║                                  to_thread pool)                    ║
   ╚═════════════════════════════════════════════════════════════════════╝

Per-file subprocess strategy:
  - Long-lived ffprobe worker (one per CPU metadata worker): JSON-RPC pattern
    over stdin/stdout — `printf "%s\n" path1 path2 ... | ffprobe-loop`
    (or use a *pool* of pre-spawned ffprobe processes; reuse rather than
     respawn)
  - GPU pHash: persistent ffmpeg server pattern (see §6) reading work items
    from stdin; 2–4 such servers saturate consumer-class NVDEC
  - audio FP: pre-spawned ffmpeg processes with stdin file-list pattern;
    pool size = min(cpu_count, 4–6)

Backpressure:
  - asyncio.Queue with maxsize bounds memory automatically
  - if hash_q is full, metadata workers naturally block on put()
  - cap total in-flight bytes (sum of stage-3 buffered frames) to ~512 MB
```

### Properties of this architecture

1. **Pipeline parallelism by default.** The metadata stage doesn't wait for
   all of stage 2 to finish — file 1's metadata feeds stage 3 while file 200's
   metadata is still ffprobing.
2. **Pool-per-stage with stage-specific tuning.** GPU stage uses 3–4 workers
   (matches NVDEC engines), CPU stage uses `cpu_count`, audio stage uses a
   conservative pool that won't oversubscribe.
3. **Bounded queues = automatic backpressure.** `maxsize` on each queue caps
   in-flight items. No explicit "memory budget" calculation needed.
4. **`TaskGroup` for failure semantics.** If a worker raises, all sibling
   workers are cancelled; cancellation propagates through `asyncio.sleep` and
   `await process.wait()` points cleanly.
5. **Cache hits never enter the slow lanes.** Stage 1.5's hit/miss partition
   feeds two queues; hits skip directly to the comparator queue. Today's code
   already does this conceptually, but with stream-pipelining the win is
   bigger because hits don't have to wait for stage 2 to finish on misses.

---

## 3. Per-finding sections

### F1. Producer-consumer with bounded queue between stages

**Where**: `backend/api/scan.py:264–576` (all four batch loops).

**Today**: each stage is a synchronous "batch → gather → barrier → batch" loop.
The barrier means stage *N+1* waits for the last file of stage *N* — slowest
file in batch gates the whole batch (head-of-line blocking).

**Proposed**: replace the four `for batch_start in range(...)` loops with a
`TaskGroup` of worker pools draining bounded `asyncio.Queue`s. See §4
(Backpressure & failure handling) for the complete sketch.

**Quantified win**: head-of-line blocking ratio depends on duration variance.
For a library where the longest video in a batch is `K×` the median, the
serial-batch pipeline pays `K×` per batch where streaming pays `1×`. On a
realistic mixed library (`K ≈ 4–6`) the win is 25–40% over identical worker
counts.

**Pause/stop semantics**: the simplest correct pattern is for each worker to
do `await _pipeline_check(...)` at top of each `while` iteration. When stopped,
workers `break` out; `TaskGroup.__aexit__` then cancels any pending workers
holding `process.communicate()` futures. **Pause** is harder: blocking workers
on `resume_event.wait()` causes queues to fill up — once they fill, producers
block too, so the pipeline naturally drains to a frozen state. This is fine
but slow to respond. A better pattern: pause is just a flag, workers spin-wait
on it at `_check()`, and the queue's maxsize provides the natural memory
ceiling.

**Effort**: M (3–6 hours including pause/stop wiring).
**Risk**: M (correctness across cancellation; sentinel-routing bugs are easy).

---

### F2. Work-stealing thread pool vs `asyncio.Semaphore` — when GIL bites

**Background**:

- **`asyncio.Semaphore`** is purely a coroutine-scheduling primitive. It does
  *not* parallelise CPU work; it just limits how many coroutines can be in the
  critical section at once. When the coroutine inside the semaphore calls a
  *blocking* function (like `subprocess.run` or `numpy.bitwise_xor` on a large
  array), the event loop stalls.
- **`concurrent.futures.ThreadPoolExecutor`** parallelises blocking C-level
  calls (subprocess, file I/O, numpy ufuncs that release the GIL). It does
  **not** parallelise Python-level CPU work because of the GIL.
- **`concurrent.futures.ProcessPoolExecutor`** parallelises Python-level CPU
  work by spawning interpreters. Has IPC overhead (pickle the args + return).

**Where each fits in this pipeline**:

| Stage | Work shape | Primitive |
|-------|------------|-----------|
| Stage 2 ffprobe | Subprocess spawn + wait (I/O-bound from Python's view) | `asyncio.create_subprocess_exec` + `Semaphore` — current choice is correct |
| Stage 2 thumbnail | Subprocess spawn + wait | Same as above |
| Stage 3 ffmpeg decode | Subprocess spawn + wait | Same — but consider persistent process (§F4) |
| Stage 3 pHash compute (`imagehash.phash`) | Python CPU + PIL + numpy. PIL ops release GIL on resize; numpy on DCT. *Mostly* GIL-released but with Python glue between steps | `ThreadPoolExecutor` via `loop.run_in_executor` — current pattern using `asyncio.to_thread` works |
| Stage 4b audio FP RMS | numpy reduction on 86 MB array — GIL released entirely | `ThreadPoolExecutor` |
| Stage 5 comparator inner loop | Pure Python; `compare_hash_sets` builds 12×12 distance matrix in a Python `for` loop | **Currently single-threaded.** Vectorise (§F9) so numpy releases GIL, then `ProcessPoolExecutor` for very large groups |

**Where `ProcessPoolExecutor` would win**: if `compare_hash_sets` and
`compare_audio_fingerprints` ran on the same 100k pairs without numpy, you'd
be GIL-bound. Today the python all-pairs loop in `comparator.py:117–155` is
the most CPU-bound code that *doesn't* release the GIL. For a duration group
of 1000 items (the pathological "60.0s TikTok" cluster), that's ~500k pairs ×
a ~50 µs Python comparison ≈ **25 seconds single-threaded**. Vectorising drops
this; sharding across `os.cpu_count()` processes drops it again by ~6–8×.

**Effort**: S to wire `ProcessPoolExecutor` once #F9 is in place.
**Risk**: M — pickle overhead for large groups; need to chunk smartly.

**Recommendation**: keep `asyncio.Semaphore` for subprocess-orchestrating
stages, `ThreadPoolExecutor` for numpy-heavy parts (pHash compute, audio RMS,
vectorised Hamming), and reserve `ProcessPoolExecutor` for the *single* place
where Python is in the inner loop (the comparator), and only when group size
exceeds a threshold (say, 200).

---

### F3. Optimal batch size for ffmpeg subprocess spawn on Windows vs Linux

**Measured spawn overhead (typical numbers)**:

| Platform | Mechanism | Overhead per `subprocess.Popen()` of ffmpeg |
|----------|-----------|---------------------------------------------|
| Linux (3.10 with `posix_spawn` / `vfork`) | vfork+exec | **2–5 ms** |
| Linux pre-3.10 (preexec_fn or `close_fds`) | fork+exec, COW pages | 10–30 ms |
| Windows 10/11 | `CreateProcess` | **20–50 ms** |
| Windows (Defender on) | `CreateProcess` + AV scan | 50–150 ms |

Source: CPython issue tracker (BPO #11314 cites ~40% overhead on subprocess
creation; `vfork`-based subprocess fixes brought Linux from ~hour to ~8 min
on workloads with 100 subprocess spawns). Windows Defender real-time scanning
of newly spawned executables is well-known to add 30–100 ms for every
ffmpeg.exe spawn — adding a Defender exclusion for the project's ffmpeg path
is a free 30% win on Windows.

**Implication**: for a video that decodes in `T` seconds, the *amortised*
overhead is `spawn_ms / T`. On Linux with a 30-second clip, spawn is ~0.02%
of work; ignore it. On Windows with a 5-second clip and Defender on, spawn
might be **30% of total time** — and at 12 concurrent files, you have 12
context switches per spawn, each ~5 µs, so total wall-clock effect compounds.

**Right batch size**:

- **Linux**: batch size doesn't matter for spawn overhead. Pick whatever fits
  your I/O / GPU pool. The current `BATCH = max_concurrent * 4 = 48` is fine.
- **Windows**: spawn overhead favours **fewer, larger batches** to amortise
  filesystem cache warmup for `ffmpeg.exe` and its DLLs. But the *biggest* win
  on Windows is to **avoid spawning at all** — see §F4 (persistent processes)
  and the Defender exclusion mentioned above.

**Concrete recommendation**: ship a one-time "first-scan" diagnostic that
reports `time.perf_counter()` deltas around the first subprocess spawn. If
they exceed 100 ms on Windows, the docs already say to add the Defender
exclusion; surface this prominently.

**Effort**: S to add timing diagnostic.
**Risk**: L.

---

### F4. Persistent ffmpeg processes (the big Windows win)

This is the single biggest under-explored optimisation in the pipeline. On
Windows, **30–50 ms × 4 subprocesses × 10000 files = 20 minutes of pure
`CreateProcess` overhead**, all of which is avoidable.

**Important caveat**: native ffmpeg/ffprobe CLI tools do **not** support
"reading the next filename from stdin and processing it". Each invocation
takes file paths in `argv` and exits when done. So "persistent ffmpeg" in
the literal sense is *not* possible without a thin wrapper.

The two practical patterns:

1. **PyAV (recommended)**: PyAV is a direct Python binding to FFmpeg's
   libraries (libavcodec, libavformat, libswscale). It eliminates subprocess
   spawn *and* the JSON-parse/string-marshal overhead of the ffprobe wrapper.
   Per-call cost drops to a single Python function call against a long-lived
   in-process FFmpeg runtime. Mature and widely used (`decord`,
   `torchvision.io`, `imageio`).
2. **Recycled subprocess pool**: spawn N long-lived Python child processes
   that have already imported PyAV/imagehash/PIL. Drive them via stdin JSON.
   Saves Python-import cost but not the CUDA-context-init cost (each process
   still allocates its own CUDA context). Worth it on Windows for the
   sub-process-spawn savings alone; less so on Linux.

The persistent-ffmpeg sketch (§ next subsection) shows pattern #2. PyAV
(pattern #1) makes the whole subprocess-wrapper layer obsolete and is the
direction the codebase should head if the goal is "minimise per-file overhead
to the floor".

#### Persistent-ffmpeg sketch (illustrative)

A concrete code shape for §F4b option 3 (subprocess-recycling — not
production-ready, but shows the pattern):

```python
# backend/services/persistent_ffmpeg.py  (NOT for ship — sketch only)
"""
A long-lived ffmpeg worker that accepts JSON command-objects on stdin
and replies with JSON results on stdout. Maintains a stable Python parent
process so module imports (PyAV, imagehash, PIL) only happen once.

Each "command" is a dict {"path": str, "num_frames": int, ...}.
The worker decodes that many frames using PyAV (CUDA when available)
and writes back {"path": str, "hashes": [str, ...]} per command.
"""
import asyncio, json, sys
import av  # PyAV
import imagehash
from PIL import Image

# In the CHILD process (one per slot in the pool):
def _child_main():
    # Imports happen ONCE per child lifetime, not per file.
    for line in sys.stdin:
        cmd = json.loads(line)
        try:
            with av.open(cmd["path"], options={"hwaccel": "cuda"}) as cont:
                stream = cont.streams.video[0]
                duration = float(stream.duration * stream.time_base)
                hashes = []
                for i in range(cmd["num_frames"]):
                    pts = int((duration * i / cmd["num_frames"]) / stream.time_base)
                    cont.seek(pts, stream=stream)
                    for frame in cont.decode(stream):
                        img = frame.to_image().convert("L").resize((320, 180))
                        hashes.append(str(imagehash.phash(img, hash_size=16)))
                        break
            resp = {"path": cmd["path"], "hashes": hashes}
        except Exception as e:
            resp = {"path": cmd["path"], "error": str(e)}
        sys.stdout.write(json.dumps(resp) + "\n"); sys.stdout.flush()


# In the PARENT (FastAPI) process:
class PersistentHashPool:
    """Round-robin dispatch of JSON commands to N pre-spawned children."""
    async def start(self, n: int):
        self._procs = [
            await asyncio.create_subprocess_exec(
                sys.executable, "-u", __file__, "--child",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            ) for _ in range(n)
        ]
        # In production: use one asyncio.Queue per worker, manager picks
        # the shortest queue, per-worker pending count bounded for backpressure.

    async def hash(self, path: str, num_frames: int) -> list[str]:
        proc = self._pick_worker()                # least-busy worker
        proc.stdin.write((json.dumps({"path": path, "num_frames": num_frames}) + "\n").encode())
        try:
            await proc.stdin.drain()
        except ConnectionResetError:               # BPO #39010 on Windows
            raise WorkerCrashedError(path)
        resp = json.loads(await proc.stdout.readline())
        if "error" in resp: raise RuntimeError(resp["error"])
        return resp["hashes"]

    async def shutdown(self):
        for p in self._procs:
            p.stdin.close()
            try: await asyncio.wait_for(p.wait(), timeout=5)
            except asyncio.TimeoutError: p.kill(); await p.wait()
```

**Caveats**:

1. **Worker affinity / ordering**: this naive round-robin can starve. A real
   implementation needs per-worker `asyncio.Queue` with a manager picking the
   shortest queue.
2. **Stuck child**: if a child hangs on a malformed file (PyAV's `av.open`
   can spin on certain corrupt MKVs), the parent must time it out and recycle.
3. **Windows**: stdin/stdout pipes on Windows asyncio use the Proactor loop;
   `proc.stdin.drain()` can raise `ConnectionResetError` if the child exits
   mid-write (BPO #39010). Wrap drain in try/except.
4. **Backpressure**: per-worker pending response count must be capped or the
   pool will buffer unread responses until OOM.

**Realistic estimated win**: on Windows, eliminating subprocess spawn for the
GPU pHash stage saves 20–40 ms per file. For 10000 files this is 3–7 minutes
of wall-clock — not huge but a *consistent* 5–15% reduction on Windows. On
Linux the win is much smaller (~1–2%) since `vfork` is fast.

**Effort**: M (one well-tested module; subprocess lifecycle is fiddly).
**Risk**: M — primarily testability and Windows pipe-buffering edge cases.
**Better alternative**: **switch to PyAV** entirely (no subprocesses), which
gets you everything this pattern gets plus eliminates per-process VRAM
duplication. PyAV is BSD-licensed, pip-installable, and is what every major
Python video tool uses (`decord`, `torchvision.io`, `imageio`).

---

### F5. GPU saturation patterns: how many concurrent NVDEC sessions?

This is the **largest unrealised win** in the current configuration.

**Hardware facts**:

| GPU       | Chip   | NVDEC units (enabled on GeForce) | NVENC units | Notes |
|-----------|--------|----------------------------------|-------------|-------|
| RTX 3060  | GA106  | 1 NVDEC                          | 1 NVENC     | Ampere — single decode engine |
| RTX 4060/4070 | AD104 | 2 NVDEC (4 physical, 2 enabled) | 2 NVENC | Ada Lovelace |
| RTX 4080/4090 | AD103/AD102 | 2 NVDEC enabled            | 2 NVENC     | Same NVDEC count |
| A100 (datacenter) | GA100 | 5 NVDEC                     | 0 NVENC     | Decode-heavy server SKU |

**Per-engine throughput scaling** (from public benchmarks and NVIDIA docs):

> Two streams hold about 50%, three about 80% and more than four streams to
> reach 100% NVDEC saturation.

That figure is **per NVDEC engine**. So for a GeForce consumer card:

| GPU       | Engines | Saturation knee (streams) | Practical max concurrent |
|-----------|---------|---------------------------|---------------------------|
| RTX 3060  | 1       | ~3–4                      | 4                         |
| RTX 4070  | 2       | ~6–8                      | 8                         |
| RTX 4090  | 2       | ~6–8                      | 8                         |
| A100      | 5       | ~15–20                    | 20                        |

**Current config (`GPU_MAX_CONCURRENT = 12`)** is:

- **3× past the knee on RTX 3060** — extra 9 streams contribute almost zero
  GPU throughput, but consume 9 ffmpeg subprocesses, 9× CUDA context init
  (Ampere is ~40 MB VRAM per CUDA context), and 12-wide CPU thread contention
  on filter-chain execution.
- **~1.5× past the knee on RTX 4070** — modest waste.
- **Right-sized for A100** (which nobody is running this scanner on).

**Diagnostic to ship**: `nvidia-smi dmon -s puc` running for the first 60s of
a scan will reveal NVDEC utilisation. If saturating at <90% with 12 concurrent
streams, the bottleneck is **on the host (CPU filter chain or PCIe transfer)**
not the GPU. In that case fewer concurrent streams + more per-stream work
(batched filter graph, longer clips) is the win.

**Recommendation**: introduce **per-GPU heuristics** in
`backend/services/gpu_detector.py`:

```python
def recommended_hash_concurrency(gpu: GPUInfo) -> int:
    name = (gpu.gpu_name or "").upper()
    if "3060" in name or "3050" in name:  # GA106/GA107: 1 NVDEC
        return 4
    if "3070" in name or "3080" in name or "3090" in name:  # GA104/2/2: 1–2 NVDEC
        return 6
    if "4060" in name or "4070" in name or "4080" in name:  # AD104/3/3: 2 NVDEC
        return 8
    if "4090" in name:  # AD102: 2 NVDEC
        return 8
    if "A100" in name or "H100" in name:  # datacenter: 5 NVDEC
        return 16
    return 6  # safe default
```

Then **stage 2 (metadata, mostly CPU-bound ffprobe)** and **stage 3 (pHash,
GPU-bound)** use *different* limits:

```python
hash_sem  = asyncio.Semaphore(recommended_hash_concurrency(gpu))
meta_sem  = asyncio.Semaphore(min(os.cpu_count() or 4, 8))
audio_sem = asyncio.Semaphore(min(os.cpu_count() or 4, 4))  # CPU PCM decode
```

**Expected win**: 20–40% reduction in stage 3 wall-clock on RTX 3060 (less
context-switch thrash; more per-stream throughput). Negligible on A100.

**Effort**: S — small lookup table + three semaphores.
**Risk**: L — strictly more conservative than the current setting on most
hardware; risk is only "leaves throughput on the table for tomorrow's GPUs".

---

### F6. Async vs threaded for I/O-bound: ffprobe vs pHash compute

The distinction matters because asyncio's "concurrency" relies on the await
point. A blocking call inside an async function blocks the loop entirely.

**ffprobe / ffmpeg subprocess**: I/O-bound from Python's view. The right
primitive is `asyncio.create_subprocess_exec` + `await proc.communicate()`.
The asyncio event loop suspends the calling coroutine when waiting on the
subprocess; another coroutine runs in the meantime. The current code uses
this correctly:

```python
# services/metadata.py — uses asyncio.create_subprocess_exec internally
proc = await asyncio.create_subprocess_exec(*cmd, ...)
stdout, stderr = await proc.communicate()
```

**pHash compute** (`imagehash.phash(img, hash_size=16)`): mostly CPU, calls
into PIL + numpy. PIL's resize releases the GIL (it's C). The DCT (which
phash computes internally as `numpy.fft.fft2` or scipy DCT) releases the GIL.
The Python glue between them does *not* release it.

The right primitive is `loop.run_in_executor(thread_pool, ...)`. The current
hasher does this:

```python
# services/hasher.py:38
_executor = ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_FFMPEG * 3)
```

That `* 3` factor is unused (semaphore caps incoming work at `max_concurrent`)
but harmless. The real concern: is 24 threads competing for the GIL during
the Python-glue parts of `phash`? Probably yes. For 12 concurrent files each
finishing one phash every ~30 ms (CPU stage of pHash compute), 12 × 30 ms /
sec = 360 ms of GIL contention per second. Not catastrophic but measurable.

**Audio FP RMS**: pure numpy reduction. `np.sqrt(np.mean(seg**2))` releases
the GIL for the whole call. A thread pool is fine here; threading scales
near-linearly until you saturate the CPU cores.

**Recommendation**: shrink the hasher thread pool to `max_concurrent` (no
`* 3`) and consider a separate **`ProcessPoolExecutor` for the comparator's
inner loop** (once vectorised, §F9, even more wins from sharding across
processes).

**Effort**: S.
**Risk**: L.

---

### F7. `asyncio.gather` failure semantics: are we leaking subprocesses?

The pipeline uses:

```python
results = await asyncio.gather(*tasks, return_exceptions=True)
```

This is **safe** in the sense that no exception propagates to abort the
batch. But it does **not** cancel sibling tasks when one fails. With
`return_exceptions=True`:

- Every task runs to completion (success or exception).
- Exceptions are returned as values in the result list.
- **Pending ffmpeg subprocesses owned by sibling tasks keep running.** They
  finish naturally — fine if they're well-behaved.

**Where this can leak**:

1. If `_hash_one` raises midway between `await sem.acquire()` and the actual
   subprocess `communicate()`, the subprocess (already spawned) keeps running
   detached. Today the code raises only inside `extract_and_hash`, which is
   itself wrapped in try/except in the hasher, so this is unlikely.
2. **The pause/stop signal path** uses `scan_control.is_stopped()` flag
   checks, not `task.cancel()`. So when stop is pressed during a batch:
   - Workers that haven't yet acquired the semaphore see `is_stopped()` and
     bail.
   - Workers *in* the semaphore continue to completion (waiting on their
     subprocess).
   - Their subprocess completes its decode (potentially seconds of work),
     then their result is discarded.
   - Net: stop is "graceful" but slow — you wait for in-flight subprocesses.
3. **Process crash midway through a batch**: if Python crashes (OOM, segfault
   from PyAV bug), all spawned subprocesses become orphans. Linux: they get
   re-parented to init and run to completion. Windows: they are NOT reaped
   by their parent's exit — they stay running until they finish (or, for an
   ffmpeg with a stdin pipe, until they error out).

**Fix**: `asyncio.TaskGroup` (Python 3.11+) is strictly better here. On any
task error it cancels siblings. Combined with explicit subprocess cleanup in
worker code:

```python
async def _hash_one(video):
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(..., ...)
        stdout, _ = await proc.communicate()
        return parse(stdout)
    except asyncio.CancelledError:
        if proc is not None and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        raise
```

**Stop becomes**: convert the in-memory `stop_event` into an `asyncio.Event`
checked at every queue-`get`, and have the pipeline's TaskGroup raise on stop.
TaskGroup cancels every worker; workers cleanly kill their owned subprocesses.

**Recommendation**: replace `asyncio.gather(..., return_exceptions=True)` with
`asyncio.TaskGroup` in any new streaming-pipeline implementation; keep
per-task try/except for per-file errors (which should *not* abort the
pipeline).

**Effort**: M (need to refactor the per-worker error handling).
**Risk**: M (Python 3.11+ requirement; check `requirements.txt`).

---

### F8. Backpressure: what bounds memory when stages run at different speeds?

In a streaming pipeline, if stage 3 produces faster than stage 5 consumes,
memory bloats. In the current "barrier-after-stage" code this is not a
concern — every stage drains entirely before the next starts. But in any
streaming design (§F1), you need backpressure.

**Pattern**: `asyncio.Queue(maxsize=N)`. When full, `await queue.put()` blocks
the producer until a consumer takes an item.

**Sizing the queues**:

| Queue              | What it carries                       | Estimated size per item | Recommended maxsize |
|--------------------|---------------------------------------|--------------------------|----------------------|
| metadata_q         | `(file_info, cache_row)` tuple        | ~1 KB                    | 100–200              |
| hash_q             | `VideoFile` ORM + `_meta_video_info`  | ~2 KB                    | 50                   |
| audio_q            | `VideoFile` ORM + path                | ~2 KB                    | 50                   |
| done_q (comparator input) | dict with up to 12 × 32-byte hashes | ~1 KB                  | unbounded — drained by comparator |

The **memory ceiling** with these caps: ~200 × 1 KB + 50 × 2 KB + 50 × 2 KB
= ~400 KB queue overhead. Pleasantly small.

The real memory concern is **stage 3 in-flight buffers**: each running pHash
extraction holds the decoded thumbnail set in memory until phash() runs. At 12
concurrent workers × 12 frames × ~250 KB raw 320×180 image = ~36 MB. Trivial.

**Cross-stage memory pressure** comes from `video_records` (the
identity-map list documented in `pipeline-optimizations.md` finding #5),
which is *separate* from the streaming queues. Both should be addressed but
they're independent issues.

**Pattern for "stop the world if anyone falls behind"**:

```python
async def metadata_worker(in_q, out_q):
    while True:
        item = await in_q.get()
        if item is SENTINEL:
            await out_q.put(SENTINEL)
            in_q.task_done()
            break
        try:
            result = await process(item)
            await out_q.put(result)   # ← blocks if out_q is full
        finally:
            in_q.task_done()
```

The `await out_q.put(...)` is the backpressure point. If stage 3 is full, the
metadata worker just blocks here — its `sem` slot is held but doing nothing
of interest, so other workers in the pool can advance.

**Effort**: included in §F1.
**Risk**: included in §F1.

---

### F9. Comparison-stage parallelism

The comparator is currently single-threaded:

```python
# services/comparator.py:117–155
for i in range(n):
    for j in range(i + 1, n):
        if not _file_size_compatible(videos[i], videos[j]):
            continue
        hashes_j = videos[j].get("hashes") or []
        dist = compare_hash_sets(hashes_i, hashes_j)
        ...
```

Union-find is inherently serial (path-compression breaks parallel
correctness), but the **inner work** — building the 12×12 distance matrix in
`compare_hash_sets` — is embarrassingly parallel:

```python
# Current: hasher.py:540–551 — Python double loop
dist = np.full((n1, n2), 999, dtype=np.int32)
for i, b1 in enumerate(bits1):
    if b1 is None: continue
    for j, b2 in enumerate(bits2):
        if b2 is not None and len(b1) == len(b2):
            dist[i, j] = int(np.count_nonzero(b1 != b2))

# Vectorised replacement
B1 = np.stack([b for b in bits1 if b is not None])  # (n1, 256)
B2 = np.stack([b for b in bits2 if b is not None])  # (n2, 256)
# Pairwise XOR-popcount in a single broadcast
diffs = B1[:, None, :] != B2[None, :, :]            # (n1, n2, 256) bool
dist = diffs.sum(axis=-1, dtype=np.int32)           # (n1, n2)
```

For 12×12 hashes the constant-time gain is modest (~3–5×) but the GIL is
released for the whole compute. The bigger win: **batch the
inter-video comparisons too**. Instead of pairwise `compare_hash_sets(i, j)`,
build a `(n_videos, n_frames, 256)` tensor for the entire duration group and
compute all pairwise distances in one numpy call:

```python
# Stack all videos' hashes into one tensor
V = np.stack([np.stack([h for h in v["hashes"]]) for v in group])  # (k, f, 256)
# Pairwise distance matrix: (k, k, f, f)
# Then collapse to per-video-pair score via greedy / Hungarian assignment
pairwise = (V[:, None, :, None, :] != V[None, :, None, :, :]).sum(axis=-1)
# pairwise[i, j] is a (f, f) matrix of frame-to-frame distances for videos i,j
```

This is RAM-hungry — for a group of 50 videos × 12 frames × 256 bits, the
intermediate is 50×50×12×12×256 = 92 MB. Manageable for groups up to ~200,
chunk for larger ones.

**Further parallelism for huge groups**: chunk the i-dimension across a
`ProcessPoolExecutor`. Each worker computes a slab of the row-set, and the
union-find merge happens serially in the parent.

**Bigger algorithmic win** (already in `algorithmic-improvements.md`): a
BK-tree on aggregate hashes turns O(n²) into O(n log n) within a group, which
matters far more than vectorising the constant factor of an O(n²) sweep.

**Effort**: S for vectorisation; M for full BK-tree integration.
**Risk**: L for vectorisation (drop-in numpy); M for BK-tree (regression
testing).

---

## 4. Backpressure & failure handling — concrete pattern

A complete pattern for the streaming pipeline (illustrative — not for the repo):

```python
SENTINEL = object()

async def run_streaming_scan(scan_id, root_path, options):
    metadata_q = asyncio.Queue(maxsize=64)
    hash_q     = asyncio.Queue(maxsize=32)
    audio_q    = asyncio.Queue(maxsize=64)
    done_q     = asyncio.Queue()  # unbounded — comparator drains

    meta_sem  = asyncio.Semaphore(min(os.cpu_count() or 4, 8))
    hash_sem  = asyncio.Semaphore(recommended_hash_concurrency(gpu_info))
    audio_sem = asyncio.Semaphore(min(os.cpu_count() or 4, 4))
    NUM_META, NUM_HASH, NUM_AUDIO = 4, hash_sem._value, audio_sem._value

    async def metadata_worker():
        while True:
            item = await metadata_q.get()
            try:
                if item is SENTINEL:
                    await hash_q.put(SENTINEL); await audio_q.put(SENTINEL)
                    break
                async with meta_sem:
                    meta = await extract_metadata(item.path)
                video = build_video_record(item, meta)
                await hash_q.put(video)
                if is_duration_candidate(video):
                    await audio_q.put(video)
            except Exception:
                log.exception("metadata failed for %s", item.path)
            finally:
                metadata_q.task_done()

    async def hash_worker():        # similar — drains hash_q → done_q
        ...

    async def audio_worker():       # similar — drains audio_q → done_q
        ...

    async def comparator_consumer():
        # Drain done_q until both hash + audio stages have signalled SENTINEL,
        # then run the (vectorised) comparator on the accumulated set.
        ...

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(discover_and_dispatch(metadata_q, done_q))
            for _ in range(NUM_META):  tg.create_task(metadata_worker())
            for _ in range(NUM_HASH):  tg.create_task(hash_worker())
            for _ in range(NUM_AUDIO): tg.create_task(audio_worker())
            tg.create_task(comparator_consumer())
    except* _ScanStopped:
        # TaskGroup cancelled siblings on stop; subprocess cleanup
        # is handled inside each worker's CancelledError handler.
        await _mark_stopped(...)
```

The sentinel-propagation pattern (each upstream worker emits one SENTINEL;
each downstream stage counts `NUM_upstream` of them before signalling its own
end-of-stream) lets workers stay simple. Per-task `task_done()` after every
`get()` allows `queue.join()` for clean shutdown.

Key safety properties:

1. **Failure propagation**: `TaskGroup` cancels siblings on any uncaught
   exception. Per-file errors are caught inside workers and logged — they
   don't bring down the pipeline. Out-of-pipeline failures (DB error,
   `_ScanStopped`) do.
2. **Subprocess cleanup**: each worker wraps its subprocess in
   try/except/`CancelledError` handlers (sketch in §F7) that explicitly
   `terminate()` + `kill()` on cancellation.
3. **Backpressure**: `metadata_q.maxsize` caps how far ahead the discoverer
   can run. `hash_q.maxsize` caps how far ahead metadata can run before stage
   3 catches up. Memory is bounded by the product of maxsize × per-item size.
4. **Cache hits short-circuit**: hits never enter the slow lanes; they go
   straight to `done_q`. Today this is fine because there's a final
   `video_records` mega-list anyway; in the streaming design the comparator
   sees them at the same time as fresh-computed entries.
5. **Pause**: a global `pause_event: asyncio.Event` is awaited inside each
   worker's loop at the top. Pause causes all workers to suspend; queues
   stay full but bounded, no memory bloat.

---

## 5. Caveats

### Windows asyncio quirks

- **ProactorEventLoop is the default** since Python 3.8 and is required for
  `asyncio.create_subprocess_*`. `SelectorEventLoop` does not support
  subprocesses on Windows. (Confirm `asyncio.get_event_loop_policy()` is the
  default; FastAPI does this for you.)
- **ConnectionResetError** on `proc.stdin.drain()` is a known Proactor quirk
  (BPO #38856, #39010). It fires when the child exits while the parent is
  still writing to stdin. Wrap drain calls in try/except, treat as "child
  closed early".
- **`ProactorEventLoop.subprocess_shell` limitations**: avoid `shell=True`
  with asyncio on Windows — use `create_subprocess_exec` exclusively.
- **Defender real-time scan** adds 30–100 ms per `ffmpeg.exe` spawn. Add the
  project's `ffmpeg.exe` path to Defender exclusions before benchmarking. This
  is *the* biggest single thing a Windows user can do for short-clip libraries.
- **Number-of-handles limit**: each pipe to a child uses ~3 OS handles. With
  12 concurrent subprocesses × 3 pipes × 2 ends = ~72 handles plus DB
  connections, WebSocket. Well below Windows' 16k handle limit for a process,
  but if you scale to hundreds of concurrent workers, watch out.

### GIL and thread-pool sizing

- **GIL is released by**: numpy/scipy ufuncs operating on C arrays, PIL
  `Image.resize`, file I/O, subprocess waits, sleep, time, and most OS calls.
- **GIL is held by**: any pure-Python loop, dict/list mutation, attribute
  access, function call overhead, exception handling.
- **Implication for thread pool sizing**: pools larger than ~2× cpu_count
  give diminishing returns even for GIL-released work because Python's `await`
  context switches still go through the GIL. The current
  `_executor = ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_FFMPEG * 3)`
  is oversized; capped at `cpu_count()` would be plenty.

### GC pressure on long-lived bytearrays

The current pipeline accumulates `video_records` (the ORM list), per-file
`_meta_video_info` dicts, and per-batch result lists. With the streaming
pipeline above, each item is referenced by exactly one queue at a time and
discarded after the comparator processes it. **Memory residency drops from
O(N) to O(maxsize)**.

Watch for these GC traps:

1. **Cyclic refs between `VideoFile` ORM and `FileCache`**: SQLAlchemy
   creates a `cache_row.video_files` collection. Setting `cascade="all, delete"`
   without `single_parent=True` can keep references alive in the session.
2. **Holding `bytes` of full PCM audio in memory**: today's audio FP decodes
   the *entire* track (`pipeline-optimizations.md` finding #1) — at 8 kHz mono
   16-bit, a 90-minute file is 86 MB of PCM. With 12 concurrent workers
   that's 1 GB transient. The sampled-decode fix is required *first*; only
   then does pHash buffer size become the dominant memory term.
3. **Numpy intermediates in the vectorised comparator**: the
   `(k, k, f, f)` distance tensor for big duration groups should be chunked
   in the i-dimension. Python's reference counting plus numpy's strided
   views can pin memory longer than you'd expect if you slice without
   `.copy()`.

### Asyncio cancellation tax

Adopting `TaskGroup` (§F7) means workers must handle `CancelledError`
correctly. Specifically:

```python
async def worker():
    try:
        while True:
            item = await queue.get()
            try:
                ...
            except asyncio.CancelledError:
                # We were cancelled mid-item; clean up *this* item's subprocess
                raise
    except asyncio.CancelledError:
        # TaskGroup is shutting us down; release any process pool slots
        # held in this worker
        raise
```

If you swallow `CancelledError` (e.g. `except Exception:` without re-raising
`CancelledError`), `TaskGroup` will hang in `__aexit__` waiting for the worker
to actually exit. This is a common source of bug reports against asyncio
projects post-3.11.

### SQLAlchemy + asyncio + long-lived workers

The current pipeline uses a single `async_session` for the whole pipeline.
With pool-of-workers, you have two choices:

1. **Single session, single producer of DB writes**: workers send DB-write
   intents through a `db_q` to a dedicated DB-writer coroutine. Cleanest.
2. **Session-per-worker**: each worker has its own session. Requires careful
   isolation (foreign key dependencies between workers' writes can deadlock).

Option 1 is strongly recommended. It's how Channels/aiohttp do similar
streaming workloads.

### `expire_on_commit=False` and memory

`pipeline-optimizations.md` finding #5 notes that `expire_on_commit=False`
in `database.py:16` keeps ORM instances "warm" across commits. This is fine
in the streaming model where instances live for ms-to-seconds, *bad* in the
mega-list model where they live for the whole scan. With the streaming
design proposed here, you can leave the setting alone — instances are
short-lived anyway.

### `BackgroundTasks` vs `asyncio.create_task`

`run_scan_pipeline` is registered as a FastAPI `BackgroundTasks` task in
`scan.py:760`. `BackgroundTasks` runs after the response is sent, in the same
event loop. The queue-handoff to `_start_next_queued()` uses
`asyncio.create_task` (`scan.py:731`) directly, **not** `BackgroundTasks`,
which is correct because there's no request response to wait on. Both share
the same loop and same `scan_control` registry — fine.

The streaming pipeline doesn't change this; `BackgroundTasks` is still the
right entry point.

---

## 6. Recommended rollout order

If implementing incrementally, here's an order that gives wins early without
multi-week refactors:

| Order | Change | Effort | Win |
|-------|--------|--------|-----|
| 1 | Per-GPU `recommended_hash_concurrency()` lookup (§F5) | 1 day | 20–40% stage-3 wallclock on consumer NVIDIA |
| 2 | Separate `meta_sem` / `hash_sem` / `audio_sem` (also in `pipeline-optimizations.md` #4) | 0.5 day | Decoupled tuning |
| 3 | Audio-FP sampling (cross-ref `pipeline-optimizations.md` #1) — independent of concurrency | 1 day | 80–95% stage-4b time |
| 4 | Run stage 3 + 4b in parallel via `asyncio.gather` of two `asyncio.create_task` (§F1 light) | 0.5 day | 10–25% overlap |
| 5 | Vectorise `compare_hash_sets` with numpy XOR-popcount (§F9) | 1 day | Constant factor + GIL release |
| 6 | Switch hot stages to `asyncio.TaskGroup` + subprocess `CancelledError` cleanup (§F7) | 1–2 days | Eliminates leak class |
| 7 | Full streaming pipeline with `asyncio.Queue` + sentinels (§F1) | 3–5 days | 25–50% wallclock on mixed work |
| 8 | Replace ffprobe + frame-extract subprocesses with **PyAV** (§F4) | 3–7 days | 5–15% on Windows |
| 9 | Persistent-ffmpeg pool (alternative to PyAV; only worth doing if PyAV is rejected) | 3–5 days | similar to #8 |

Items 1–5 are independent and can be reviewed/shipped as separate PRs.
Items 6–9 build on each other (#7 requires #6's cancellation discipline;
#8/9 only need #1 in place).

---

## 7. Things deliberately *not* recommended

- **Trio / anyio**: better cancellation than stdlib asyncio, but switching
  the project's async runtime is a major migration; the wins are marginal
  given Python 3.11's TaskGroup brought most of trio's safety to stdlib.
- **`multiprocessing.shared_memory` for the comparator's hash matrix**:
  needed only if you're sharding huge groups across processes. Save for when
  you actually see 10000+ video duration groups in practice.
- **`uvloop`**: drop-in faster event loop, **not supported on Windows**, so
  it doesn't help this project's primary target platform.
- **Removing the `_executor` ThreadPoolExecutor in `hasher.py`**: it's
  pulling its weight — phash's PIL/numpy work *is* GIL-friendly and benefits
  from threads. Right-size it (§F6), don't remove it.
- **Spawning new asyncio loops per task**: `asyncio.run_coroutine_threadsafe`
  with sub-loops is a known anti-pattern that creates more problems than it
  solves.

---

## Sources

- [Python 3.14 asyncio docs (Coroutines and Tasks)](https://docs.python.org/3/library/asyncio-task.html) — `gather`, `TaskGroup`, `wait_for`, cancellation semantics.
- [Python 3.14 asyncio docs (Subprocesses)](https://docs.python.org/3/library/asyncio-subprocess.html) — `create_subprocess_exec` and Windows ProactorEventLoop notes.
- [Python 3.14 asyncio Queue docs](https://docs.python.org/3/library/asyncio-queue.html) — bounded queues, `task_done()`, `join()`.
- [Python 3.14 `concurrent.futures` docs](https://docs.python.org/3/library/concurrent.futures.html) — ThreadPoolExecutor vs ProcessPoolExecutor.
- [BPO #11314 — subprocess creation overhead](https://bugs.python.org/issue11314) — 40% process creation overhead penalty on Linux pre-vfork.
- [BPO #39010 — ProactorEventLoop unhandled ConnectionResetError](https://bugs.python.org/issue39010) — Windows asyncio quirk.
- [BPO #38856 — Proactor wait_closed ConnectionResetError](https://bugs.python.org/issue38856)
- [SuperFastPython — asyncio.gather Exception in Task Does Not Cancel](https://superfastpython.com/asyncio-gather-exception-not-cancel/) — failure semantics analysis.
- [SuperFastPython — ThreadPoolExecutor vs ProcessPoolExecutor](https://superfastpython.com/threadpoolexecutor-vs-processpoolexecutor/) — GIL impact.
- [SuperFastPython — Forking Processes is 20× faster than Spawning](https://superfastpython.com/fork-faster-than-spawn/) — Linux vs Windows process creation.
- [movq.de — fork() vs vfork() and subprocess.Popen](https://movq.de/blog/postings/2023-11-26/0/POSTING-en.html) — Linux subprocess vfork details.
- [BPO #112334 — subprocess.Popen Linux regression CVE-2023-6507](https://github.com/python/cpython/issues/112334)
- [asyncio backpressure with semaphores](https://blog.changs.co.uk/asyncio-backpressure-processing-lots-of-tasks-in-parallel.html)
- [SQLAlchemy + asyncio TaskGroup discussion](https://github.com/sqlalchemy/sqlalchemy/discussions/9312) — TaskGroup recommended over gather for multi-session work.
- [NVIDIA FFmpeg with GPU acceleration docs](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html)
- [NVDEC Wikipedia — engine counts per Ampere/Ada chip](https://en.wikipedia.org/wiki/Nvidia_NVDEC)
- [Hostkey blog — multi-threaded video streaming on gaming GPUs](https://hostkey.com/blog/9-testing-multi-threaded-video-distribution-on-gaming-gpus/) — "Two streams hold about 50%, three about 80% and more than four streams to reach 100% NVDEC saturation."
- [Tom's Hardware — Nvidia consumer GPU encoding limits](https://www.tomshardware.com/news/nvidia-increases-concurrent-nvenc-sessions-on-consumer-gpus)
- [imageio-ffmpeg README](https://github.com/imageio/imageio-ffmpeg) — subprocess-per-file architecture.
- [imageio-ffmpeg #17 — hang in subprocess.communicate](https://github.com/imageio/imageio-ffmpeg/issues/17) — Windows pipe-buffer deadlock pattern.
- [ffmpeg-python feeding stdin](https://python-ffmpeg.readthedocs.io/en/latest/examples/feeding-data-to-stdin/)
- [ffmpeg-python #647 — multiple streams in one subprocess pipeline](https://github.com/kkroening/ffmpeg-python/issues/647)
- [hexhamming — SIMD-accelerated Hamming distance](https://github.com/mrecachinas/hexhamming) — popcount intrinsics for fixed-size bit strings.
- [FastAPI Background Tasks docs](https://fastapi.tiangolo.com/tutorial/background-tasks/)
- [oneuptime — Python asyncio queues](https://oneuptime.com/blog/post/2026-01-30-python-asyncio-queues/view) — bounded queue backpressure patterns.
- [oneuptime — FastAPI background task processing](https://oneuptime.com/blog/post/2026-01-25-background-task-processing-fastapi/view)
- [dev.to — CPU-intensive tasks from asynchronous stream](https://dev.to/ksaaskil/how-to-process-cpu-intensive-tasks-from-asynchronous-stream-17hf) — hybrid asyncio + executor pattern.
- [agentfactory — hybrid I/O + CPU workloads](https://agentfactory.panaversity.org/docs/Python-Fundamentals/asyncio/hybrid-workloads)
- [runebook — asyncio Windows pitfalls](https://runebook.dev/en/docs/python/library/asyncio-platforms/windows)
