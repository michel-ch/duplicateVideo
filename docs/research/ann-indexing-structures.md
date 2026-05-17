# ANN Indexing Structures for Binary Hamming Search at Scale

Companion / deepening of `algorithmic-improvements.md` section 1. That note
recommended a BK-tree at a high level and named FAISS as a fallback for
"50k+." This document goes deeper: what is the right index when we have
**N videos × 12 pHashes = 12·N points in 64-bit (or 256-bit) Hamming space**,
queried at radius **r ≤ 14**, across library sizes from 10k to 1M?

We assume the existing pipeline shape:
- Duration grouping is already a cheap, coarse bucket — the ANN step runs
  **within each duration bucket** in the realistic case, only ever globally
  if we drop that bucketing.
- The 12-frame best-match comparator stays as the final verifier on a
  shortlist; the ANN structure only produces candidate pairs.
- `imagehash.phash(hash_size=16)` produces 256-bit hashes today
  (`hasher.py:344`), even though the comparator and the `HASH_SIMILARITY_THRESHOLD=14`
  default were calibrated to feel like a 64-bit-scale threshold. We address
  this discrepancy explicitly in the recommendation section.

> **TL;DR.** At our current size (~10k–50k), build a per-duration-bucket
> **`faiss.IndexBinaryFlat`** keyed by an aggregate hash per video and
> use `range_search` with radius set to `HASH_SIMILARITY_THRESHOLD * 1.5`.
> Verify shortlisted pairs with the existing 12×12 best-match. At 100k–1M,
> swap the per-bucket Flat for a single global **`IndexBinaryIVF`** with
> `nlist≈sqrt(N)`, `nprobe∈{8,16,32}`. **BK-tree is fine but obsolete here**
> — Faiss Flat with popcount+AVX2 is already sub-linear in practice at this
> dataset size, dependency-free in Python (`pip install faiss-cpu`), and
> stays useful when we eventually move to CLIP embeddings.

---

## 1. Executive summary — top 3 picks for ~50k videos, 12 hashes each, r ≤ 14

| Rank | Structure | Why it wins for this size | Risk |
|------|-----------|---------------------------|------|
| **1** | **FAISS `IndexBinaryFlat` per duration-bucket** + 12-frame verifier | Exhaustive, exact, AVX2 popcount; bucket sizes are 50–500 so "exhaustive" is microseconds. Index "build" is a single `add()` call. No training. Single dependency: `faiss-cpu`. Same library serves CLIP embeddings later. | Approximately none — it's the exact answer with hardware-optimised distance kernels. |
| **2** | **FAISS `IndexBinaryIVF` global** (skip duration-bucketing for the ANN, keep duration only as a verifier filter) | At 50k–500k total entries `nlist≈sqrt(N)` gives ~200 cells of ~250 vectors each; `nprobe=8` is sub-millisecond per query at 99% recall@r=14. Survives the audio-fallback path (multiple per-video hashes can be queried in one batch). | Requires a training step (30k–256k samples for k-means). Recall depends on `nprobe`. |
| **3** | **`pynear.BKTreeBinaryIndex`** (C++ BK-tree, AVX2 popcount) | The classic discrete-metric structure done well. Native binary. No training. Roughly the simplest "structure" worth keeping in tools/. | Build cost is sequential O(N log N) hamming evals (≈O(N·b)); does not parallelize trivially. Degenerates at r > 20 or when many points share the same distance from the pivot. |

**Not in the top 3, and why:**
- `pybktree` (pure Python BK-tree): ~50× slower to build than `cppbktree`,
  ~5–10× slower to query. Dead on arrival at >10k items.
- `hnswlib` native binary: hnswlib **does not support Hamming** in the upstream
  Python wheel as of 2025 ([issue #535 still open](https://github.com/nmslib/hnswlib/issues/535)).
  A fork exists (`HNSW-HAMMING`). Not worth the maintenance.
- `datasketch.MinHashLSH`: built for **set** similarity (Jaccard). Adapting
  to per-bit Hamming requires unnatural reshaping. Skip.
- Multi-Index Hashing (MIH/IndexBinaryMultiHash): excellent at very large r
  on huge corpora but Faiss's own benchmark concludes it loses to plain hashing
  at our radii ([Faiss binary-hashing-index-benchmark](https://github.com/facebookresearch/faiss/wiki/Binary-hashing-index-benchmark)).
  Useful only past 1M items.
- pgvector: real but only if we already use Postgres. SQLite-backed today.

---

## 2. Decision matrix

Build/query times are order-of-magnitude estimates for **256-bit binary
hashes, AVX2 desktop CPU**, derived from the benchmark sources cited later.
Where 64-bit numbers exist (`cppbktree`, native BK-tree), they are noted.

| Structure | Build (N=50k) | Query (radius r=14) | Recall @ r=14 | Memory | Code complexity | Native binary? | Incremental add? | Python dep |
|-----------|---------------|---------------------|----------------|--------|------------------|-----------------|-------------------|-------------|
| **Linear scan** (numpy XOR+popcount) | 0 ms | ~5–15 ms / query (50k×256 bit) | 100% (exact) | 1.6 MB raw | trivial | yes | yes | numpy |
| `pybktree` (pure Py) | ~30–60 s for 50k | ~20–80 ms / query | 100% (exact) | ~12 MB | very low | yes | yes (slow) | pure Python |
| `cppbktree.BKTree64` | ~3 s for 50k @ 64-bit | ~0.01–30 ms / query, r-dependent | 100% (exact) | ~5 MB | low | yes (64-bit only) | yes | pip wheel |
| `pynear.BKTreeBinaryIndex` | ~1–5 s for 50k | ~0.5–10 ms / query | 100% (exact) | ~6 MB | low | yes (uint8) | rebuild | pip wheel |
| `pynear.VPTreeBinaryIndex` | ~1–3 s for 50k | similar to BK; slightly slower at r>6 | 100% (exact) | ~6 MB | low | yes | rebuild | pip wheel |
| **`faiss.IndexBinaryFlat`** | <50 ms for 50k | ~0.5–2 ms / query batched, exhaustive but AVX2 popcount | 100% (exact) | 1.6 MB | very low | yes | yes (`add`) | `faiss-cpu` |
| **`faiss.IndexBinaryIVF`** (nlist≈√N) | ~1–3 s + train ~1 s | ~0.05–0.3 ms / query @ nprobe=16 | 95–99% tunable | ~2 MB | medium | yes | yes after train | `faiss-cpu` |
| `faiss.IndexBinaryHash` (BHash32) | ~0.5 s | competitive at small r, falls off at r>32 | 99% at r=15 in Faiss bench | ~3–5 MB (hash tables) | medium | yes | yes | `faiss-cpu` |
| `faiss.IndexBinaryMultiHash` (BHash4x32) | ~1 s | similar to BHash but more random accesses | 99% at r=15 | ~6–12 MB | medium | yes | yes | `faiss-cpu` |
| `faiss.IndexBinaryHNSW` (BHNSW16) | ~10–20 s for 50k (graph) | sub-ms / query, kNN-style | ~95–99%, not range-native | ~10 MB | medium | yes | yes (graph mod) | `faiss-cpu` |
| **Multi-Index Hashing** (norouzi/mih) | ~10–30 s for 50k (C++ binary) | sub-ms for small r | 100% (exact) | high (m tables) | high (external binary) | yes | append-only | external |
| `falconn`/multi-probe LSH | ~1–2 s | sub-ms, recall-tunable | 80–95% w/ banding | medium | medium | indirectly (LSH on bits) | rebuild | `falconn-python` |

**Important caveat on recall:**
"Recall @ r=14" is **with respect to the per-video aggregate hash** (one
hash per video, e.g. bitwise-median of the 12 frames). The end-to-end recall
of *whether two videos are duplicates* depends on what we put through the
ANN index. With the aggregate-hash strategy + a generous index radius
(~21, i.e. 1.5× threshold) followed by the 12-frame verifier, we recover
near-100% of the matches the current pairwise comparator finds, while only
running the 12×12 cost on a tiny shortlist.

---

## 3. Per-finalist deep dive

### 3.1 FAISS `IndexBinaryFlat` — the surprise winner at our scale

**Why it's a serious answer, not a fallback.** "Flat" sounds like O(n²),
but Faiss's binary distance kernels use **AVX2 / NEON popcount** intrinsics
tuned for 256-bit vectors specifically. The "Flat" search over a 50k-row
binary index returns in low milliseconds on a single core; against a
duration-bucket of 50–500 entries it's microseconds. The "build" is a
straight memcpy.

**API.** `faiss.IndexBinaryFlat(d)` where `d` is the **bit-length** (256
for our pHashes), with the constraint `d % 8 == 0`. Vectors fed in as
`uint8` numpy arrays of shape `(n, d//8)`.

```python
import faiss
import numpy as np

d = 256  # 16x16 pHash, 32 bytes
index = faiss.IndexBinaryFlat(d)

# x_db shape: (N_videos, 32) uint8, one aggregate hash per video
index.add(x_db)

# Query: radius search returns ALL neighbors within Hamming radius
lims, D, I = index.range_search(x_query, radius=21)
# lims[i:i+1] -> slice into I, D for the i-th query
```

**Range search return values:** `lims` is a `(n_query+1,)` int64 array
of CSR-style offsets; `I` are int64 neighbor IDs;
`D` are int32 Hamming distances. So for query i, neighbors are
`I[lims[i]:lims[i+1]]` and their distances are `D[lims[i]:lims[i+1]]`.
([Faiss Special-operations-on-indexes](https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes).)

**Incremental add.** `add()` and `add_with_ids()` both work and are O(1)
per vector (literally a memcpy). To remove: `remove_ids(IDSelectorBatch(...))`.
Persistence: `faiss.write_index(index, "scan.faissbin")` →
`faiss.read_index("scan.faissbin")`. Tiny on disk: 50k × 32 bytes + a
header ≈ 1.6 MB.

**Integration sketch (matches the existing pipeline).**

```python
# In comparator.py, replace find_duplicates_in_group() pair loop with:
import faiss, numpy as np

def _aggregate_hash(hash_hex_list):
    """Bitwise majority across 12 hashes → 32-byte uint8."""
    bits = np.stack([
        np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8))
        for h in hash_hex_list
    ])  # shape (12, 256)
    majority = (bits.sum(axis=0) >= (len(hash_hex_list) // 2 + 1)).astype(np.uint8)
    return np.packbits(majority)  # shape (32,)

def candidate_pairs_faiss(videos, hash_threshold=14):
    aggs = np.stack([_aggregate_hash(v["hashes"]) for v in videos])
    d = aggs.shape[1] * 8
    index = faiss.IndexBinaryFlat(d)
    index.add(aggs)
    radius = int(hash_threshold * 1.5)  # widen for the verifier
    lims, D, I = index.range_search(aggs, radius=radius)
    pairs = set()
    for i in range(len(videos)):
        for j_idx in range(lims[i], lims[i+1]):
            j = int(I[j_idx])
            if j > i:
                pairs.add((i, j))
    return pairs  # then verify each pair with existing compare_hash_sets
```

**Gotchas.**
- `range_search` returns the query itself if it's also in the database
  (distance 0). Filter `j > i` to deduplicate pairs.
- The `d` argument is **bits**, not bytes. Easy to mis-type.
- `IndexBinaryFlat` doesn't accept `IDMap` automatically — wrap with
  `IndexBinaryIDMap(index)` if you need stable IDs surviving deletes.
- Faiss on Windows ships only `faiss-cpu`. `faiss-gpu` requires CUDA and
  the project's Docker image already has CUDA, but the CPU path is so
  fast for binary Flat that GPU adds nothing here.

### 3.2 FAISS `IndexBinaryIVF` — the scale-up

When **total N > ~200k** (which currently means >16k videos × 12 hashes,
if we ever index per-frame instead of per-video) or when we drop the
duration bucketing entirely and want one global structure, the inverted
file gives a sub-linear query.

**Construction.**
```python
quantizer = faiss.IndexBinaryFlat(d)
nlist = int(np.sqrt(N))         # ~225 for 50k, ~1000 for 1M
index = faiss.IndexBinaryIVF(quantizer, d, nlist)

index.train(x_train)             # 30k–256k samples; can be the dataset itself
index.add(x_db)
index.nprobe = 16                # tune for recall/speed
lims, D, I = index.range_search(x_query, radius=21)
```

**Parameter guidance.**
- `nlist ≈ √N` is the Faiss community rule of thumb; for binary it
  works well up to ~4·√N.
- `nprobe` is the main tuning knob. The Faiss wiki binary benchmark
  fixes recall at 99% and reports the cheapest `nprobe`; at radius 15
  with 50M·256-bit vectors, `nprobe ∈ {32, 64}` at 99% recall, far less
  than nlist=4096. At our 50k scale you'd see `nprobe ∈ {8, 16}`.
- `IndexBinaryIVF` returns *strictly less than* `radius` — for ≤14
  pass `radius=15`. Easy off-by-one.

**Build time / training samples.** Faiss recommends 30k–256k training
samples for k-means. With 12·50k = 600k points we have plenty. The k-means
itself is ~1–2 s for nlist=225 on commodity CPU.

**Memory.** Mostly the codes themselves (32 bytes × N) plus nlist
centroids and the inverted-list overhead. Roughly 1.2× IndexBinaryFlat.

**Disk-resident / mmap.** "Only IVF indices can be memory-mapped in faiss"
([Faiss Indexes-that-do-not-fit-in-RAM](https://github.com/facebookresearch/faiss/wiki/Indexes-that-do-not-fit-in-RAM)).
At 1M×32 bytes = 32 MB this is irrelevant for us, but worth knowing if
we ever move to 1024-bit hashes or per-frame indexing of 12M points.

**Incremental updates.** `add()` after the initial `train()` is O(1) per
vector; the centroids stay fixed. After many adds (10×+ original size)
recall starts to drift because the partitioning becomes stale — re-train
on a fresh sample. For our nightly-rescan use case that's never an issue.

### 3.3 BK-tree — still good, but in 2025 not the first pick for Hamming

The classic Burkhard-Keller tree partitions points by integer distance to
a pivot. Build is O(N log N) sequential distance computations. Query at
radius r prunes children whose `|d(pivot, q) - r_child| > r`. Hamming is
the canonical metric for which this works.

**Why we used to recommend it.** Pure-Python implementation (`pybktree`)
is ~150 LOC, MIT-licensed, no native build. Drop-in.

**Why we don't recommend it as #1 anymore.**

1. **Faiss Flat with AVX2 popcount is now competitive on raw throughput.**
   A 50k-point exhaustive XOR+popcount over 256-bit codes takes ~1 ms
   on a single core. A BK-tree with r=14 on 64-bit codes traverses
   1–5% of the database per query ([metric-tree benchmark](https://daniel-j-h.github.io/post/nearest-neighbors-in-metric-spaces/)),
   i.e. visits ~500–2500 nodes; each visit is a Hamming evaluation +
   pointer chases. On 256-bit codes the prune ratio is *worse* because
   distances are spread over 0–256 not 0–64.
2. **Build cost matters during incremental scans.** Faiss `add` is a memcpy;
   BK-tree `add` walks the tree and does ~log N Hamming evals. At
   100 new videos per scan it doesn't matter, but the bigger N gets the
   wider the build-cost gap.
3. **Pure-Python pybktree is too slow.** The Github issue
   ["misleading time complexity"](https://github.com/benhoyt/pybktree/issues/5)
   discusses how the structure degrades when many distances are equal.
   `cppbktree.BKTree64` is **56× faster to build** and 6.7–383× faster to
   query, but it's 64-bit-only and on PyPI in a single-author wheel that
   isn't widely tested on Windows.
4. **The aggregate-hash strategy makes the tree's structural advantage
   smaller.** Once we hash 12 frames into one vector per video, the
   "database" is N ~ 50k points, not 12N ~ 600k. At that size, even
   linear scan is fast enough; the indexing dividend is small.

**When BK-tree IS the right answer for this project:**
- We refuse new C-dependency footprint (no faiss-cpu wheel allowed).
  `pybktree` is then the only pure-Python option.
- We want exact answers and want the dependency footprint to be a single
  pure-Python file. (FAISS is itself C++/SWIG.)

**Best modern BK-tree implementation: `pynear`'s `BKTreeBinaryIndex`.**
Both VP- and BK- variants share the same library, AVX2 popcount, works on
arbitrary `uint8` arrays so 256-bit is fine.

```python
from pynear import BKTreeBinaryIndex
bk = BKTreeBinaryIndex()
bk.set(aggregate_hashes_uint8)        # shape (N, 32)
neighbors, distances = bk.find_threshold(query, threshold=14)
```

### 3.4 VP-tree — quasi-tied with BK for our use case

VP-trees partition on continuous metrics; for discrete Hamming they work
but lose some of their structural elegance. The published comparison on
64-bit hashes at radius up to 16 shows VP-tree **build is ~10× slower than
BK-tree**; query is comparable or slightly slower at small radii and
slightly *faster* at very small radii (r ≤ 4)
([metric-tree-demo benchmark](https://github.com/depp/metric-tree-demo),
[INNOQ blog](https://www.innoq.com/en/blog/looks-the-same-to-me/)).

For our threshold (r=14 on 256-bit, equivalent in spirit to r≈3–4 on
64-bit if we re-calibrate), the practical difference between BK and VP
is **lost in the noise of process overhead**. Pick BK because pynear
already exposes both via one API.

### 3.5 Multi-Index Hashing (Norouzi et al., CVPR'12)

The cleverest exact-Hamming-r index ever published. Splits each b-bit
hash into m disjoint substrings of b/m bits each, indexes each substring
in a separate exact hash table. For a query with radius r, by pigeonhole
**at least one** of the m substrings must match within radius ⌊r/m⌋.
So you enumerate all bitstrings within ⌊r/m⌋ of each query substring,
look them up in the corresponding table, union the results, and verify
the full b-bit Hamming distance on the candidate set.

**Complexity.** For uniformly distributed b-bit codes and r/b small,
query time is **sub-linear in N**. The original paper reports speedups
up to 100× over linear scan at b=64, r ≤ 25, N ≤ 1 billion
([Norouzi 2014 paper](https://www.cs.toronto.edu/~norouzi/research/papers/multi_index_hashing.pdf)).

**Library options.**
- **`pyMIH`** (AiLECS) — Python implementation of Norouzi's algorithm.
  Production-ready for the original paper's regime but not heavily
  maintained.
- **`mih-rs`** — Rust implementation with a Python binding planned but
  not on PyPI as of writing.
- **`faiss.IndexBinaryMultiHash`** — Faiss's own MIH. Configure with
  the factory string `"BHash{nhash}x{b_per_table}"`, e.g. `"BHash4x64"`.

**Why we're not putting it in the top 3 despite the math being beautiful.**
Faiss's own binary-index benchmark concludes:
> "While increasing the number of hashtables decreases the number of
> distances to compute, it increases the number of random accesses
> proportionally, and each distance computation requires a random
> access — therefore, in this case, MultiHash may not be an attractive
> solution."
> ([binary-hashing-index-benchmark](https://github.com/facebookresearch/faiss/wiki/Binary-hashing-index-benchmark))

In other words: on modern memory hierarchies, the cache-miss cost of
m random lookups dominates the savings from skipping distance
computations, *for our radii (r ≤ 14 on 256-bit ≈ r ≤ 14/256 = 5.5%
of bits)*. MIH pulls ahead at larger N and larger r/b — neither of
which is our regime.

**When MIH IS the right answer for this project:**
- We move to 64-bit aggregate hashes (e.g. `imagehash.phash(hash_size=8)`)
  to make r/b smaller (r=14 of 64 = 22% → MIH degrades; r=4 of 64 = 6.3%
  → MIH wins) and want a single global structure across all videos
  including transitions across duration boundaries.
- Catalogue grows past 1M videos.

### 3.6 HNSW for binary — works, but not natively in Python wheels

HNSW is the king of float ANN. For Hamming:
- **`faiss.IndexBinaryHNSW`** exists. Factory string `"BHNSW16"`.
  Query is sub-millisecond. Returns kNN, not range — for our radius
  query we'd ask for `k=N_in_bucket` and filter by distance, which
  partially defeats the point.
- **`hnswlib`** (the standalone one not in Faiss) does **not** ship
  Hamming distance support in the upstream wheel. Issue #535 still open
  as of 2025. A fork (`HaoZeSun2016/HNSW-HAMMING`) adds it for uint32
  vectors but it's unmaintained.
- The Rust port `hnswlib-rs` supports Hamming. Not Python-callable
  without effort.

**Verdict.** Use `faiss.IndexBinaryHNSW` only if we move toward kNN-style
"top-K nearest" queries (e.g. "show me the 10 most similar videos to
this one"). For range-search at fixed r=14, IVF or Flat is simpler and
recall is exact.

### 3.7 LSH (single- and multi-probe)

Faiss's `IndexBinaryHash` is the binary specialization of LSH: pick the
top-b bits as the hash key, look up the bucket, optionally explore
buckets within `nflip` bit-flips of the query bucket. The published
benchmark calls this "very competitive with IVF" at small radii — at
r=15 with 50M×256-bit it beats IVF on distance evaluations.

**`falconn-python`** does multi-probe cosine LSH on float vectors — not
a natural fit for binary Hamming; would require converting bits to ±1
floats.

**Verdict.** `IndexBinaryHash` is a perfectly reasonable scale-up if
IVF disappoints. Same Python dependency, same `range_search` API.
Slightly more memory than IVF (the hash tables).

### 3.8 Disk-resident vs in-memory

- **256-bit hash, 1 per video:**
  - 1M videos = 32 MB. Negligible. Always in RAM.
- **256-bit hash, 12 per video** (no aggregation):
  - 1M videos = 384 MB. Still RAM-friendly.
- **CLIP/DINO embeddings, 384–512 float32 per video:**
  - 1M videos = 1.5–2 GB. Starts to matter.

At our current 50k × 32 byte scale, **no disk strategy is required**.
The thumbnails directory dwarfs the index. Mmap (`IO_FLAG_MMAP`) is a
single-line escape hatch in Faiss if we ever cross the RAM threshold —
only IVF supports it, and only after `write_index`/`read_index`.

### 3.9 Incremental updates after a re-scan

The current pipeline already does the right thing at the *data* layer
(`FileCache` keyed by `(file_path, file_size, mtime_ns)`). The ANN
question is: when 100 new videos arrive in a re-scan, do we rebuild
the whole index?

| Structure | "Add 100 new vectors" cost | "Remove deleted vectors" cost |
|-----------|-----------------------------|--------------------------------|
| `IndexBinaryFlat` | 100 byte-copies (microseconds) | `remove_ids()` linear in N |
| `IndexBinaryIVF` | 100 quantizer probes + 100 list appends (sub-ms) | `remove_ids()` walks lists |
| `IndexBinaryHNSW` | 100 graph insertions (ms-scale; graph modification) | `mark_deleted` (lazy) |
| `cppbktree` | rebuild (the lib does not expose insert; `pybktree.add` does, but the tree drifts unbalanced) | rebuild |
| `pyMIH` | append-only by design | rebuild |

**Recommendation for our workflow:** any time we run a re-scan that
touches a particular duration-bucket, just rebuild that bucket's
`IndexBinaryFlat` from scratch. At ~500 entries × 32 bytes that's
16 KB and a microsecond. The "incremental" question only matters once
we go global; at that point `IndexBinaryIVF.add()` is the answer.

---

## 4. Concrete recommendation by library size

### 10k videos (today's "small" libraries)

**Do nothing fancy.** The current O(n²) within-bucket loop is fine —
on a 200-video bucket it's 19,900 numpy XOR+popcount calls, each on a
12×12 distance matrix. Total ~50 ms.

If you want a single change with no regressions: precompute a per-video
**bitwise-median aggregate hash** and use `numpy.unpackbits` +
`np.count_nonzero(a^b)` in a single batched broadcast — that's
literally a vectorized linear scan and beats almost everything at this
scale.

### 50k videos (today's "large" libraries)

**`faiss.IndexBinaryFlat` per duration bucket** + 12-frame best-match
verifier on the shortlist (the snippet in §3.1). The shape of the
pipeline doesn't change; only the within-bucket comparator. Estimated
end-to-end matching speedup vs current O(n²) loop: **20–60×**, dominated
by the reduction in the number of `compare_hash_sets` calls — only the
~1% of pairs that are *plausible* duplicates get the expensive verifier.

The wall-clock impact is modest because, per the existing
`algorithmic-improvements.md`, frame extraction (not the matching loop)
dominates total scan time. The matching loop becomes ~5–10 seconds
instead of ~minutes on a 50k library — meaningful, but no longer the
bottleneck.

### 100k–500k videos

**Drop duration bucketing as the ANN organizer; use global
`faiss.IndexBinaryIVF`** with `nlist≈√N`, `nprobe=16`, `radius=21`.
Duration grouping becomes a *post-filter* applied to the ANN's
candidate pairs (cheap: dict lookup on (i,j) duration_bucket equality).

Why drop the duration buckets here:
- Once N gets large, the per-bucket overhead (one Faiss index per
  bucket, ~thousands of buckets) overwhelms the per-query savings.
- A single global index is simpler to persist and to update.
- The post-filter on duration is O(pairs × 1) and pairs are now sparse.

Estimated total matching time at 500k: **5–15 seconds**, vs intractable
for naive O(n²).

### 1M+ videos

Three things change:

1. **Hash size — drop to 64-bit aggregate hashes.** A 256-bit hash with
   threshold 14 is a 5.5% bit-error tolerance. At 64 bits that's 3.5
   bits which is far too tight; either reduce hash_size to 8 (= 64 bits)
   and re-calibrate the threshold to ~4, or stay at 256-bit. The MIH
   math favors 64-bit hashes with small r.
2. **Switch to `faiss.IndexBinaryIVF` with `nlist≈4·√N`** (~4000 cells)
   or **`faiss.IndexBinaryHash` (BHash32)** if the recall plot favors
   it on your real data. Run the Faiss-style operating-point sweep
   once with a held-out set.
3. **Memory-map the index** (`faiss.read_index(..., IO_FLAG_MMAP)`)
   if we exceed available RAM. At 1M × 32 bytes (per-video aggregate)
   we're at 32 MB and don't need this; at 1M × 12 × 32 (per-frame) we
   are at 384 MB and still don't need it.

If we move to **CLIP/DINO embeddings** (covered in section 3 of
`algorithmic-improvements.md`), the binary indexes go away entirely and
HNSW-Flat (`IndexHNSWFlat`) over the 384–512-d float embedding is the
right structure.

---

## 5. Hybrid strategy

No single structure is optimal because the problem is multi-stage:

```
Stage A: cheap filter (duration grouping)
Stage B: ANN candidate-pair generation
Stage C: expensive verifier (12-frame best-match)
Stage D: audio fallback for pairs Stage C rejected
```

The current pipeline collapses B and C into one O(n²) loop. The
recommendation is to **separate them**:

| Stage | Structure | Cost | What it filters |
|-------|-----------|------|------------------|
| A | sorted-anchor duration grouping (existing) | O(N log N) | duration mismatches |
| B | `IndexBinaryFlat` on aggregate hash, `range_search(r=21)` | <1 ms per bucket | bitwise-unrelated content |
| **C** | existing `compare_hash_sets` 12×12 best-match | O(144) per **candidate pair only** | precise visual match |
| D | audio fingerprint (existing or future Chromaprint) | O(64) per **rejected pair only** | audio-only matches |

**Why this is optimal:** stage C is the only "expensive" step per pair,
and it's now only invoked for ANN candidates. The verifier preserves the
fps/trim robustness the project specifically designed for; the ANN
provides the sub-quadratic candidate generation it currently lacks.

**Quantification.** For a 200-video duration bucket:
- Current: 19,900 calls to `compare_hash_sets`, each 144-cell.
- With Flat: ~50–500 candidate pairs (depends on how strongly clustered
  the bucket is), so 50–500 verifier calls. **40×–400× fewer verifications.**

---

## 6. Code sketch for the recommended approach

Drop-in replacement for `find_duplicates_in_group()` in
`backend/services/comparator.py`. No data-flow change above or below.

```python
# backend/services/comparator.py — proposed find_duplicates_in_group
import faiss
import numpy as np
from typing import List, Dict
from collections import defaultdict
from services.hasher import compare_hash_sets, _hex_to_bits
from services.audio_fingerprint import compare_audio_fingerprints


HASH_BITS = 256             # imagehash.phash(hash_size=16)
HASH_BYTES = HASH_BITS // 8 # 32
RADIUS_MULTIPLIER = 1.5     # widen ANN radius vs verifier threshold


def _aggregate_hash(hash_hex_list: List[str]) -> np.ndarray:
    """Bitwise-majority across the 12 frame hashes → 32-byte aggregate."""
    bit_arrs = []
    for h in hash_hex_list:
        b = _hex_to_bits(h)
        if b is not None and len(b) == HASH_BITS:
            bit_arrs.append(b)
    if not bit_arrs:
        return np.zeros(HASH_BYTES, dtype=np.uint8)
    stacked = np.stack(bit_arrs)               # (k, 256)
    majority = (stacked.sum(axis=0) >= (len(bit_arrs) + 1) // 2).astype(np.uint8)
    return np.packbits(majority)               # (32,)


def find_duplicates_in_group(
    videos: List[dict],
    hash_threshold: int = 14,
    audio_threshold: float = 80.0,
) -> List[List[dict]]:
    if len(videos) < 2:
        return []

    # ── 1. Build the Faiss binary index for this duration bucket ──
    aggs = np.stack([_aggregate_hash(v.get("hashes") or []) for v in videos])
    index = faiss.IndexBinaryFlat(HASH_BITS)
    index.add(aggs)

    radius = int(hash_threshold * RADIUS_MULTIPLIER)
    lims, D, I = index.range_search(aggs, radius=radius + 1)  # strictly <

    # ── 2. Candidate pairs from ANN, deduplicated (i < j) ──
    candidates = set()
    for i in range(len(videos)):
        for k in range(lims[i], lims[i + 1]):
            j = int(I[k])
            if j > i:
                candidates.add((i, j))

    # ── 3. Existing 12×12 verifier on the candidate shortlist ──
    n = len(videos)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (i, j) in candidates:
        if not _file_size_compatible(videos[i], videos[j]):
            continue
        h1 = videos[i].get("hashes") or []
        h2 = videos[j].get("hashes") or []
        if not (h1 and h2):
            continue
        is_similar, sim = compare_hash_sets(h1, h2, hash_threshold)
        if is_similar:
            union(i, j)
            videos[i].setdefault("_similarities", {})[j] = sim
            videos[j].setdefault("_similarities", {})[i] = sim
            videos[i].setdefault("_match_methods", {})[j] = "video"
            videos[j].setdefault("_match_methods", {})[i] = "video"

    # ── 4. Audio fallback for pairs the ANN rejected (existing behaviour) ──
    # Critical: we must still consider non-candidate pairs whose audio matches.
    # Option A (safest): keep the existing O(n²) audio-only fallback loop.
    # Option B (faster): only run audio fallback on pairs that *almost* matched
    #                    the ANN radius (e.g. radius+5). Tune empirically.
    for i in range(n):
        ai = videos[i].get("audio_fp") or []
        if not ai:
            continue
        for j in range(i + 1, n):
            # If we already matched via video, skip
            if find(i) == find(j):
                continue
            aj = videos[j].get("audio_fp") or []
            if not aj:
                continue
            audio_sim = compare_audio_fingerprints(ai, aj)
            if audio_sim >= audio_threshold:
                union(i, j)
                videos[i].setdefault("_similarities", {})[j] = audio_sim
                videos[j].setdefault("_similarities", {})[i] = audio_sim
                videos[i].setdefault("_match_methods", {})[j] = "audio"
                videos[j].setdefault("_match_methods", {})[i] = "audio"

    # ── 5. Collect groups (existing logic) ──
    group_map: Dict[int, List[dict]] = defaultdict(list)
    for i in range(n):
        if videos[i].get("hashes") or videos[i].get("audio_fp"):
            group_map[find(i)].append(videos[i])
    return [g for g in group_map.values() if len(g) > 1]
```

The dependency is a single line in `requirements.txt`:
```
faiss-cpu>=1.7.4
```

Faiss wheels for Windows x86_64 Python 3.10+ are on PyPI as
`faiss-cpu`. No CUDA toolkit required for the binary path.

---

## 7. Risks

### 7.1 Recall loss from the aggregate hash

The single biggest risk. Bitwise-majority across 12 frames hides
duplicates that share only *some* frames (different intros, different
endings). The current pipeline catches these because best-match works
position-independently across the 12-frame sets.

**Mitigations, in increasing complexity:**
1. **Widen the ANN radius.** Setting `radius = threshold × 1.5` (21 for
   threshold=14) catches pairs whose aggregate hashes diverge moderately
   but whose underlying frames still pair up well. The verifier filters
   false positives back out.
2. **Index each frame, not each video.** Insert 12 hashes per video into
   the index with `add_with_ids(np.array([video_id]*12))`. Then any pair
   that shares at least one near-matching frame is a candidate. Costs
   12× the memory but the index is still ~ a few MB. This is the
   highest-recall option and the closest to current behavior.
3. **Index both: 1 aggregate + 12 frames.** Use the aggregate index for
   the bulk; if a video has zero aggregate-radius matches, query the
   per-frame index as a safety net.

For the first deployment, option 1 is the right choice. If a
regression-test set surfaces missed duplicates, escalate to option 2.

### 7.2 Build/update cost on incremental scans

`IndexBinaryFlat.add()` is a memcpy, so this is essentially free.

The realistic concern is the **`compare_hash_sets` verifier** during
adds, not the ANN. If 100 new videos arrive in a duration bucket of 500
existing entries, the ANN produces ~10–50 candidate pairs to verify —
microseconds compared to the verifier itself.

### 7.3 New dependency

`faiss-cpu` adds ~30 MB to the wheel size and pulls in OpenBLAS. It's
maintained by Meta, has been around since 2017, ships wheels for
Linux/macOS/Windows × Python 3.8–3.12, and is the de-facto standard for
ANN in Python. Low risk.

If absolute zero-new-deps is required, fall back to `pynear` (single
pip install, ~5 MB) for the BK-tree route. Two-thirds of the speedup,
none of the AVX2 popcount kernel optimizations.

### 7.4 Threshold/radius mis-calibration

The recommended `radius = hash_threshold * 1.5` is empirical. On a
specific dataset it might be too tight (recall drop) or too loose
(verifier load creeps up). Run `diagnose_pair.py` over the existing
held-out set to confirm before flipping the default.

### 7.5 Index state divergence across re-scans

If we ever persist the binary index (`faiss.write_index`) and reload
it between scans, the index can drift from the `FileCache` if a write
fails mid-scan. Mitigation: don't persist. Rebuild on every scan from
the `FileCache.perceptual_hashes` blob — it's microseconds.

### 7.6 Faiss `range_search` "<" semantics

Faiss range_search returns vectors with distance *strictly less than*
radius. Off-by-one matters when the test set has exact-threshold
pairs. Pass `radius = threshold + 1` to include equality, or
`radius = threshold * 1.5 + 1` for the widened version.

---

## 8. What we explicitly did *not* recommend, and why

| Option | Reason for rejection |
|--------|----------------------|
| `hnswlib` directly | No upstream Hamming support; community fork unmaintained. |
| `datasketch.MinHashLSH` | Set similarity, not bit-Hamming. Awkward fit. |
| `annoy` | Float-only, no Hamming kernel. |
| `pgvector` | We're on SQLite; not switching. |
| `nmslib` non-Faiss | Same Hamming-support gap as hnswlib; project less active. |
| Roll-your-own LSH banding | Faiss `IndexBinaryHash` is the same algorithm, vectorized, with `range_search`. |
| ResNet / CLIP embeddings | Separate concern (covered in §3 of `algorithmic-improvements.md`). When we go there, the binary index gets replaced by `IndexHNSWFlat` on floats. |
| Switch to 64-bit pHashes | r=14 of 64 bits = 22% bit-error tolerance, way too loose. Would need full re-calibration of the threshold. Out of scope. |

---

## 9. Appendix — Faiss factory string cheat-sheet for binary

The `faiss.index_binary_factory(d, "<spec>")` shortcut accepts:

| Factory string | Maps to | Notes |
|----------------|---------|-------|
| `"BFlat"` | `IndexBinaryFlat` | exact, brute force, AVX2 popcount |
| `"BIVF{nlist}"` | `IndexBinaryIVF` | e.g. `"BIVF256"` |
| `"BHash{b}"` | `IndexBinaryHash` | e.g. `"BHash32"` uses top 32 bits |
| `"BHash{nhash}x{b}"` | `IndexBinaryMultiHash` | e.g. `"BHash4x32"`, 4 tables of 32 bits |
| `"BHNSW{M}"` | `IndexBinaryHNSW` | e.g. `"BHNSW16"` |

All accept `range_search` since Faiss 1.6.3.

---

## 10. References

- **Faiss binary indexes wiki:**
  https://github.com/facebookresearch/faiss/wiki/Binary-indexes
- **Faiss binary-hashing-index-benchmark:**
  https://github.com/facebookresearch/faiss/wiki/Binary-hashing-index-benchmark
- **Faiss IndexBinaryFlat C++ doc:**
  https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexBinaryFlat.html
- **Faiss IndexBinaryIVF C++ doc:**
  https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexBinaryIVF.html
- **Faiss IndexBinaryMultiHash C++ doc:**
  https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexBinaryMultiHash.html
- **Faiss Special-operations-on-indexes (range_search return format):**
  https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes
- **Faiss Indexes-that-do-not-fit-in-RAM (mmap):**
  https://github.com/facebookresearch/faiss/wiki/Indexes-that-do-not-fit-in-RAM
- **Faiss library paper (Douze et al. 2024):**
  https://arxiv.org/abs/2401.08281
- **Norouzi et al., Fast Exact Search in Hamming Space with Multi-Index Hashing:**
  https://www.cs.toronto.edu/~norouzi/research/papers/multi_index_hashing.pdf
- **Norouzi MIH reference C++ implementation:**
  https://github.com/norouzi/mih
- **pyMIH Python implementation of MIH:**
  https://github.com/AiLECS/pyMIH
- **pybktree (pure-Python BK-tree):**
  https://github.com/benhoyt/pybktree
- **cppbktree (C++/Python BK-tree, BKTree64):**
  https://github.com/mxmlnkn/cppbktree
- **pynear (VP-tree + BK-tree, AVX2 popcount):**
  https://github.com/pablocael/pynear
- **hnswlib Hamming-distance issue (still open 2025):**
  https://github.com/nmslib/hnswlib/issues/535
- **FALCONN LSH library:**
  https://github.com/FALCONN-LIB/FALCONN
- **datasketch MinHashLSH docs:**
  https://ekzhu.com/datasketch/lsh.html
- **pgvector 0.7.0 Hamming-distance support (2024):**
  https://www.postgresql.org/about/news/pgvector-070-released-2852/
- **PDQ / TMK+PDQF (Facebook video hash, 256-bit) paper:**
  https://arxiv.org/abs/1912.07745
- **Nearest neighbors in metric spaces (BK-tree vs VP-tree practical notes):**
  https://daniel-j-h.github.io/post/nearest-neighbors-in-metric-spaces/
- **It All Looks the Same to Me (INNOQ — BK/VP empirical comparison):**
  https://www.innoq.com/en/blog/looks-the-same-to-me/
- **PyImageSearch — VP-tree image hashing tutorial:**
  https://pyimagesearch.com/2019/08/26/building-an-image-hashing-search-engine-with-vp-trees-and-opencv/
