# Quality Scoring

How the system picks the "best" file in a duplicate group.

Implemented in [`backend/services/quality_scorer.py`](../backend/services/quality_scorer.py).

## The score

Each video gets a score in `[0, 100]` computed as a weighted sum of five factors:

```
score = (
    resolution_norm  * RESOLUTION_WEIGHT   # 0.40
  + bitrate_norm     * BITRATE_WEIGHT      # 0.25
  + codec_score      * CODEC_WEIGHT        # 0.15
  + file_size_norm   * FILE_SIZE_WEIGHT    # 0.10
  + fps_norm         * FPS_WEIGHT          # 0.10
) * 100
```

Weights live in [`config.py`](../backend/config.py) and **must sum to 1.0**. The Settings API allows them to be edited at runtime.

## Per-factor normalisation

| Factor | Source field | Normalisation |
|---|---|---|
| Resolution | `width × height` | `min(pixels / 8_294_400, 1.0)` (4K reference) |
| Bitrate | `bitrate` (bps) | `min(bitrate / 50_000_000, 1.0)` (50 Mbps reference) |
| Codec | `video_codec` | Lookup in `CODEC_SCORES` (default 0.5 if unknown) |
| File size | `file_size` (bytes) | `min(size / 10_737_418_240, 1.0)` (10 GiB reference) |
| FPS | `fps` | `min(fps / 120, 1.0)` |

### Codec table

```python
CODEC_SCORES = {
    "hevc":  1.0,  "h265":  1.0,
    "av1":   1.0,
    "vp9":   0.85,
    "h264":  0.8,  "avc":   0.8,
}
```

The lookup is `for key, score in CODEC_SCORES.items(): if key in codec_lower: return score`. So `"avc1"` matches `"avc"` (= 0.8). Anything not found returns 0.5.

This favours newer, more efficient codecs — a 2 Mbps HEVC and a 4 Mbps H.264 of the same content will rank similarly even though the H.264 has higher bitrate.

## Group-relative normalisation

When a group context is provided (always the case during scans), each factor is **renormalised against the group max**, not against absolute references:

```python
if all_videos_in_group and len(all_videos_in_group) > 1:
    group_max_res = max((w * h) for v in group)
    if group_max_res > 0:
        resolution_norm = pixels / group_max_res
    # ... same for bitrate, file_size, fps
```

This means: if the highest-resolution file in a group is 1080p, that file gets `resolution_norm = 1.0`, not `0.25` (which is what the absolute 4K-referenced normalisation would yield). The "best" file always pegs to 1.0 on at least one factor.

The codec score is **not** group-relative — it stays absolute. This is intentional: if the only options are H.264 and H.264, neither should get a free 1.0.

## Ranking

`rank_group(videos)`:

1. Computes `quality_score` for every video, passing the full group as `all_videos_in_group`.
2. Sorts by `quality_score` descending.
3. Marks `videos[0].is_best_quality = True` and the rest False.
4. Returns the sorted list.

The result is persisted to `VideoFile.quality_score` and `VideoFile.is_best_quality`, and `DuplicateGroup.best_file_id` points at the top one.

## Wasted space calculation

`calculate_wasted_space(videos)`:

```python
sizes = sorted([v.file_size for v in videos], reverse=True)
return sum(sizes[1:])
```

Sum of all but the **largest file**. This is what's reported as `total_wasted_space` per group and `recoverable_space` per scan.

A subtle thing: this is "largest file," **not** "best-quality file." The largest is usually but not always the best. The figure is therefore conservative (under-reports recoverable space when the best file isn't the largest). That's deliberate — overstating recoverable space is worse for user trust than understating it.

## Auto-clean uses `is_best_quality`

When the user clicks "auto-clean" (`POST /api/auto-clean`), the actual deletion rule is:

```python
if not video.is_best_quality and not video.is_deleted:
    move_to_trash(video.file_path, scan_root)
```

So auto-clean follows the **quality ranking**, not the "largest file" rule used for wasted-space estimation. They will agree most of the time but not always.

## Tuning

The defaults are tuned to favour visual quality over file size:

- Resolution dominates (40%): a 4K version always beats a 1080p version unless they're encoded very differently.
- Bitrate (25%) and codec (15%) together (40%) reward efficient high-bitrate encodes.
- File size (10%) is a tie-breaker — among visually equivalent files, bigger usually means less aggressive compression.
- FPS (10%) catches the rare case of 30 fps vs 60 fps versions.

To favour **storage efficiency** instead (prefer smaller files all else being equal), invert the file_size weight or lower it. To favour **HEVC strongly**, raise `CODEC_WEIGHT` to ~0.30 and lower `BITRATE_WEIGHT` proportionally.

The weights are exposed at `PUT /api/settings` and editable in the Settings UI page. They do not require a restart but only affect **future** scans; existing rows keep their previously computed scores.
