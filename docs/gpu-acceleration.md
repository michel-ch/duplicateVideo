# GPU Acceleration

NVIDIA CUDA / CUVID acceleration for FFmpeg decoding and pixel transforms. Implemented in [`backend/services/gpu_detector.py`](../backend/services/gpu_detector.py) and used throughout [`backend/services/hasher.py`](../backend/services/hasher.py).

## Detection

`detect_gpu()` runs **once at startup** (cached in a module-level `_cached_gpu_info`) and probes:

| Probe | Command | What it learns |
|---|---|---|
| 1 | `nvidia-smi --query-gpu=...` | GPU name, driver version, total/free VRAM |
| 2 | `ffmpeg -hwaccels` | Whether `cuda` is in the list |
| 3 | `ffmpeg -decoders` | Which `*_cuvid` decoders exist (h264_cuvid, hevc_cuvid, …) |
| 4 | `ffmpeg -filters` | Which `*_cuda` filters and `hwupload`/`hwdownload` exist |
| 5 | `ffmpeg -encoders` | NVENC encoders available (informational only — the app never encodes) |

If probe 1 fails, `available=False` and all subsequent probes are skipped. If probe 2 finds no `cuda` hwaccel, probes 3–5 are skipped. This means a working GPU but a non-CUDA-built FFmpeg degrades gracefully to CPU mode.

## Codec → decoder mapping

`detect_gpu()` populates `codec_decoder_map`:

```python
{
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "vp9":  "vp9_cuvid",
    ...
    # aliases
    "avc":  "h264_cuvid",
    "avc1": "h264_cuvid",
    "h265": "hevc_cuvid",
}
```

`gpu.get_decoder(codec)` returns the right CUVID decoder name or None. The hasher uses this when building the FFmpeg command line.

## Where GPU is used

**Frame extraction** in `hasher._build_frame_extract_cmd`:

```
ffmpeg -y \
  -hwaccel cuda \
  -hwaccel_output_format cuda \
  -c:v hevc_cuvid \
  -i FILE \
  -vf "fps=N/D,hwdownload,format=nv12,
       scale=iw*sar:ih,setsar=1,    # if anamorphic
       transpose=1,                  # if portrait
       scale=320:-2" \
  -q:v 2 -frames:v N OUT_%04d.jpg
```

Decoding happens on the GPU; pixel transforms run on the CPU after `hwdownload`. This is intentional — `scale_cuda` exists, but the SAR-then-transpose-then-scale chain is more reliable on CPU and the bottleneck is decode anyway.

**Thumbnail extraction** in `hasher._extract_thumbnail_sync` follows the same pattern, with one optimisation: if `scale_cuda` is in `gpu.cuda_filters`, the scale runs on GPU before download.

## Where GPU is *not* used

- **Audio fingerprinting** (`audio_fingerprint.py`) — pure FFmpeg audio decode. CPU-only.
- **Metadata extraction** (`metadata.py`) — ffprobe doesn't need a GPU.
- **Duplicate comparison** (`comparator.py`) — pure NumPy on hashes.
- **Quality scoring** — arithmetic.

So GPU acceleration only helps stages 2 and 3 of the pipeline, but those are by far the slowest stages on big datasets.

## CPU fallback

Inside `_extract_frames_sync`, if a GPU run produces zero frames (rare, but happens with pathological inputs), the function automatically retries with `codec=None`, which forces the CPU path. The user never sees the failure — the fallback is invisible.

## Concurrency

GPU mode raises the FFmpeg concurrency cap:

```python
if gpu_active:
    max_concurrent = options.get("max_concurrent", settings.GPU_MAX_CONCURRENT)  # 12
else:
    max_concurrent = options.get("max_concurrent", settings.MAX_CONCURRENT_FFMPEG)  # 8
```

Why higher: GPU decoding has lower per-task CPU cost, so more parallel ffmpeg invocations don't oversubscribe the CPU. The limit is set conservatively for an RTX 3060 Ti (8 GB VRAM); newer cards with more VRAM can go higher.

The semaphore is process-wide — not per-batch — so 12 ffmpeg subprocesses is the total active count, not per stage.

## Forcing CPU mode

Set `GPU_ENABLED = False` in [`config.py`](../backend/config.py) (or via `.env` `GPU_ENABLED=false`). This bypasses detection entirely. Useful for:

- Debugging output differences between GPU and CPU paths.
- Running on a machine where the GPU is needed for other work.
- Confirming a suspected GPU-decode bug.

## Docker

The Docker image is based on `nvidia/cuda:12.2.2-runtime-ubuntu22.04` so the runtime CUDA libs are present. The `docker-compose.yml` declares the NVIDIA runtime and binds the device files:

```yaml
runtime: nvidia
devices:
  - /dev/nvidia0:/dev/nvidia0
  - /dev/nvidiactl:/dev/nvidiactl
  - ...
```

This requires the **NVIDIA Container Toolkit** to be installed on the host. Without it, the container falls back to CPU mode.

## Status endpoint

`GET /api/gpu-status` returns the cached `GPUInfo` as JSON. The frontend Dashboard polls this on load to show a "GPU acceleration active: ⚡" badge. The data does not change after startup (detection runs once).
