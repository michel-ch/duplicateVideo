"""FFprobe metadata extraction for video files."""

import json
import subprocess
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor

from config import settings


def _extract_metadata_sync(file_path: str) -> Dict[str, Any]:
    """
    Extract video metadata using ffprobe (synchronous, for process pool).
    """
    try:
        cmd = [
            settings.FFPROBE_PATH,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            "-show_entries", "stream_side_data=rotation",
            str(file_path)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )

        if result.returncode != 0:
            return {"error": f"ffprobe failed: {result.stderr.decode('utf-8', errors='replace')[:200]}"}

        probe_data = json.loads(result.stdout.decode('utf-8'))
        return _parse_probe_data(probe_data, file_path)

    except subprocess.TimeoutExpired:
        return {"error": "ffprobe timed out"}
    except json.JSONDecodeError:
        return {"error": "Invalid ffprobe output"}
    except FileNotFoundError:
        return {"error": "ffprobe not found. Ensure ffmpeg is installed and in PATH."}
    except Exception as e:
        return {"error": str(e)}


def _parse_probe_data(data: dict, file_path: str) -> Dict[str, Any]:
    """Parse ffprobe JSON output into structured metadata."""
    result = {
        "file_path": file_path,
        "duration": None,
        "width": None,
        "height": None,
        "bitrate": None,
        "video_codec": None,
        "audio_codec": None,
        "fps": None,
        "audio_channels": None,
        "audio_sample_rate": None,
        # SAR / rotation — used by hasher to skip a redundant ffprobe call
        "sar_num": 1,
        "sar_den": 1,
        "rotation": 0,
    }

    # Duration from format
    fmt = data.get("format", {})
    if "duration" in fmt:
        try:
            result["duration"] = float(fmt["duration"])
        except (ValueError, TypeError):
            pass

    # Overall bitrate
    if "bit_rate" in fmt:
        try:
            result["bitrate"] = int(fmt["bit_rate"])
        except (ValueError, TypeError):
            pass

    # Parse streams
    streams = data.get("streams", [])
    for stream in streams:
        codec_type = stream.get("codec_type", "")

        if codec_type == "video" and result["video_codec"] is None:
            result["video_codec"] = stream.get("codec_name", "unknown")
            result["width"] = stream.get("width")
            result["height"] = stream.get("height")

            # Frame rate
            r_frame_rate = stream.get("r_frame_rate", "0/1")
            try:
                num, den = r_frame_rate.split("/")
                if int(den) > 0:
                    result["fps"] = round(int(num) / int(den), 2)
            except (ValueError, ZeroDivisionError):
                pass

            # Duration from video stream if not in format
            if result["duration"] is None and "duration" in stream:
                try:
                    result["duration"] = float(stream["duration"])
                except (ValueError, TypeError):
                    pass

            # Bitrate from video stream
            if result["bitrate"] is None and "bit_rate" in stream:
                try:
                    result["bitrate"] = int(stream["bit_rate"])
                except (ValueError, TypeError):
                    pass

            # SAR (Sample Aspect Ratio) — e.g. "81:256", "1:1"
            sar_str = stream.get("sample_aspect_ratio", "1:1")
            if sar_str and ":" in sar_str:
                try:
                    sn, sd = sar_str.split(":")
                    sn, sd = int(sn), int(sd)
                    if sn > 0 and sd > 0:
                        result["sar_num"] = sn
                        result["sar_den"] = sd
                except (ValueError, TypeError):
                    pass

            # Rotation from side_data_list
            for sd in stream.get("side_data_list", []):
                rot = sd.get("rotation")
                if rot is not None:
                    result["rotation"] = int(float(rot)) % 360

            # Rotation from tags (fallback)
            tags = stream.get("tags", {})
            rot = tags.get("rotate", tags.get("ROTATE"))
            if rot and result["rotation"] == 0:
                result["rotation"] = int(float(rot)) % 360

        elif codec_type == "audio" and result["audio_codec"] is None:
            result["audio_codec"] = stream.get("codec_name", "unknown")
            result["audio_channels"] = stream.get("channels")
            try:
                result["audio_sample_rate"] = int(stream.get("sample_rate", 0))
            except (ValueError, TypeError):
                pass

    return result


async def extract_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata asynchronously using process pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _extract_metadata_sync, file_path)


async def extract_metadata_batch(file_paths: list, max_concurrent: int = 4) -> list:
    """Extract metadata for multiple files with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def _extract(path):
        async with semaphore:
            return await extract_metadata(path)

    tasks = [_extract(p) for p in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return [
        r if not isinstance(r, Exception) else {"error": str(r), "file_path": file_paths[i]}
        for i, r in enumerate(results)
    ]
