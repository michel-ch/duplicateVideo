"""Quality scoring and ranking for duplicate video files."""

from typing import List, Dict, Optional
from config import settings


def get_codec_score(codec_name: Optional[str]) -> float:
    """Get quality score for a video codec."""
    if not codec_name:
        return 0.5

    codec_lower = codec_name.lower()

    for key, score in settings.CODEC_SCORES.items():
        if key in codec_lower:
            return score

    return 0.5


def compute_quality_score(video: dict, all_videos_in_group: List[dict] = None) -> float:
    """
    Compute a quality score for a video file.
    All factors are normalized to 0-1 scale.
    """
    # Resolution
    width = video.get("width") or 0
    height = video.get("height") or 0
    resolution_pixels = width * height

    # Normalize resolution (4K = 3840x2160 = 8,294,400 as max reference)
    max_resolution = 8_294_400
    resolution_norm = min(resolution_pixels / max_resolution, 1.0) if max_resolution > 0 else 0

    # Bitrate
    bitrate = video.get("bitrate") or 0
    # Normalize bitrate (50 Mbps as max reference)
    max_bitrate = 50_000_000
    bitrate_norm = min(bitrate / max_bitrate, 1.0) if max_bitrate > 0 else 0

    # Codec
    codec_score = get_codec_score(video.get("video_codec"))

    # File size
    file_size = video.get("file_size") or 0
    # Normalize file size (10 GB as max reference)
    max_file_size = 10_737_418_240
    file_size_norm = min(file_size / max_file_size, 1.0) if max_file_size > 0 else 0

    # FPS
    fps = video.get("fps") or 0
    # Normalize FPS (120 fps as max)
    max_fps = 120
    fps_norm = min(fps / max_fps, 1.0) if max_fps > 0 else 0

    # If we have the group context, normalize relative to the group
    if all_videos_in_group and len(all_videos_in_group) > 1:
        group_max_res = max(
            (v.get("width", 0) or 0) * (v.get("height", 0) or 0)
            for v in all_videos_in_group
        )
        group_max_bitrate = max(v.get("bitrate", 0) or 0 for v in all_videos_in_group)
        group_max_size = max(v.get("file_size", 0) or 0 for v in all_videos_in_group)
        group_max_fps = max(v.get("fps", 0) or 0 for v in all_videos_in_group)

        if group_max_res > 0:
            resolution_norm = resolution_pixels / group_max_res
        if group_max_bitrate > 0:
            bitrate_norm = bitrate / group_max_bitrate
        if group_max_size > 0:
            file_size_norm = file_size / group_max_size
        if group_max_fps > 0:
            fps_norm = fps / group_max_fps

    # Weighted sum
    score = (
        resolution_norm * settings.RESOLUTION_WEIGHT
        + bitrate_norm * settings.BITRATE_WEIGHT
        + codec_score * settings.CODEC_WEIGHT
        + file_size_norm * settings.FILE_SIZE_WEIGHT
        + fps_norm * settings.FPS_WEIGHT
    )

    return round(score * 100, 2)  # Scale to 0-100


def rank_group(videos: List[dict]) -> List[dict]:
    """
    Rank videos in a duplicate group by quality score.
    Returns videos sorted by quality_score descending, with
    is_best_quality set on the best one.
    """
    for v in videos:
        v["quality_score"] = compute_quality_score(v, videos)

    videos.sort(key=lambda v: v["quality_score"], reverse=True)

    # Mark best
    for i, v in enumerate(videos):
        v["is_best_quality"] = (i == 0)

    return videos


def calculate_wasted_space(videos: List[dict]) -> float:
    """Calculate total wasted space (sum of all but the largest file)."""
    if len(videos) <= 1:
        return 0

    sizes = sorted([v.get("file_size", 0) or 0 for v in videos], reverse=True)
    # Everything except the largest file is "wasted"
    return sum(sizes[1:])
